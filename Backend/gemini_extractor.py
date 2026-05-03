import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from config import GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Model config — paid tier
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 100_000          # chars per chunk (was 28000 on free tier)
CHUNK_THRESHOLD = 200_000     # only split PDFs larger than this (was 60000)
MAX_WORKERS = 10              # parallel chunk workers (was 3)
TIMEOUT_CHUNK = 45            # seconds per chunk call (was 20)
TIMEOUT_SMALL = 45            # seconds for single small-PDF call (was 25)
TIMEOUT_UPLOAD = 90           # seconds for file-upload path (was 60)


# ---------------------------------------------------------------------------
# Retry helper — handles 429 and transient errors gracefully
# ---------------------------------------------------------------------------
def _call_with_retry(
    model: Any,
    contents: Any,
    timeout_s: int = TIMEOUT_SMALL,
    max_retries: int = 3,
) -> Any:
    """
    Calls model.generate_content with automatic retry on 429 rate-limit errors.
    On paid tier, recovery is fast (usually < 10s); we still respect the
    retry_delay hint from the error body when present.
    """
    for attempt in range(max_retries):
        try:
            return model.generate_content(
                contents,
                request_options={"timeout": timeout_s},
            )
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "quota" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                retry_after = 10  # paid tier recovers quickly
                m = re.search(r"retry in (\d+)", err_str)
                if m:
                    retry_after = int(m.group(1)) + 1
                print(
                    f"rate-limited. Retrying in {retry_after}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_after)
            else:
                raise
    raise Exception("Gemini: max retries exceeded.")


# ---------------------------------------------------------------------------
# Internal helpers (unchanged logic, same as original)
# ---------------------------------------------------------------------------

def _extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_articles(articles: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(articles, start=1):
        heading = (item.get("heading") or "").strip() or "No heading"
        subheading = (item.get("subheading") or "").strip()
        body_raw = item.get("body") or []
        body = [str(p).strip() for p in body_raw if str(p).strip()]
        full_text = (item.get("full_text") or "").strip()
        if not full_text:
            parts: List[str] = [heading]
            if subheading:
                parts.append(subheading)
            if body:
                parts.append("\n".join(body))
            full_text = "\n".join(parts).strip()

        normalized.append(
            {
                "article_id": item.get("article_id") or f"{prefix}_{i}",
                "heading": heading,
                "subheading": subheading,
                "body": body,
                "full_text": full_text,
                "column": item.get("column") or "full",
            }
        )
    return normalized


def _finish_reason_value(response: Any) -> Any:
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if hasattr(reason, "value"):
            return reason.value
        return reason
    except Exception:
        return None


def _response_text_safe(response: Any) -> str:
    """
    Avoids response.text quick accessor crashes when model returns no valid Part.
    """
    try:
        return (getattr(response, "text", "") or "").strip()
    except Exception:
        pass

    chunks: List[str] = []
    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                txt = getattr(part, "text", None)
                if txt:
                    chunks.append(str(txt))
    except Exception:
        return ""

    return "\n".join(chunks).strip()


def _extract_pdf_text_fast(file_path: str) -> str:
    """
    Fast native PDF text extraction (avoids slow multimodal upload path).
    """
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(file_path)
        pages: List[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def _build_prompt(resource_type: str, source_kind: str) -> str:
    return f"""
You are an extraction engine for scanned/printed documents.
Resource type: {resource_type}
Input source: {source_kind}

Extract all meaningful article/chapter units.
Handle noisy OCR and keep original language text.

Return strict JSON only (no markdown, no extra text) with this schema:
{{
  "full_text": "all extracted text merged in reading order",
  "articles": [
    {{
      "article_id": "stable_id",
      "heading": "title or chapter heading",
      "subheading": "optional subheading",
      "column": "left|right|full",
      "body": ["paragraph 1", "paragraph 2"],
      "full_text": "heading + subheading + body text"
    }}
  ]
}}

Rules:
- For newspapers/magazines: split by individual articles.
- For books: split by chapters/sections where possible.
- If only one unit is identifiable, return one article with column "full".
- Never return empty full_text.
- If policy prevents verbatim extraction, return best-effort structure using short snippets.
"""


def _build_fallback_prompt(resource_type: str, source_kind: str) -> str:
    return f"""
You are an extraction engine. Direct verbatim extraction was restricted by policy.
Resource type: {resource_type}
Input source: {source_kind}

Please provide a BEST-EFFORT structural extraction.
Instead of full text, provide:
1. Clear headings and subheadings.
2. Short 1-2 sentence summaries or key snippets for each paragraph.
3. The overall layout structure.

Return strict JSON only:
{{
  "full_text": "Summarized version of the document",
  "articles": [
    {{
      "article_id": "stable_id",
      "heading": "title",
      "body": ["Summary snippet 1", "Summary snippet 2"],
      "full_text": "Heading + Summaries"
    }}
  ]
}}
"""


def _should_skip_gemini_upload(resource_type: str, file_path: str) -> bool:
    """
    Gemini file-upload extraction often gets blocked for periodicals due to
    copyrighted-recitation policy. Newspapers now use fail-and-retry with a
    best-effort fallback prompt, so we only skip magazines for image uploads.
    """
    if file_path.lower().endswith(".pdf"):
        return False
    return resource_type in {"magazines"}


def _chunk_text_for_gemini(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into paragraph-aware chunks.
    Paid tier: default chunk size raised to 100k chars (was 28k on free tier).
    """
    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]
    if not paragraphs:
        return [clean[i: i + max_chars] for i in range(0, len(clean), max_chars)]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + 2
        if current and (current_len + para_len > max_chars):
            chunks.append("\n\n".join(current).strip())
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def _reindex_articles(
    articles: List[Dict[str, Any]], resource_type: str
) -> List[Dict[str, Any]]:
    prefix = "chapter" if resource_type == "books" else "article"
    reindexed: List[Dict[str, Any]] = []
    for i, item in enumerate(articles, start=1):
        row = dict(item)
        row["article_id"] = f"{prefix}_{i}"
        reindexed.append(row)
    return reindexed


def _structure_pdf_text_in_chunks(
    model: Any, resource_type: str, pdf_text: str
) -> Optional[Dict[str, Any]]:
    """
    Structure big PDFs by sending bounded chunks in parallel.
    Paid tier: chunk size = 100k, max_workers = 10, timeout = 45s.
    """
    chunks = _chunk_text_for_gemini(pdf_text, max_chars=CHUNK_SIZE)
    if not chunks:
        return None

    prompt = _build_prompt(resource_type, "plain_pdf_text_chunk")
    results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)

    def process_chunk(idx: int, text_chunk: str) -> Optional[Dict[str, Any]]:
        # Uses retry wrapper — handles transient 429s on paid tier
        response = _call_with_retry(
            model,
            [prompt, text_chunk],
            timeout_s=TIMEOUT_CHUNK,
        )
        finish_reason = _finish_reason_value(response)
        if finish_reason == 4:
            return {"_blocked": True}

        raw_text = _response_text_safe(response)
        data = _extract_json_block(raw_text)
        if not data:
            return None
        return _normalize_result(data, resource_type, fallback_full_text=text_chunk)

    try:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as pool:
            future_map = {
                pool.submit(process_chunk, idx, text_chunk): idx
                for idx, text_chunk in enumerate(chunks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"chunk {idx + 1}/{len(chunks)} failed ({exc}).")
    except Exception as exc:
        print(f"Chunked PDF structuring setup failed ({exc}).")
        return None

    if any((res or {}).get("_blocked") for res in results if isinstance(res, dict)):
        print(
            "blocked one or more PDF chunks due to policy (finish_reason=4). "
            "Using fast PDF fallback."
        )
        return _fast_pdf_local_fallback(pdf_text, resource_type)

    merged_articles: List[Dict[str, Any]] = []
    for res in results:
        if not res:
            continue
        merged_articles.extend(res.get("structured_articles", []) or [])

    if not merged_articles:
        return None

    merged_articles = _reindex_articles(merged_articles, resource_type)
    return {
        "full_text": pdf_text.strip(),
        "article_texts": [a.get("full_text", "") for a in merged_articles if a.get("full_text")],
        "method": "gemini_pdf_chunked",
        "structured_articles": merged_articles,
    }


def _normalize_result(
    data: Dict[str, Any], resource_type: str, fallback_full_text: str = ""
) -> Optional[Dict[str, Any]]:
    full_text = (data.get("full_text") or "").strip()
    raw_articles = data.get("articles") or []
    articles = _normalize_articles(
        raw_articles,
        "chapter" if resource_type == "books" else "article",
    )

    if not full_text and articles:
        full_text = "\n\n".join(a["full_text"] for a in articles if a.get("full_text"))
    if not full_text and fallback_full_text:
        full_text = fallback_full_text.strip()
    if not full_text:
        print("result had no text. Falling back.")
        return None
    if not articles:
        articles = [
            {
                "article_id": "full_document",
                "heading": "Full Document",
                "subheading": "",
                "column": "full",
                "body": [full_text],
                "full_text": full_text,
            }
        ]

    return {
        "full_text": full_text,
        "article_texts": [a["full_text"] for a in articles],
        "method": "gemini",
        "structured_articles": articles,
    }


def _fast_pdf_local_fallback(pdf_text: str, resource_type: str) -> Optional[Dict[str, Any]]:
    """
    Fast local fallback when Gemini is blocked or fails on PDFs.
    """
    clean_text = (pdf_text or "").strip()
    if not clean_text:
        return None

    if resource_type == "books":
        try:
            from ocr_processor import structure_book_text

            structured_book = structure_book_text(clean_text)
            chapters = structured_book.get("chapters", []) or []
            if chapters:
                return {
                    "full_text": clean_text,
                    "article_texts": [c.get("full_text", "") for c in chapters],
                    "method": "pdf_text_fast_book_fallback",
                    "structured_articles": chapters,
                }
        except Exception:
            pass

    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    body = paragraphs[:200] if paragraphs else [clean_text]
    full_article_text = "\n\n".join(body).strip() or clean_text
    article = {
        "article_id": "full_document",
        "heading": "Full Document",
        "subheading": "",
        "column": "full",
        "body": body,
        "full_text": full_article_text,
    }
    return {
        "full_text": clean_text,
        "article_texts": [article["full_text"]],
        "method": "pdf_text_fast_fallback",
        "structured_articles": [article],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_with_gemini(file_path: str, resource_type: str) -> Optional[Dict[str, Any]]:
    """
    Gemini-based structured extraction for newspapers, magazines and books.
    Returns normalized data compatible with existing pipeline.

    Paid-tier changes vs original:
      - Model: gemini-2.5-flash (same string, now on paid project)
      - Chunk size: 100k chars (was 28k)
      - Chunking threshold: 200k chars (was 60k)
      - Max parallel workers: 10 (was 3)
      - Timeouts: 45-90s (was 20-60s)
      - Retry logic: added _call_with_retry wrapper on all generate calls
    """
    if not GEMINI_API_KEY:
        print("API key not configured. Falling back to existing extraction.")
        return None

    try:
        import google.generativeai as genai
    except Exception as exc:
        print(f"SDK unavailable ({exc}). Falling back to existing extraction.")
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        is_pdf = file_path.lower().endswith(".pdf")
        if _should_skip_gemini_upload(resource_type, file_path):
            print(
                "Skipping upload extraction for periodicals to avoid "
                "copyrighted-recitation blocks. Using existing OCR extraction."
            )
            return None

        # ------------------------------------------------------------------
        # Fast PDF path: extract text natively, then ask Gemini to structure.
        # ------------------------------------------------------------------
        if is_pdf:
            pdf_text = _extract_pdf_text_fast(file_path)
            if pdf_text:
                if len(pdf_text) > CHUNK_THRESHOLD:
                    # Large PDF — split into 100k-char chunks, process in parallel
                    chunked_result = _structure_pdf_text_in_chunks(model, resource_type, pdf_text)
                    if chunked_result:
                        return chunked_result
                    print("Chunked PDF structuring failed. Using fast PDF fallback.")
                else:
                    prompt = _build_prompt(resource_type, "plain_pdf_text")
                    # Small/medium PDFs — single call with retry
                    for max_chars, timeout_s in ((150_000, TIMEOUT_SMALL), (100_000, 35)):
                        try:
                            trimmed_text = pdf_text[:max_chars]
                            response = _call_with_retry(
                                model,
                                [prompt, trimmed_text],
                                timeout_s=timeout_s,
                            )
                        except Exception as exc:
                            print(f"PDF structuring attempt failed ({exc}).")
                            continue

                        finish_reason = _finish_reason_value(response)
                        if finish_reason == 4:
                            print(
                                "blocked PDF structuring due to policy (finish_reason=4). "
                                "Using fast PDF fallback."
                            )
                            local = _fast_pdf_local_fallback(pdf_text, resource_type)
                            if local:
                                return local
                            return None

                        raw_text = _response_text_safe(response)
                        data = _extract_json_block(raw_text)
                        if not data:
                            continue

                        normalized = _normalize_result(
                            data, resource_type, fallback_full_text=pdf_text
                        )
                        if normalized:
                            return normalized

                    print("PDF structuring failed. Using fast PDF fallback.")
                local = _fast_pdf_local_fallback(pdf_text, resource_type)
                if local:
                    return local

            print("Fast PDF text path unavailable/empty. Trying file upload fallback...")

        # ------------------------------------------------------------------
        # Image path (and PDF fallback): use file upload.
        # ------------------------------------------------------------------
        uploaded = genai.upload_file(path=file_path)
        try:
            # Attempt 1: full extraction.
            response = _call_with_retry(
                model,
                [uploaded, _build_prompt(resource_type, "file_upload")],
                timeout_s=TIMEOUT_UPLOAD,
            )

            # Attempt 2: fallback best-effort extraction if recitation blocked.
            if _finish_reason_value(response) == 4:
                print(
                    "Copyright block detected. Retrying with best-effort summary prompt..."
                )
                response = _call_with_retry(
                    model,
                    [uploaded, _build_fallback_prompt(resource_type, "file_upload")],
                    timeout_s=TIMEOUT_UPLOAD,
                )
        finally:
            try:
                if hasattr(uploaded, "name") and uploaded.name:
                    genai.delete_file(uploaded.name)
            except Exception:
                pass

        finish_reason = _finish_reason_value(response)
        if finish_reason == 4:
            print(
                "blocked extraction even after best-effort retry "
                "(finish_reason=4). Falling back to existing extraction."
            )
            return None

        raw_text = _response_text_safe(response)
        data = _extract_json_block(raw_text)
        if not data:
            print("did not return parseable JSON. Falling back.")
            return None
        return _normalize_result(data, resource_type)

    except Exception as exc:
        print(f"extraction failed: {exc}. Falling back to existing extraction.")
        return None