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
CHUNK_SIZE = 100_000          # chars per chunk
CHUNK_THRESHOLD = 200_000     # only split PDFs larger than this
MAX_WORKERS = 10              # parallel chunk workers
TIMEOUT_CHUNK = 600           # seconds per chunk call (10 minutes)
TIMEOUT_SMALL = 600           # seconds for single small-PDF call (10 minutes)
TIMEOUT_UPLOAD = 600          # seconds for file-upload path (10 minutes)

# ---------------------------------------------------------------------------
# FIX: Deterministic generation config — temperature=0 ensures the SAME
#      output every time for the same image + prompt combination.
# ---------------------------------------------------------------------------
GENERATION_CONFIG = {
    "temperature": 0,          # Fully deterministic — no random sampling
    "top_p": 1,                # No nucleus sampling (irrelevant at temp=0)
    "top_k": 1,                # Always pick the single most likely token
}


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
    temperature=0 in the model config ensures consistent outputs across retries.
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
# Internal helpers
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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _build_newspaper_image_prompt() -> str:
    """
    Dedicated prompt for newspaper/magazine IMAGE extraction.

    KEY FIX: Explicitly handles the pattern where a PHOTO appears between
    a heading and its body text — this is extremely common in newspapers and
    causes models to incorrectly split one article into multiple pieces.

    Also handles large in-body section headings (like "THE LION FLAG OF LANKA")
    that are part of the article body, NOT separate standalone articles.
    """
    return """
You are a precise newspaper OCR and article-extraction engine.

Your job: Read this newspaper page image and extract EVERY article with its COMPLETE text.

=== CRITICAL RULES ===

1. COMPLETE TEXT ONLY — Never truncate, summarize, or omit any words.
   Every sentence and every paragraph must appear in full in "body" and "full_text".

2. HEADING + SUBHEADING + PHOTO + BODY = ONE SINGLE ARTICLE
   This is the most important rule. Newspaper layouts frequently place a photo
   BETWEEN the heading and the body text. This does NOT mean they are separate articles.
   Everything that belongs to a story — heading, subheading, any photos in between,
   and ALL body paragraphs below or beside — is ONE article.

   *** NEVER create a new article just because there is a photo or image gap
   between the heading and the body text. ***

3. LARGE IN-BODY SECTION HEADINGS ARE NOT NEW ARTICLES
   Sometimes a large bold/decorative text appears WITHIN the body of an article
   as a section title or visual divider (e.g. "THE LION FLAG OF LANKA").
   If this text does NOT have its own independent body paragraphs and is
   visually part of a larger article, it is a SECTION HEADING inside that
   article — NOT a separate article. Include it as a paragraph in the body,
   prefixed with the text as-is.

4. MULTI-COLUMN BODY — Body text often flows across 2-3 columns. Read all columns
   left-to-right, top-to-bottom and join them into one continuous body for that article.

5. ARTICLE BOUNDARIES — Only start a new article when you see a NEW, DISTINCT heading
   that clearly introduces a completely different topic or story with its OWN body text.
   A photo alone, a column break alone, or a decorative section title alone is NOT
   a new article boundary.

6. COLUMN LABELS — Set "column" to:
   - "left"  if the article sits only in the left column
   - "right" if the article sits only in the right column
   - "full"  if the article spans the full width or multiple columns

7. OPINION / SECTION LABELS — If an article has a section label like "OPINION" above
   its heading, prefix it into the heading field:
   e.g. heading = "OPINION: Sri Lanka, Our Country to Celebrate!"
   Do NOT make the section label a separate article.

8. body field — JSON array of strings, one string per paragraph.
   Never use placeholders like "[continues...]" or "[body text]".

9. full_text — heading + "\\n" + subheading (if any) + "\\n\\n" + all body paragraphs
   joined by "\\n\\n". Must be the complete readable article text.

=== STEP-BY-STEP APPROACH ===

Before writing JSON, do this mentally:
  Step 1. Count only TRUE article headings — ones that introduce a full independent story.
  Step 2. For each heading, collect ALL body paragraphs belonging to it,
          even if a photo or decorative section title appears in between.
  Step 3. Treat decorative/bold section titles within the body as part of the body text,
          not as new article headings.
  Step 4. Only then write the JSON.

=== OUTPUT FORMAT ===

Return ONLY valid JSON. No markdown, no code fences, no extra commentary.

{
  "full_text": "<all article texts from the page joined together>",
  "articles": [
    {
      "article_id": "article_1",
      "heading": "<exact heading text>",
      "subheading": "<exact subheading/deck line, or empty string>",
      "column": "left|right|full",
      "body": [
        "<full paragraph 1 text>",
        "<full paragraph 2 text>"
      ],
      "full_text": "<heading + subheading + all body paragraphs>"
    }
  ]
}

=== CORRECT EXAMPLE FOR THIS SPECIFIC PAGE ===

This page has exactly 2 articles:
  1. The OPINION column on the left.
  2. "A 75 YEAR JOURNEY" — which includes the subheading, the photo, the decorative
     "THE LION FLAG OF LANKA" section title, and ALL body paragraphs in both columns.

{
  "full_text": "OPINION: Sri Lanka, Our Country to Celebrate!\\n\\n[full text]...\\n\\nA 75 YEAR JOURNEY\\nGoing down memory lane with nostalgia\\n\\n[full text]...",
  "articles": [
    {
      "article_id": "article_1",
      "heading": "OPINION: Sri Lanka, Our Country to Celebrate!",
      "subheading": "",
      "column": "left",
      "body": [
        "Sri Lankans recovered a land, resources and the right to make decisions, but not in any way they wanted...",
        "This is why, looking back at the 75 years that have passed since that day, we can be proud of how far we have come...",
        "The hope, understandably exaggerated at the moment of Independence, was tempered by realities about the tasks at hand...",
        "There have been disappointments and yet significant achievements as well...",
        "Independence arrived after almost five centuries of conquest, plunder and subjugation in various degrees by various European powers...",
        "We have always travelled towards a country called Tomorrow. Our people have hoped, worked, struggled, faltered and wept...",
        "Sri Lanka. Our country. Ours to protect, ours to rebuild, ours to recover, ours to celebrate!"
      ],
      "full_text": "OPINION: Sri Lanka, Our Country to Celebrate!\\n\\nSri Lankans recovered a land..."
    },
    {
      "article_id": "article_2",
      "heading": "A 75 YEAR JOURNEY",
      "subheading": "Going down memory lane with nostalgia",
      "column": "full",
      "body": [
        "Although controversial at times, the lion in the flag of Sri Lanka has long been Sri Lanka's pride, be it on a jersey representing Sri Lankan sport or as the symbol of a social revolution...",
        "The kings who succeeded Vijaya were said to have used this lion banner extensively, making the lion flag a representation of liberation and hope...",
        "THE LION FLAG OF LANKA",
        "Legendary King Dutugemunu brought a flag with him that featured a lion holding a sword on his right forepaw together with two other emblems, the Sun and the Moon...",
        "By 1815, the banner was still in use, even though the reign of the last king of the Kandyan Kingdom, King Sri Vikrama Rajasinha, was brought to an end by the colonizers...",
        "As an independence movement took hold of the subcontinent, a Ceylonese movement too grew in strength during the early 20th century...",
        "A picture of it was subsequently published in a special edition of the Dinamina newspaper, a publication owned by Wijewardene to mark 100 years since the end of Sri Lankan independence...",
        "The first Prime Minister of independent Ceylon Hon. D.S. Senanayake, hoisted the Lion Flag at the ceremony on February 4, 1948...",
        "Finally in 1972, the flag was modified once more, with four stylized leaves of the Bo (Pipul) tree, a Buddhist symbol, added to the four corners to replace the four pinnacles."
      ],
      "full_text": "A 75 YEAR JOURNEY\\nGoing down memory lane with nostalgia\\n\\nAlthough controversial at times..."
    }
  ]
}

Now extract all articles from the newspaper image with COMPLETE text following the rules above.
"""


def _build_prompt(resource_type: str, source_kind: str) -> str:
    """
    For PDFs and generic sources (books etc.) use the original prompt.
    For newspaper/magazine IMAGE uploads use the dedicated prompt above.
    """
    if resource_type in {"newspapers", "magazines"} and source_kind == "file_upload":
        return _build_newspaper_image_prompt()

    return f"""
You are an extraction engine for scanned/printed documents.
Resource type: {resource_type}
Input source: {source_kind}

Extract all meaningful article/chapter units.
Handle noisy OCR and keep original language text.

IMPORTANT: Extract COMPLETE text for every article. Never truncate or summarize body content.
An article's heading + subheading + all body paragraphs (even across multiple columns)
all belong to the SAME article entry.

Return strict JSON only (no markdown, no extra text) with this schema:
{{
  "full_text": "all extracted text merged in reading order",
  "articles": [
    {{
      "article_id": "stable_id",
      "heading": "title or chapter heading",
      "subheading": "optional subheading or deck line",
      "column": "left|right|full",
      "body": ["paragraph 1", "paragraph 2"],
      "full_text": "heading + subheading + body text"
    }}
  ]
}}

Rules:
- For newspapers/magazines: split by individual articles. Each article = one heading + all its body text.
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
      "subheading": "deck line if present",
      "body": ["Summary snippet 1", "Summary snippet 2"],
      "full_text": "Heading + Summaries"
    }}
  ]
}}
"""


def _should_skip_gemini_upload(resource_type: str, file_path: str) -> bool:
    if file_path.lower().endswith(".pdf"):
        return False
    return resource_type in {"magazines"}


def _chunk_text_for_gemini(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
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
    chunks = _chunk_text_for_gemini(pdf_text, max_chars=CHUNK_SIZE)
    if not chunks:
        return None

    prompt = _build_prompt(resource_type, "plain_pdf_text_chunk")
    results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)

    def process_chunk(idx: int, text_chunk: str) -> Optional[Dict[str, Any]]:
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
# Post-processing validation
# ---------------------------------------------------------------------------

def _validate_and_repair_articles(
    articles: List[Dict[str, Any]],
    resource_type: str,
    min_body_words: int = 30,
) -> List[Dict[str, Any]]:
    repaired = []
    for art in articles:
        full_text = (art.get("full_text") or "").strip()
        heading = (art.get("heading") or "").strip()
        body = art.get("body") or []
        body_text = " ".join(body).strip()
        word_count = len(body_text.split()) if body_text else 0

        if heading and word_count < min_body_words and not body_text:
            heading_escaped = re.escape(heading)
            subheading = (art.get("subheading") or "").strip()
            remainder = re.sub(r"^\s*" + heading_escaped, "", full_text, count=1).strip()
            if subheading:
                subheading_escaped = re.escape(subheading)
                remainder = re.sub(r"^\s*" + subheading_escaped, "", remainder, count=1).strip()
            if remainder:
                art = dict(art)
                art["body"] = [p.strip() for p in remainder.split("\n\n") if p.strip()] or [remainder]
                art["full_text"] = full_text

        repaired.append(art)
    return repaired


def _needs_retry(articles: List[Dict[str, Any]], min_body_words: int = 30) -> bool:
    for art in articles:
        heading = (art.get("heading") or "").strip()
        body = art.get("body") or []
        body_text = " ".join(body).strip()
        word_count = len(body_text.split()) if body_text else 0
        if heading and word_count < min_body_words:
            return True
    return False


def _total_words(arts: List[Dict[str, Any]]) -> int:
    return sum(
        len(" ".join(a.get("body") or []).split())
        for a in arts
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_with_gemini(file_path: str, resource_type: str) -> Optional[Dict[str, Any]]:
    """
    Gemini-based structured extraction for newspapers, magazines and books.

    KEY CHANGE FOR CONSISTENCY:
      - Model is initialized with temperature=0, top_k=1, top_p=1.
        This makes Gemini fully deterministic — uploading the same image
        with the same prompt will always produce the same output.
      - The retry logic still runs if the first pass looks incomplete,
        but because temperature=0 the retry will produce the same result
        as the first attempt (so it won't silently change the output).
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

        # ------------------------------------------------------------------
        # FIX: temperature=0 for fully deterministic, consistent output.
        # Every call with the same image + prompt will return identical JSON.
        # ------------------------------------------------------------------
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0,   # No randomness — always pick the top token
                top_p=1,         # Disabled nucleus sampling
                top_k=1,         # Only the single most probable token
            ),
        )

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
                    chunked_result = _structure_pdf_text_in_chunks(model, resource_type, pdf_text)
                    if chunked_result:
                        return chunked_result
                    print("Chunked PDF structuring failed. Using fast PDF fallback.")
                else:
                    prompt = _build_prompt(resource_type, "plain_pdf_text")
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
            primary_prompt = _build_prompt(resource_type, "file_upload")

            # Attempt 1: full extraction with improved prompt.
            response = _call_with_retry(
                model,
                [uploaded, primary_prompt],
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

        normalized = _normalize_result(data, resource_type)
        if not normalized:
            return None

        # ------------------------------------------------------------------
        # Validate extracted articles and repair/retry if incomplete.
        # NOTE: With temperature=0, retrying with the same prompt will
        # produce the same result. The retry is kept only as a safety net
        # for cases where the FALLBACK prompt produces a better structure.
        # ------------------------------------------------------------------
        structured = normalized.get("structured_articles") or []
        structured = _validate_and_repair_articles(structured, resource_type)

        if _needs_retry(structured) and resource_type in {"newspapers", "magazines"}:
            print(
                "One or more articles appear incomplete (very short body). "
                "Retrying extraction with stricter completeness prompt..."
            )
            try:
                uploaded_retry = genai.upload_file(path=file_path)
                try:
                    retry_response = _call_with_retry(
                        model,
                        [uploaded_retry, _build_newspaper_image_prompt()],
                        timeout_s=TIMEOUT_UPLOAD,
                    )
                finally:
                    try:
                        if hasattr(uploaded_retry, "name") and uploaded_retry.name:
                            genai.delete_file(uploaded_retry.name)
                    except Exception:
                        pass

                retry_raw = _response_text_safe(retry_response)
                retry_data = _extract_json_block(retry_raw)
                if retry_data:
                    retry_normalized = _normalize_result(retry_data, resource_type)
                    if retry_normalized:
                        retry_structured = retry_normalized.get("structured_articles") or []
                        retry_structured = _validate_and_repair_articles(
                            retry_structured, resource_type
                        )
                        # Only use retry result if it has more content
                        if _total_words(retry_structured) > _total_words(structured):
                            print("Retry produced more complete extraction. Using retry result.")
                            structured = retry_structured
                            normalized = retry_normalized
            except Exception as exc:
                print(f"Retry extraction failed ({exc}). Using original result.")

        normalized["structured_articles"] = structured
        normalized["article_texts"] = [
            a.get("full_text", "") for a in structured if a.get("full_text")
        ]
        return normalized

    except Exception as exc:
        print(f"extraction failed: {exc}. Falling back to existing extraction.")
        return None