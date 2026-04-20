import json
import re
from typing import Any, Dict, List, Optional

from config import GEMINI_API_KEY


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


def extract_with_gemini(file_path: str, resource_type: str) -> Optional[Dict[str, Any]]:
    """
    Gemini-based structured extraction for newspapers, magazines and books.
    Returns normalized data compatible with existing pipeline.
    """
    if not GEMINI_API_KEY:
        print("Gemini API key not configured. Falling back to existing extraction.")
        return None

    try:
        import google.generativeai as genai
    except Exception as exc:
        print(f"Gemini SDK unavailable ({exc}). Falling back to existing extraction.")
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        uploaded = genai.upload_file(path=file_path)

        prompt = f"""
You are an extraction engine for scanned documents.
Resource type: {resource_type}

Extract all meaningful article/chapter units from the file.
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
- If policy prevents verbatim extraction, still return best-effort structure using short snippets.
"""
        response = model.generate_content([uploaded, prompt])
        finish_reason = _finish_reason_value(response)
        if finish_reason == 4:
            print(
                "Gemini blocked direct extraction due to copyrighted-recitation policy "
                "(finish_reason=4). Falling back to existing extraction."
            )
            return None

        raw_text = _response_text_safe(response)
        data = _extract_json_block(raw_text)
        if not data:
            print("Gemini did not return parseable JSON. Falling back.")
            return None

        full_text = (data.get("full_text") or "").strip()
        raw_articles = data.get("articles") or []
        articles = _normalize_articles(
            raw_articles,
            "chapter" if resource_type == "books" else "article",
        )

        if not full_text and articles:
            full_text = "\n\n".join(a["full_text"] for a in articles if a.get("full_text"))
        if not full_text:
            print("Gemini result had no text. Falling back.")
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
    except Exception as exc:
        print(f"Gemini extraction failed: {exc}. Falling back to existing extraction.")
        return None
