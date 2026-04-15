# azure_processor.py
from typing import Dict, Any, List
import re
import json
import tempfile
import os
from pathlib import Path
from config import GEMINI_API_KEY
from pdf2image import convert_from_path


NOISE_TITLE_KEYWORDS = {
    "advertisement", "advertisements", "advertorial", "advt", "promo",
    "classified", "sponsored", "paid", "public notice", "tender notice",
    "obituary", "vacancy", "for sale", "for rent", "buy now", "sale",
    "edition", "daily", "times", "tribune", "express", "chronicle",
    "newspaper", "press", "gazette", "today", "e-paper", "epaper"
}


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def _is_noise_title_candidate(text: str) -> bool:
    """
    Filters non-article headings such as ad labels and newspaper mastheads.
    """
    cleaned = text.strip()
    if not cleaned:
        return True

    norm = _normalize_text_for_match(cleaned)
    words = norm.split()
    if not words:
        return True

    # Very short labels and single-word mastheads are rarely article titles.
    if len(words) <= 2 and len(cleaned) <= 18:
        return True

    # Typical newspaper masthead / metadata patterns.
    if any(token in norm for token in ("www.", ".com", "vol ", "issue ", "dated ", "page ")):
        return True
    if re.search(r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", norm):
        return True

    # Advertisement or publication-name keywords.
    if any(keyword in norm for keyword in NOISE_TITLE_KEYWORDS):
        return True

    return False

def _build_gemini_prompt(resource_type: str) -> str:
    resource = resource_type.lower()
    if resource == "books":
        unit_name = "chapters"
        heading_hint = "chapter title"
    else:
        unit_name = "articles"
        heading_hint = "article headline"

    return (
        "You are an OCR and document-structure extractor.\n"
        "Read the provided page image and extract text only from the visible page.\n"
        "Do not summarize and do not add missing content.\n"
        f"Return strict JSON with this schema:\n"
        "{\n"
        '  "full_text": "all extracted text in reading order",\n'
        f'  "{unit_name}": [\n'
        "    {\n"
        f'      "heading": "{heading_hint}",\n'
        '      "subheading": "",\n'
        '      "body": ["paragraph 1", "paragraph 2"],\n'
        '      "full_text": "heading + body text"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Output valid JSON only.\n"
        "- Keep body as paragraph strings.\n"
        "- If no structure is clear, still return one item in the list with detected text."
    )


def _safe_parse_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()
    return json.loads(cleaned)


def _image_bytes_from_path(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _pdf_to_temp_images(pdf_path: str) -> List[str]:
    pages = convert_from_path(pdf_path, dpi=300)
    temp_paths: List[str] = []
    for page in pages:
        tmp_path = os.path.join(tempfile.gettempdir(), f"gemini_page_{os.urandom(4).hex()}.jpg")
        page.save(tmp_path, "JPEG", quality=95)
        temp_paths.append(tmp_path)
    return temp_paths


def _normalize_units(parsed: Dict[str, Any], resource_type: str) -> List[Dict[str, Any]]:
    resource = resource_type.lower()
    key = "chapters" if resource == "books" else "articles"
    units = parsed.get(key, []) or []

    normalized = []
    for i, unit in enumerate(units):
        heading = str(unit.get("heading", "") or "").strip()
        subheading = str(unit.get("subheading", "") or "").strip()
        body = unit.get("body", []) or []
        if not isinstance(body, list):
            body = [str(body)]
        body = [str(p).strip() for p in body if str(p).strip()]

        full_text = str(unit.get("full_text", "") or "").strip()
        if not full_text:
            full_text = "\n".join([x for x in [heading, subheading, *body] if x])

        normalized.append(
            {
                "heading": heading,
                "subheading": subheading if subheading else None,
                "body": body,
                "full_text": full_text,
                "article_id": f"{'chapter' if resource == 'books' else 'article'}_{i + 1}",
            }
        )
    return normalized


def extract_with_azure(file_path: str, resource_type: str = "newspapers") -> Dict[str, Any]:
    """
    Gemini-based structured text extraction.
    Function name kept for backward compatibility with existing pipeline code.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai package not installed. Falling back...")
        return None

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is missing. Falling back...")
        return None

    temp_images: List[str] = []
    image_paths: List[str] = []
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        lower_path = file_path.lower()
        if lower_path.endswith(".pdf"):
            temp_images = _pdf_to_temp_images(file_path)
            image_paths = temp_images
        else:
            image_paths = [file_path]

        extracted_unit_list: List[Dict[str, Any]] = []
        all_page_text: List[str] = []
        prompt = _build_gemini_prompt(resource_type)

        for image_path in image_paths:
            img_bytes = _image_bytes_from_path(image_path)
            response = model.generate_content(
                [
                    prompt,
                    {
                        "mime_type": f"image/{Path(image_path).suffix.replace('.', '').lower() or 'jpeg'}",
                        "data": img_bytes,
                    },
                ]
            )
            parsed = _safe_parse_json(response.text)
            page_text = str(parsed.get("full_text", "") or "").strip()
            if page_text:
                all_page_text.append(page_text)

            extracted_unit_list.extend(_normalize_units(parsed, resource_type))

        full_text = "\n\n".join(all_page_text).strip()
        return {
            "full_text": full_text,
            "articles": extracted_unit_list,
            "method": "gemini_image_extraction",
        }
    except Exception as e:
        print(f"Gemini extraction error: {e}. Falling back...")
        return None
    finally:
        for tmp in temp_images:
            try:
                os.remove(tmp)
            except Exception:
                pass

def extract_articles_exact_colab_logic(result) -> List[Dict[str, Any]]:
    """
    Extract structured articles from Azure Document Intelligence result.

    """
    try:
        # Step 1: Group paragraphs by column (left/right)
        articles_by_column = {}
        
        for p in result.paragraphs:
            if not p.bounding_regions:
                continue
                
            box = p.bounding_regions[0].polygon
            
            # Calculate average X position 
            x_avg = sum(point.x for point in box) / len(box)
            
            # Determine column 
            column = "left" if x_avg < 0.5 else "right"
            
            if column not in articles_by_column:
                articles_by_column[column] = []
            
            # Add paragraph with its original properties
            articles_by_column[column].append(p)
        
        print(f"Paragraphs grouped by column: { {k: len(v) for k, v in articles_by_column.items()} }")
        
        # Step 2: Function to split paragraphs into articles 
        def split_into_articles(paragraphs):
            article_list = []
            current_article = None

            for p in paragraphs:
                text = p.content.strip()
                role = getattr(p, "role", "")  # use role if exists

                if len(text) < 5:
                    continue

                if role == "title":
                    if _is_noise_title_candidate(text):
                        continue
                    if current_article:
                        article_list.append(current_article)
                    current_article = {
                        "heading": text,
                        "subheading": None,
                        "body": []
                    }

                elif role == "sectionHeading" and current_article:
                    if current_article["subheading"] is None:
                        current_article["subheading"] = text
                    else:
                        current_article["body"].append(text)

                elif current_article:
                    current_article["body"].append(text)

            if current_article:
                article_list.append(current_article)

            return article_list
        
        # Step 3: Process each column
        all_articles = []
        
        for col, paras in articles_by_column.items():
            # Sort paragraphs by Y position (top to bottom)
            def get_top_y(p):
                if not p.bounding_regions:
                    return 0
                box = p.bounding_regions[0].polygon
                return min(point.y for point in box)
            
            sorted_paras = sorted(paras, key=get_top_y)
            
            # Apply your exact article splitting function
            column_articles = split_into_articles(sorted_paras)
            
            print(f"\n📰 COLUMN: {col.upper()} - Found {len(column_articles)} articles")
            
            for i, art in enumerate(column_articles, 1):
                print(f"  ARTICLE {i}")
                print(f"    HEADING: {art['heading'][:80] if art['heading'] else 'No heading'}...")
                print(f"    SUBHEADING: {art['subheading'][:80] if art['subheading'] else 'No subheading'}...")
                print(f"    BODY PARAGRAPHS: {len(art['body'])}")
                
                # Create full text for each article
                full_text = ""
                if art["heading"]:
                    full_text += art["heading"] + "\n"
                if art["subheading"]:
                    full_text += art["subheading"] + "\n"
                if art["body"]:
                    full_text += "\n".join(art["body"])
                
                # Add column information
                art["column"] = col
                art["full_text"] = full_text
                art["article_number"] = i
                art["article_id"] = f"{col}_{i}"  # Unique ID for selection
                
                all_articles.append(art)
        
        print(f"\nTotal articles found across all columns: {len(all_articles)}")
        
        # Step 4: If no articles found with role-based detection, try alternative method
        if len(all_articles) == 0:
            print("No articles found with role detection. Trying alternative grouping...")
            all_articles = alternative_article_grouping(result)
        
        # Step 5: Sort articles by position (left column first, then right column, top to bottom)
        def get_article_position(article):
            # Find the first paragraph of the article to determine position
            for p in result.paragraphs:
                if p.content.strip() == article.get("heading", "").strip():
                    if p.bounding_regions:
                        box = p.bounding_regions[0].polygon
                        x_avg = sum(point.x for point in box) / len(box)
                        y_avg = sum(point.y for point in box) / len(box)
                        column = 0 if x_avg < 0.5 else 1
                        return (column, y_avg)
            return (0, 0)
        
        # Sort articles by column, then by Y position
        sorted_articles = sorted(all_articles, key=lambda a: get_article_position(a))
        
        # Reindex articles with unique IDs
        for i, article in enumerate(sorted_articles):
            article["article_index"] = i + 1
            if "article_id" not in article:
                article["article_id"] = f"article_{i+1}"
        
        return sorted_articles
        
    except Exception as e:
        print(f"Error extracting articles: {e}")
        import traceback
        traceback.print_exc()
        return []

def alternative_article_grouping(result):
    """
    Alternative method for grouping paragraphs into articles when role detection fails.
    This is still based on your Colab logic but with different heuristics.
    """
    try:
        articles = []
        current_group = []
        last_y = -1
        y_threshold = 0.02  # 2% of page height
        
        # Get all paragraphs sorted by Y position
        def get_top_y(p):
            if not p.bounding_regions:
                return 0
            box = p.bounding_regions[0].polygon
            return min(point.y for point in box)
        
        sorted_paragraphs = sorted(result.paragraphs, key=get_top_y)
        
        for p in sorted_paragraphs:
            text = p.content.strip()
            if len(text) < 10:  # Skip very short text
                continue
            
            current_y = get_top_y(p)
            
            # Check if this should start a new article
            is_new_article = False
            
            # Heuristic 1: Check for heading patterns (like in Colab)
            words = text.split()
            if 2 <= len(words) <= 15:
                # Check for uppercase or ending with punctuation
                if (text.isupper() or 
                    text.endswith(".") or 
                    text.endswith(":") or
                    text.endswith("?") or
                    text.endswith("!")):
                    is_new_article = True
            
            # Heuristic 2: Large vertical gap
            if last_y != -1 and (current_y - last_y) > y_threshold and current_group:
                is_new_article = True
            
            if is_new_article and current_group:
                # Create article from current group
                heading = current_group[0].content.strip() if len(current_group[0].content.split()) <= 15 else ""
                if _is_noise_title_candidate(heading):
                    heading = ""
                body = [p.content for p in current_group[1:]] if len(current_group) > 1 else [p.content for p in current_group]
                
                articles.append({
                    "heading": heading,
                    "subheading": None,
                    "body": body,
                    "full_text": "\n".join([p.content for p in current_group])
                })
                current_group = [p]
            else:
                current_group.append(p)
            
            last_y = current_y
        
        # Add the last group
        if current_group:
            heading = current_group[0].content if len(current_group[0].content.split()) <= 15 else ""
            if _is_noise_title_candidate(heading):
                heading = ""
            body = [p.content for p in current_group[1:]] if len(current_group) > 1 else [p.content for p in current_group]
            
            articles.append({
                "heading": heading,
                "subheading": None,
                "body": body,
                "full_text": "\n".join([p.content for p in current_group])
            })
        
        print(f"Alternative grouping found {len(articles)} articles")
        
        # Add article IDs
        for i, article in enumerate(articles):
            article["article_id"] = f"alt_article_{i+1}"
        
        return articles
        
    except Exception as e:
        print(f"Alternative grouping error: {e}")
        return []