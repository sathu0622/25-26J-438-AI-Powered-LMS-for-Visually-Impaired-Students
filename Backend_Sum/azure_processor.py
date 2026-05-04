# azure_processor.py
from typing import Dict, Any, List, Optional  
import re
from config import AZURE_ENDPOINT, AZURE_KEY


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

def extract_with_azure(file_path: str) -> Dict[str, Any]:
    """
    Extract text and layout using Azure Document Intelligence.
    This is specifically for newspapers and magazines.
    """
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
        from azure.core.exceptions import HttpResponseError
        
        print(f"Using Azure Document Intelligence for enhanced extraction...")
        
        # Initialize Azure client
        client = DocumentAnalysisClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_KEY)
        )
        
        # Open and analyze the file
        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                document=f
            )
        
        result = poller.result()
        
        # Extract all text
        full_text = ""
        if hasattr(result, 'content'):
            full_text = result.content
        else:
            # Fallback: combine all paragraphs
            paragraphs = []
            for p in result.paragraphs:
                paragraphs.append(p.content)
            full_text = "\n".join(paragraphs)
        
        # Extract structured articles 
        articles = extract_articles_exact_colab_logic(result)
        
        return {
            "full_text": full_text,
            "articles": articles,
            "raw_result": result,
            "method": "azure_document_intelligence"
        }
        
    except ImportError:
        print("Azure packages not installed. Falling back to simple OCR...")
        return None
    except HttpResponseError as e:
        print(f"Azure API error: {e}. Falling back to simple OCR...")
        return None
    except Exception as e:
        print(f"Azure processing error: {e}. Falling back to simple OCR...")
        return None

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