# summarizer.py
import torch
from typing import List, Dict, Any, Optional  # ADD Optional here
from datetime import datetime
from models import DEVICE

def get_prefix(type_name: str) -> str:
    """Get summarization prefix based on type."""
    if type_name == "newspapers":
        return "summarize: short summary: "
    elif type_name == "magazine":
        return "summarize: medium summary: "
    elif type_name == "books":
        return "summarize: long summary in detail: "
    return "summarize: "

def summarize_text(text: str, source_type: str, summ_tokenizer, summ_model) -> str:
    """Summarize text using T5 model."""
    try:
        if not text.strip():
            return "No text available for summarization."
            
        prefix = get_prefix(source_type)
        
        # Take first 2000 characters for summarization
        input_text = prefix + text[:2000]
        
        inputs = summ_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        ).to(DEVICE)
        
        with torch.no_grad():
            output_ids = summ_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=300,
                num_beams=4,
                early_stopping=True
            )
        
        return summ_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Summarization error: {e}")
        # Fallback: return first 500 characters
        return text[:500] + "..." if len(text) > 500 else text

def summarize_specific_article(
    articles: List[Dict[str, Any]], 
    article_id: str, 
    resource_type: str,
    summ_tokenizer,
    summ_model
) -> Dict[str, Any]:
    """Summarize a specific article by its ID."""
    for article in articles:
        if article.get("article_id") == article_id or article.get("heading", "").strip() == article_id.strip():
            print(f"Summarizing article: {article.get('heading', 'No heading')[:50]}...")
            
            # Get full text of the article
            article_text = article.get("full_text", "")
            if not article_text:
                # Reconstruct from parts
                article_text = ""
                if article.get("heading"):
                    article_text += article["heading"] + "\n"
                if article.get("subheading"):
                    article_text += article["subheading"] + "\n"
                if article.get("body"):
                    article_text += "\n".join(article["body"])
            
            # Generate summary
            summary = summarize_text(article_text, resource_type, summ_tokenizer, summ_model)
            
            return {
                "article_id": article_id,
                "heading": article.get("heading", "No heading"),
                "subheading": article.get("subheading", ""),
                "column": article.get("column", "unknown"),
                "article_index": article.get("article_index", 0),
                "full_text_preview": article_text[:200] + "..." if len(article_text) > 200 else article_text,
                "summary": summary,
                "word_count": len(article_text.split()),
                "paragraph_count": len(article.get("body", [])),
                "timestamp": datetime.now().isoformat()
            }
    
    return {
        "error": "Article not found",
        "article_id": article_id,
        "available_articles": [a.get("article_id") for a in articles if a.get("article_id")]
    }