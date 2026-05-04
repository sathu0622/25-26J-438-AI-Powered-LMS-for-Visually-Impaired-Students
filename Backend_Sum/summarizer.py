# summarizer.py
import torch
from typing import List, Dict, Any
from datetime import datetime
from models import DEVICE

def get_prefix(type_name: str) -> str:
    """Get summarization prefix based on type, aligned with training prompt."""
    type_map = {
        "newspaper": "summarize newspaper article in 3-4 factual sentences: ",
        "magazine": "summarize magazine article in about half the original length with key details: ",
        "book": "summarize book excerpt in detail preserving key ideas and context: "
    }
    return type_map.get(type_name, "summarize: ")

def summarize_text(text: str, source_type: str, summ_tokenizer, summ_model) -> str:
    """Summarize text using T5 model with improved prompt handling."""
    try:
        text = text.strip()
        if not text:
            return "No text available for summarization."

        # Handle subscription/paywall content
        if "purchase a subscription" in text.lower() or len(text) < 50:
            input_text = "summarize: The article content is unavailable. Provide a 2-sentence generic summary."
        else:
            prefix = get_prefix(source_type)
            input_text = prefix + text[:2000]  # Truncate long text for model input

        inputs = summ_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        ).to(DEVICE)

        with torch.no_grad():
            
            # output_ids = summ_model.generate(
            #     input_ids=inputs['input_ids'],
            #     attention_mask=inputs['attention_mask'],
            #     max_length=300 if source_type != "book" else 600,  # Longer for books
            #     num_beams=5,
            #     no_repeat_ngram_size=3,     # 🔥 prevents phrase repetition
            #     repetition_penalty=2.0,     # 🔥 penalizes repeated tokens

            #     early_stopping=True,
            #     length_penalty=1.2,         # encourages proper length

            #     do_sample=False             # keep deterministic (important)
            # )
            output_ids = summ_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=300 if source_type != "book" else 600,

                num_beams=5,
                no_repeat_ngram_size=4,
                repetition_penalty=2.0,
                length_penalty=1.5,

                early_stopping=True,

                do_sample=True,          # 🔥 allow controlled randomness
                top_p=0.9,               # 🔥 nucleus sampling
                temperature=0.7          # 🔥 reduces weird repetition
            )
        return summ_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Summarization error: {e}")
        return (text[:500] + "...") if len(text) > 500 else text

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
                article_text = ""
                if article.get("heading"):
                    article_text += article["heading"] + "\n"
                if article.get("subheading"):
                    article_text += article["subheading"] + "\n"
                if article.get("body"):
                    # Filter out empty strings
                    article_text += "\n".join(filter(None, article["body"]))

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