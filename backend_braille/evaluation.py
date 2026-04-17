import torch
import re
import numpy as np
from utils import (
    semantic_similarity,
    keyword_overlap_score,
    jaccard_similarity,
    length_penalty
)
from logger_config import logger
from rag_retriever import retrieve_context

# =========================
# CACHE (IMPORTANT FIX)
# =========================
_context_cache = {}


# =========================
# CLEANING FUNCTION
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"/[a-zA-Z]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# RAG ANSWER GENERATION
# =========================
def generate_correct_answer(question, tokenizer, model):
    torch.manual_seed(42)
    np.random.seed(42)

    # 🔥 CACHE FIX (VERY IMPORTANT)
    if question in _context_cache:
        context = _context_cache[question]
    else:
        raw_context = retrieve_context(question, top_k=3)
        context = clean_text(raw_context)
        _context_cache[question] = context

    logger.info(f"Cleaned context: {context}")

    prompt = f"""
You are a Sri Lankan G.C.E. O/L History examiner.

TASK:
Write a marking scheme answer using ONLY the context provided.

STRICT RULES:
- Write ONLY the answer content.
- Do NOT include marks, scores, percentages, or evaluation.
- Do NOT explain your reasoning.
- Do NOT include headings or extra commentary.
- Do NOT add anything outside the answer.
- Use ONLY factual sentences from the context.

FORMAT:
- 2 to 3 short paragraphs.
- Each paragraph = one marking point.
- Simple O/L student-level English.
- Include key names, events, and dates if present.
- Total length: 120–160 words.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    logger.info("Generating model answer...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    answer = clean_text(response)

    # safer sentence cleanup
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 2]
    if sentences:
        answer = '. '.join(sentences) + '.'

    return answer


# =========================
# SCORING FUNCTION
# =========================
def calculate_final_score(correct, student, sbert_model):

    correct = clean_text(correct)
    student = clean_text(student)

    semantic = semantic_similarity(correct, student, sbert_model)
    keyword = keyword_overlap_score(correct, student)
    jaccard = jaccard_similarity(correct, student)

    # FIXED WEIGHTS (more realistic for essays)
    semantic_weighted = semantic * 0.70
    keyword_weighted = keyword * 0.20
    jaccard_weighted = jaccard * 0.10

    length_factor = length_penalty(correct, student)
    length_factor = max(0.85, min(1.0, length_factor))  # stability clamp

    final = (semantic_weighted + keyword_weighted + jaccard_weighted) * length_factor

    return round(final, 2), round(semantic_weighted, 2), round(keyword_weighted, 2), round(jaccard_weighted, 2)


# =========================
# FEEDBACK
# =========================
def generate_feedback(score):
    if score >= 75:
        return "Excellent answer. You have demonstrated a strong and accurate understanding of the historical topic."
    elif score >= 45:
        return "Good attempt. You have shown a basic understanding, but some key points are missing or incomplete."
    return "Limited understanding. Revise key facts and events before attempting again."


# =========================
# MAIN EVALUATION
# =========================
def evaluate_student_answer(question, student_answer, tokenizer, model, sbert):

    correct_answer = generate_correct_answer(question, tokenizer, model)

    final, semantic, keyword, jaccard = calculate_final_score(
        correct_answer,
        student_answer,
        sbert
    )

    status = "PASS" if final >= 75 else "NEEDS IMPROVEMENT" if final >= 45 else "FAIL"

    return {
        "question": question,
        "student_answer": student_answer,
        "model_answer": correct_answer,
        "final_score": final,
        "semantic_similarity": semantic,
        "keyword_match": keyword,
        "jaccard_similarity": jaccard,
        "status": status,
        "feedback": generate_feedback(final)
    }