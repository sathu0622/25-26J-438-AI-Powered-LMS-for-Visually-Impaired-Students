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
from rag_retriever import retrieve_context, retrieve_topic_info

# =========================
# CACHE
# =========================
_context_cache = {}


# =========================
# CLEAN TEXT
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"/[a-zA-Z]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# EXTRACT KEY POINTS
# =========================
def extract_key_points(text):
    parts = re.split(r'[.;]', text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


# =========================
# FIND MISSING POINTS
# =========================
def find_missing_points(correct_answer, student_answer, sbert_model, threshold=70):
    correct_points = extract_key_points(correct_answer)
    student_points = extract_key_points(student_answer)

    missing = []

    for cp in correct_points:
        matched = False

        for sp in student_points:
            sim = semantic_similarity(cp, sp, sbert_model)

            logger.info(f"COMPARE:\nMODEL: {cp}\nSTUDENT: {sp}\nSIMILARITY: {sim}")

            if sim >= threshold:
                matched = True
                break

        if not matched:
            missing.append(cp)

    logger.info(f"MISSING POINTS: {missing}")

    return missing[:3]


# =========================
# GENERATE MODEL ANSWER
# =========================
def generate_correct_answer(question, tokenizer, model):
    torch.manual_seed(42)
    np.random.seed(42)

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
# SCORE CALCULATION
# =========================
def calculate_final_score(correct, student, sbert_model):

    correct = clean_text(correct)
    student = clean_text(student)

    semantic = semantic_similarity(correct, student, sbert_model)
    keyword = keyword_overlap_score(correct, student)
    jaccard = jaccard_similarity(correct, student)

    semantic_weighted = semantic * 0.70
    keyword_weighted = keyword * 0.20
    jaccard_weighted = jaccard * 0.10

    length_factor = length_penalty(correct, student)

    final = (semantic_weighted + keyword_weighted + jaccard_weighted) * length_factor

    return (
        round(final, 2),
        round(semantic_weighted, 2),
        round(keyword_weighted, 2),
        round(jaccard_weighted, 2)
    )


# =========================
# FEEDBACK GENERATION
# =========================
def generate_feedback(score, correct_answer, student_answer, question, sbert_model):

    missing_points = find_missing_points(correct_answer, student_answer, sbert_model)

    # Fetch relevant chapter and topic from RAG for further study suggestion
    chapter, topic = retrieve_topic_info(question)

    further_study = ""
    if chapter or topic:
        further_study += "\n\n📚 Further Study Recommendation:\n"
        if chapter:
            further_study += f"  • Chapter : {chapter}\n"
        if topic:
            further_study += f"  • Topic   : {topic}\n"
        further_study += "  Review this section in your textbook to strengthen your understanding."

    # ── PASS ──────────────────────────────────────────────────────────────────
    if score >= 75:
        return (
            f"Excellent answer (Score: {score}%). "
            "You have demonstrated a strong and accurate understanding of the topic."
            + further_study
        )

    # ── NEEDS IMPROVEMENT ─────────────────────────────────────────────────────
    elif score >= 45:
        feedback = (
            f"Good attempt (Score: {score}%). "
            "You identified some main points, but important supporting historical details are missing.\n\n"
        )

        if missing_points:
            feedback += "Try to include the following points:\n"
            for i, point in enumerate(missing_points, 1):
                feedback += f"  {i}. {point}\n"
        else:
            feedback += (
                "Try to include more specific historical examples, "
                "key dates, names, and events related to the topic.\n"
            )

        feedback += "\nImprove your explanation by adding clearer supporting details."
        feedback += further_study
        return feedback

    # ── FAIL ──────────────────────────────────────────────────────────────────
    else:
        feedback = (
            f"Your answer shows limited understanding of the topic (Score: {score}%).\n\n"
        )

        if missing_points:
            feedback += "Important points that are missing from your answer:\n"
            for i, point in enumerate(missing_points, 1):
                feedback += f"  {i}. {point}\n"
        else:
            feedback += (
                "Your answer is missing major historical causes and supporting facts. "
                "Make sure to cover the key events and their significance.\n"
            )

        feedback += "\nRevise the lesson thoroughly and rewrite your answer with the main points and examples."
        feedback += further_study
        return feedback


# =========================
# MAIN EVALUATION
# =========================
def evaluate_student_answer(question, student_answer, tokenizer, model, sbert):

    correct_answer = generate_correct_answer(question, tokenizer, model)

    final, semantic, keyword, jaccard = calculate_final_score(correct_answer, student_answer, sbert)

    status = "PASS" if final >= 75 else "NEEDS IMPROVEMENT" if final >= 45 else "FAIL"

    feedback = generate_feedback(final, correct_answer, student_answer, question, sbert)

    return {
        "question": question,
        "student_answer": student_answer,
        "model_answer": correct_answer,
        "final_score": final,
        "semantic_similarity": semantic,
        "keyword_match": keyword,
        "jaccard_similarity": jaccard,
        "status": status,
        "feedback": feedback
    }