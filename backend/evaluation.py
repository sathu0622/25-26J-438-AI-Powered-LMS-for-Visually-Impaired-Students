from utils import semantic_similarity, keyword_overlap_score, jaccard_similarity, length_penalty, detect_historical_errors
from logger_config import logger
import torch

def generate_correct_answer(question, tokenizer, model):
    torch.manual_seed(42)
    prompt = f"""You are an expert Sri Lankan O/L History teacher.
Answer the question clearly and factually in a detailed, exam-style narrative suitable for a Grade 11 student.
Include all key historical points, but keep the language simple.
Limit the answer to approximately 200 words.
Do NOT mention the word limit or any instructions in the answer.
Do NOT add unnecessary commentary or endnotes.

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    logger.info("Generating answer...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.5,
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def calculate_final_score(correct, student, sbert_model):
    semantic = semantic_similarity(correct, student, sbert_model)
    keyword = keyword_overlap_score(correct, student)
    jaccard = jaccard_similarity(correct, student)

    length_factor = length_penalty(correct, student)
    error_factor = detect_historical_errors(student)

    final = (semantic*0.65 + keyword*0.25 + jaccard*0.1) * length_factor * error_factor
    if semantic >= 60 and keyword >= 50 and error_factor == 1.0:
        final += 5

    return round(final,2), semantic, keyword, jaccard, error_factor

def generate_feedback(score):
    if score >= 75: return "Excellent answer with correct historical understanding."
    elif score >= 45: return "Basic understanding shown, but key facts are missing."
    return "Incorrect or weak answer. Please revise the lesson."

def evaluate_student_answer(question, student_answer, tokenizer, model, sbert):
    correct_answer = generate_correct_answer(question, tokenizer, model)
    final, semantic, keyword, jaccard, error_factor = calculate_final_score(correct_answer, student_answer, sbert)

    status = "PASS" if final >= 75 else "NEEDS IMPROVEMENT" if final >= 45 else "FAIL"
    return {
        "question": question,
        "student_answer": student_answer,
        "model_answer": correct_answer,
        "final_score": final,
        "semantic_similarity": semantic,
        "keyword_match": keyword,
        "jaccard_similarity": jaccard,
        "error_penalty": f"{int(error_factor*100)}%",
        "status": status,
        "feedback": generate_feedback(final)
    }
