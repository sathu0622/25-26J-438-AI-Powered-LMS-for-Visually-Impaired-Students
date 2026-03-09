from utils import semantic_similarity, keyword_overlap_score, jaccard_similarity, length_penalty
from logger_config import logger
import torch

def generate_correct_answer(question, tokenizer, model):
    torch.manual_seed(42) # Same answer each time
    prompt = f"""You are an expert Sri Lankan O/L History teacher.
Write a concise, factual exam answer for a Grade 11 student.
- Cover only the most important historical facts
- Use clear, simple language
- Maximum 200 words. Stop writing after your last point.

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt") # Converts text into tokens
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    logger.info("Generating answer...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0,
            num_beams=4,
            repetition_penalty=1.2,     #avoids repeating words
            early_stopping=True,    #stops when complete
            pad_token_id=tokenizer.eos_token_id,    #proper stopping
            eos_token_id=tokenizer.eos_token_id,    #proper stopping
)

    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(    # Converts tokens back into text.
        outputs[0][input_length:],
        skip_special_tokens=True
    )
    answer = response.strip()

    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    if sentences:
        answer = '. '.join(sentences) + '.'

    return answer

def calculate_final_score(correct, student, sbert_model):
    semantic = semantic_similarity(correct, student, sbert_model)
    keyword = keyword_overlap_score(correct, student)
    jaccard = jaccard_similarity(correct, student)

    length_factor = length_penalty(correct, student)

    semantic_weighted = semantic * 0.55
    keyword_weighted = keyword * 0.35
    jaccard_weighted = jaccard * 0.10

    final = (semantic_weighted + keyword_weighted + jaccard_weighted) * length_factor

    return round(final, 2), round(semantic_weighted, 2), round(keyword_weighted, 2), round(jaccard_weighted, 2)

def generate_feedback(score):
    if score >= 75:
        return "Excellent answer. You have demonstrated a strong and accurate understanding of the historical topic, with well-explained points and relevant details."
    elif score >= 45:
        return "Good attempt. You have shown a basic understanding of the topic, but some important facts, explanations, or supporting details are missing."
    return "The answer shows limited understanding of the topic. Several key points are incorrect or incomplete. Please review the lesson and try again."

def evaluate_student_answer(question, student_answer, tokenizer, model, sbert):
    correct_answer = generate_correct_answer(question, tokenizer, model)
    final, semantic, keyword, jaccard = calculate_final_score(correct_answer, student_answer, sbert)

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