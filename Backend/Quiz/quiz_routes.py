import random
import re
import ollama
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sentence_transformers import util

# These will be set by main.py
sbert_model = None
df_syl = None
AVAILABLE_CHAPTERS = []
prefetch_cache = {}
question_history = {}
OLLAMA_MODEL_NAME = "history-tutor"  # Default, can be overwritten by main.py

router = APIRouter()

class ChapterRequest(BaseModel):
    chapter_name: str

class AnswerRequest(BaseModel):
    user_answer: str
    correct_answer: str = "Refer to context"
    key_phrase: str = ""
    chapter_name: str

# --- 4. Question genaration ---
def run_ai_generation(chapter_name: str):
    print(f"   Generating question for: {chapter_name}...")
    
    subset = df_syl[df_syl['Chapter_Clean'] == chapter_name]
    if subset.empty: return None

    all_indices = subset.index.tolist()
    idx = random.choice(all_indices)
    context = df_syl.loc[idx]['Context']

    # Anti-Duplicate
    history_key = (chapter_name, idx)
    previous_qs = question_history.get(history_key, [])
    avoid_instruction = ""
    if previous_qs:
        avoid_list = "; ".join(previous_qs[-3:])
        avoid_instruction = f"Do not ask about: {avoid_list}."

    #  Gives the AI an example to copy)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a History Teacher. 
Task: Read the context and generate 1 Question, 1 Answer, and 1 Key Phrase.
Constraint: Use the format below exactly.

Example:
QUESTION: Who was the first President of the USA?
ANSWER: George Washington.
KEY_PHRASE: George Washington

Your Turn:<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: "{context[:1500]}"
{avoid_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
QUESTION:"""

    # retry logic
    for attempt in range(2):
        try:
            response_obj = ollama.generate(model=OLLAMA_MODEL_NAME, prompt=prompt)
            resp = response_obj['response']
        except Exception as e:
            print(f"Generation Error: {e}")
            resp = ""

        # --- PARSING LOGIC ---
        clean_resp = resp.replace("OUTPUT", "").replace("###", "").strip()
        clean_resp = clean_resp.replace("Answer:", "ANSWER:").replace("Key_Phrase:", "KEY_PHRASE:")
        q_match = re.search(r"QUESTION:?\s*(.*?)(?=\n*ANSWER:|$)", clean_resp, re.IGNORECASE | re.DOTALL)
        a_match = re.search(r"ANSWER:?\s*(.*?)(?=\n*KEY_PHRASE:|$)", clean_resp, re.IGNORECASE | re.DOTALL)
        k_match = re.search(r"KEY_PHRASE:?\s*(.*?)(?=$)", clean_resp, re.IGNORECASE | re.DOTALL)

        q = q_match.group(1).strip() if q_match else ""
        a = a_match.group(1).strip() if a_match else ""
        k = k_match.group(1).strip() if k_match else ""

        # 3. Validation: Did we get a question?
        if len(q) > 5:
            if not a:
                # Fallback: Use first sentence from context
                context_text = str(context)
                a = context_text.split('.')[0].strip() + '.' if '.' in context_text else context_text.strip()
            if history_key not in question_history: question_history[history_key] = []
            question_history[history_key].append(q)
            return {"question": q, "correct_answer": a, "key_phrase": k}
        print(f"      ⚠️ Attempt {attempt+1} failed (Empty Question). Retrying...")

    # Fallback: Use context as answer if all attempts fail
    context_text = str(context)
    fallback_answer = context_text.split('.')[0].strip() + '.' if '.' in context_text else context_text.strip()
    return {"question": resp if resp else "No question generated.", "correct_answer": fallback_answer, "key_phrase": ""}

def prefetch_next_question(chapter_name: str):
    result = run_ai_generation(chapter_name)
    if result:
        prefetch_cache[chapter_name] = result

@router.get("/chapters")
def get_chapters():
    return {"chapters": AVAILABLE_CHAPTERS}

@router.post("/generate_question")
def generate_question(req: ChapterRequest):
    if req.chapter_name in prefetch_cache:
        print(f"    Cache Hit!")
        return prefetch_cache.pop(req.chapter_name)

    print(f"    Cache Miss. Generating...")
    result = run_ai_generation(req.chapter_name)
    if not result: raise HTTPException(500, "Generation Failed")
    return result

@router.post("/evaluate_answer")
def evaluate_answer(req: AnswerRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(prefetch_next_question, req.chapter_name)
    
    if not req.user_answer:
        return {"score": 0, "feedback": "Please type an answer!", "correct": False, "correct_answer": req.correct_answer}

    target = req.correct_answer if req.correct_answer and len(req.correct_answer) > 2 else "Refer to context"

    if req.key_phrase and len(req.key_phrase) > 2:
        if req.key_phrase.lower() in req.user_answer.lower():
            return {"score": 100, "feedback": "🎯 Excellent! You got it right!", "correct": True, "correct_answer": req.correct_answer}

    emb1 = sbert_model.encode(req.user_answer, convert_to_tensor=True)
    emb2 = sbert_model.encode(target, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()

    if score > 0.75:
        return {"score": int(score*100), "feedback": "✅ Your Answer is Correct", "correct": True, "correct_answer": req.correct_answer}
    if score > 0.60:
        return {"score": int(score*100), "feedback": "⚠️ You are Partially Correct, You have missed some details", "correct": True, "correct_answer": req.correct_answer}

    return {"score": int(score*100), "feedback": "❌ Your Answer is Incorrect", "correct": False, "correct_answer": req.correct_answer}
