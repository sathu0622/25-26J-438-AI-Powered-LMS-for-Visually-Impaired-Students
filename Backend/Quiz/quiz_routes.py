import random
import re
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import util
from dotenv import load_dotenv

# These will be set by main.py
sbert_model = None
df_syl = None
df_past_papers = None  # For past paper questions
AVAILABLE_CHAPTERS = []
PAST_PAPER_CHAPTERS = []  # For past paper chapters
prefetch_cache = {}
question_history = {}
llm = None  # Injected Llama model

load_dotenv()
client = MongoClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DATABASE_NAME", "vis_history")]
quiz_sets_col = db["quiz_sets"]

router = APIRouter()

class ChapterRequest(BaseModel):
    chapter_name: str

class AnswerRequest(BaseModel):
    user_answer: str
    correct_answer: str = "Refer to context"
    key_phrase: str = ""
    chapter_name: str

class StartSetRequest(BaseModel):
    username: str
    chapter_name: str
    set_id: Optional[str] = None

class AnswerSetRequest(BaseModel):
    username: str
    user_answer: str
    question_index: int

class CompleteAttemptRequest(BaseModel):
    username: str

class PastPaperChapterRequest(BaseModel):
    chapter_name: str

class PastPaperAnswerRequest(BaseModel):
    user_answer: str
    correct_answer: str
    question: str
    year: str

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

    if llm is None:
        raise HTTPException(500, "Model not initialized")

    # retry logic
    for attempt in range(2):
        try:
            chat_resp = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a History Teacher. Read the context and generate exactly one QUESTION, one ANSWER, and one KEY_PHRASE in the specified format.",
                    },
                    {
                        "role": "user",
                        "content": f"Context: \"{context[:1500]}\"\n{avoid_instruction}\nFormat:\nQUESTION: ...\nANSWER: ...\nKEY_PHRASE: ...",
                    },
                ],
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            resp = chat_resp["choices"][0]["message"]["content"]
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


def generate_question_batch(chapter_name: str, total: int = 10) -> List[Dict[str, str]]:
    """Generate a batch of unique questions for a quiz set."""
    questions: List[Dict[str, str]] = []
    attempts = 0
    seen: set[str] = set()

    while len(questions) < total and attempts < total * 5:
        attempts += 1
        result = run_ai_generation(chapter_name)
        if not result or not result.get("question"):
            continue

        question_text = result["question"].strip()
        if question_text.lower() in seen:
            continue

        seen.add(question_text.lower())
        questions.append(result)

    if len(questions) < total:
        raise HTTPException(500, f"Only generated {len(questions)} questions. Please try again.")

    return questions


def evaluate_response(user_answer: str, correct_answer: str, key_phrase: str):
    if not user_answer:
        return {
            "score": 0,
            "feedback": "Please type an answer!",
            "correct": False,
            "correct_answer": correct_answer or "Refer to context",
        }

    target = correct_answer if correct_answer and len(correct_answer) > 2 else "Refer to context"

    if key_phrase and len(key_phrase) > 2:
        if key_phrase.lower() in user_answer.lower():
            return {
                "score": 100,
                "feedback": "🎯 Excellent! You got it right!",
                "correct": True,
                "correct_answer": correct_answer,
            }

    emb1 = sbert_model.encode(user_answer, convert_to_tensor=True)
    emb2 = sbert_model.encode(target, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()

    if score > 0.75:
        return {
            "score": int(score * 100),
            "feedback": "Your Answer is Correct",
            "correct": True,
            "correct_answer": correct_answer,
        }
    if score > 0.60:
        return {
            "score": int(score * 100),
            "feedback": "You are Partially Correct, You have missed some details",
            "correct": True,
            "correct_answer": correct_answer,
        }

    return {
        "score": int(score * 100),
        "feedback": "Your Answer is Incorrect",
        "correct": False,
        "correct_answer": correct_answer,
    }


def to_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(400, "Invalid ID format")


def to_iso(dt: Optional[datetime]):
    return dt.isoformat() if dt else None

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
    return evaluate_response(req.user_answer, req.correct_answer, req.key_phrase)


@router.post("/quiz_sets/start")
def start_quiz_set(req: StartSetRequest):
    chapter_name = req.chapter_name

    if req.set_id:
        set_oid = to_object_id(req.set_id)
        doc = quiz_sets_col.find_one({"_id": set_oid})
        if not doc:
            raise HTTPException(404, "Quiz set not found")
        if doc.get("username") != req.username:
            raise HTTPException(403, "Access denied for this quiz set")
    else:
        questions = generate_question_batch(chapter_name, total=10)
        doc = {
            "username": req.username,
            "chapter_name": chapter_name,
            "questions": questions,
            "created_at": datetime.utcnow(),
            "attempts": [],
        }
        insert_result = quiz_sets_col.insert_one(doc)
        doc["_id"] = insert_result.inserted_id

    attempt_id = str(ObjectId())
    attempt_entry = {
        "attempt_id": attempt_id,
        "started_at": datetime.utcnow(),
        "answers": [],
        "completed_at": None,
        "summary": None,
    }

    quiz_sets_col.update_one({"_id": doc["_id"]}, {"$push": {"attempts": attempt_entry}})

    return {
        "set_id": str(doc["_id"]),
        "attempt_id": attempt_id,
        "chapter_name": doc.get("chapter_name", ""),
        "questions": doc.get("questions", []),
        "total_questions": len(doc.get("questions", [])),
    }


@router.post("/quiz_sets/{set_id}/attempts/{attempt_id}/answer")
def answer_quiz_set(set_id: str, attempt_id: str, req: AnswerSetRequest):
    set_oid = to_object_id(set_id)
    doc = quiz_sets_col.find_one({"_id": set_oid, "username": req.username})
    if not doc:
        raise HTTPException(404, "Quiz set not found")

    questions = doc.get("questions", [])
    if req.question_index < 0 or req.question_index >= len(questions):
        raise HTTPException(400, "Invalid question index")

    target_question = questions[req.question_index]
    evaluation = evaluate_response(
        req.user_answer,
        target_question.get("correct_answer", ""),
        target_question.get("key_phrase", ""),
    )

    answer_entry = {
        "question_index": req.question_index,
        "user_answer": req.user_answer,
        "score": evaluation["score"],
        "correct": evaluation["correct"],
        "feedback": evaluation["feedback"],
        "correct_answer": target_question.get("correct_answer", ""),
        "answered_at": datetime.utcnow(),
    }

    update_result = quiz_sets_col.update_one(
        {"_id": set_oid, "attempts.attempt_id": attempt_id},
        {"$push": {"attempts.$.answers": answer_entry}},
    )

    if update_result.matched_count == 0:
        raise HTTPException(404, "Attempt not found")

    return {**evaluation, "question_index": req.question_index}


@router.post("/quiz_sets/{set_id}/attempts/{attempt_id}/complete")
def complete_quiz_attempt(set_id: str, attempt_id: str, req: CompleteAttemptRequest):
    set_oid = to_object_id(set_id)
    doc = quiz_sets_col.find_one({"_id": set_oid, "username": req.username})
    if not doc:
        raise HTTPException(404, "Quiz set not found")

    attempts = doc.get("attempts", [])
    attempt = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
    if not attempt:
        raise HTTPException(404, "Attempt not found")

    answers = attempt.get("answers", [])
    total_questions = len(doc.get("questions", []))
    correct_count = sum(1 for ans in answers if ans.get("correct"))
    avg_score = int(sum(ans.get("score", 0) for ans in answers) / len(answers)) if answers else 0

    summary = {
        "correct_count": correct_count,
        "total_questions": total_questions,
        "average_score": avg_score,
    }

    quiz_sets_col.update_one(
        {"_id": set_oid, "attempts.attempt_id": attempt_id},
        {
            "$set": {
                "attempts.$.summary": summary,
                "attempts.$.completed_at": datetime.utcnow(),
            }
        },
    )

    return {
        "set_id": str(set_oid),
        "attempt_id": attempt_id,
        "summary": summary,
    }


@router.get("/quiz_sets/user/{username}")
def list_quiz_sets(username: str):
    docs = list(quiz_sets_col.find({"username": username}).sort("created_at", -1))
    payload = []

    for doc in docs:
        attempts = doc.get("attempts", [])
        attempts_sorted = sorted(
            attempts,
            key=lambda a: a.get("completed_at") or a.get("started_at"),
            reverse=True,
        )
        latest = attempts_sorted[0] if attempts_sorted else None

        payload.append(
            {
                "set_id": str(doc.get("_id")),
                "chapter_name": doc.get("chapter_name", ""),
                "created_at": to_iso(doc.get("created_at")),
                "questions_count": len(doc.get("questions", [])),
                "latest_attempt":
                    {
                        "attempt_id": latest.get("attempt_id"),
                        "summary": latest.get("summary"),
                        "completed_at": to_iso(latest.get("completed_at")),
                    }
                    if latest
                    else None,
            }
        )

    return {"quiz_sets": payload}

# === PAST PAPER QUIZ ENDPOINTS ===

@router.get("/past-paper/chapters")
def get_past_paper_chapters():
    """Get available chapters from past paper questions"""
    return {"chapters": PAST_PAPER_CHAPTERS}

@router.post("/past-paper/questions")
def get_past_paper_questions(req: PastPaperChapterRequest):
    """Get all past paper questions for a specific chapter"""
    if df_past_papers is None:
        raise HTTPException(500, "Past paper data not loaded")
    
    # Filter questions by chapter
    chapter_questions = df_past_papers[df_past_papers['Chapter'] == req.chapter_name]
    
    if chapter_questions.empty:
        raise HTTPException(404, f"No past paper questions found for chapter: {req.chapter_name}")
    
    # Convert to list of dictionaries
    questions = []
    for _, row in chapter_questions.iterrows():
        questions.append({
            "question": row['Question'],
            "correct_answer": row['CorrectAnswer'], 
            "unique_part": row['UniquePart'],
            "year": str(row['Year']),
            "chapter": row['Chapter']
        })
    
    return {"questions": questions}

@router.post("/past-paper/evaluate")
def evaluate_past_paper_answer(req: PastPaperAnswerRequest):
    """Evaluate past paper answer using SBERT similarity matching"""
    if sbert_model is None:
        raise HTTPException(500, "SBERT model not initialized")
    
    # Clean and prepare texts for comparison
    user_answer = req.user_answer.strip().lower()
    correct_answer = req.correct_answer.strip().lower()
    
    if not user_answer:
        return {
            "score": 0,
            "feedback": "No answer provided. Please provide an answer to continue.",
            "correct": False,
            "similarity_score": 0.0
        }
    
    try:
        # Generate embeddings
        user_embedding = sbert_model.encode([user_answer])
        correct_embedding = sbert_model.encode([correct_answer])
        
        # Calculate cosine similarity
        similarity = util.cos_sim(user_embedding, correct_embedding)[0][0].item()
        
        # Convert similarity to percentage score (0-100)
        score = max(0, min(100, int(similarity * 100)))
        
        # Generate feedback based on similarity score
        if similarity >= 0.85:
            feedback = "Excellent! Your answer demonstrates a comprehensive understanding of the topic."
            correct = True
        elif similarity >= 0.70:
            feedback = "Good answer! You've captured the main concepts well."
            correct = True
        elif similarity >= 0.50:
            feedback = "Fair attempt. Your answer partially addresses the question but could be more complete."
            correct = False
        elif similarity >= 0.30:
            feedback = "The answer doesn't adequately address the question. Please review the topic and try again."
            correct = False
        else:
            feedback = "The answer doesn't adequately address the question. Please review the topic and try again."
            correct = False
        
        return {
            "score": score,
            "feedback": feedback,
            "correct": correct,
            "similarity_score": round(similarity, 4),
            "correct_answer": req.correct_answer
        }
        
    except Exception as e:
        print(f"Error in past paper evaluation: {str(e)}")
        return {
            "score": 0,
            "feedback": "Error occurred during evaluation. Please try again.",
            "correct": False,
            "similarity_score": 0.0
        }

def load_past_paper_data():
    """Load past paper questions from CSV file"""
    global df_past_papers, PAST_PAPER_CHAPTERS
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        past_paper_file = os.path.join(base_dir, "PastPaperQuestions.csv")
        
        if not os.path.exists(past_paper_file):
            print(f"⚠️ Past paper file not found at: {past_paper_file}")
            return
        
        # Try multiple encodings to handle different file formats
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        df_past_papers = None
        
        for encoding in encodings:
            try:
                print(f"🔍 Trying to load CSV with {encoding} encoding...")
                df_past_papers = pd.read_csv(past_paper_file, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                print(f"⚠️ Failed with {encoding}: {str(e)}")
                continue
            except Exception as e:
                print(f"⚠️ Error with {encoding}: {str(e)}")
                continue
        
        if df_past_papers is None:
            raise Exception("Could not load CSV file with any supported encoding")
            
        PAST_PAPER_CHAPTERS = sorted(df_past_papers['Chapter'].unique().tolist())
        
        print(f"✅ Loaded {len(df_past_papers)} past paper questions from {len(PAST_PAPER_CHAPTERS)} chapters")
        print(f"📚 Past Paper Chapters: {PAST_PAPER_CHAPTERS}")
        
    except Exception as e:
        print(f"❌ Error loading past paper data: {str(e)}")
        df_past_papers = None
        PAST_PAPER_CHAPTERS = []
