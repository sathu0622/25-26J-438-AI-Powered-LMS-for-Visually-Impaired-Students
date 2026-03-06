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
Task: Read the context and generate 1 Question, 1 Answer, 1 Key Phrase, and 3 Distractors for MCQ.
Constraint: Use the format below exactly.

Example:
QUESTION: Who was the first President of the USA?
ANSWER: George Washington.
KEY_PHRASE: George Washington
DISTRACTOR_1: Thomas Jefferson
DISTRACTOR_2: Benjamin Franklin
DISTRACTOR_3: John Adams

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
                        "content": """You are a History Teacher. Read the context and generate exactly one MCQ question.
Generate:
- QUESTION: A clear history question
- ANSWER: The correct answer
- KEY_PHRASE: A key phrase from the answer
- DISTRACTOR_1: A plausible but incorrect answer (similar to correct answer)
- DISTRACTOR_2: Another plausible but incorrect answer 
- DISTRACTOR_3: Another plausible but incorrect answer

The distractors should be realistic and related to the topic, making the MCQ challenging but fair.""",
                    },
                    {
                        "role": "user",
                        "content": f"Context: \"{context[:1500]}\"\n{avoid_instruction}\nFormat:\nQUESTION: ...\nANSWER: ...\nKEY_PHRASE: ...\nDISTRACTOR_1: ...\nDISTRACTOR_2: ...\nDISTRACTOR_3: ...",
                    },
                ],
                max_tokens=350,
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
        clean_resp = clean_resp.replace("Distractor_1:", "DISTRACTOR_1:").replace("Distractor_2:", "DISTRACTOR_2:").replace("Distractor_3:", "DISTRACTOR_3:")
        
        q_match = re.search(r"QUESTION:?\s*(.*?)(?=\n*ANSWER:|$)", clean_resp, re.IGNORECASE | re.DOTALL)
        a_match = re.search(r"ANSWER:?\s*(.*?)(?=\n*KEY_PHRASE:|$)", clean_resp, re.IGNORECASE | re.DOTALL)
        k_match = re.search(r"KEY_PHRASE:?\s*(.*?)(?=\n*DISTRACTOR_1:|$)", clean_resp, re.IGNORECASE | re.DOTALL)
        d1_match = re.search(r"DISTRACTOR_1:?\s*(.*?)(?=\n*DISTRACTOR_2:|$)", clean_resp, re.IGNORECASE | re.DOTALL)
        d2_match = re.search(r"DISTRACTOR_2:?\s*(.*?)(?=\n*DISTRACTOR_3:|$)", clean_resp, re.IGNORECASE | re.DOTALL)
        d3_match = re.search(r"DISTRACTOR_3:?\s*(.*?)(?=$)", clean_resp, re.IGNORECASE | re.DOTALL)

        q = q_match.group(1).strip() if q_match else ""
        a = a_match.group(1).strip() if a_match else ""
        k = k_match.group(1).strip() if k_match else ""
        d1 = d1_match.group(1).strip() if d1_match else ""
        d2 = d2_match.group(1).strip() if d2_match else ""
        d3 = d3_match.group(1).strip() if d3_match else ""

        # 3. Validation: Did we get a question?
        if len(q) > 5:
            if not a:
                # Fallback: Use first sentence from context
                context_text = str(context)
                a = context_text.split('.')[0].strip() + '.' if '.' in context_text else context_text.strip()
            
            # Build options array with correct answer and distractors
            distractors = [d for d in [d1, d2, d3] if d and len(d) > 2]
            
            # Ensure we have 3 distractors, generate fallbacks if needed
            while len(distractors) < 3:
                distractors.append(f"Option {len(distractors) + 2}")
            
            # Shuffle options and track correct answer position
            options = [a] + distractors[:3]
            random.shuffle(options)
            correct_index = options.index(a)
            
            if history_key not in question_history: question_history[history_key] = []
            question_history[history_key].append(q)
            return {
                "question": q, 
                "correct_answer": a, 
                "key_phrase": k,
                "options": options,
                "correct_index": correct_index
            }
        print(f"      ⚠️ Attempt {attempt+1} failed (Empty Question). Retrying...")

    # Fallback: Use context as answer if all attempts fail
    context_text = str(context)
    fallback_answer = context_text.split('.')[0].strip() + '.' if '.' in context_text else context_text.strip()
    fallback_options = [fallback_answer, "Option 2", "Option 3", "Option 4"]
    random.shuffle(fallback_options)
    return {
        "question": resp if resp else "No question generated.", 
        "correct_answer": fallback_answer, 
        "key_phrase": "",
        "options": fallback_options,
        "correct_index": fallback_options.index(fallback_answer)
    }


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


# ========== ROMAN NUMERAL / ORDINAL NORMALIZATION ==========
# Maps for converting between different numeral formats
ROMAN_TO_ARABIC = {
    'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
    'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10',
    'xi': '11', 'xii': '12', 'xiii': '13', 'xiv': '14', 'xv': '15',
    'xvi': '16', 'xvii': '17', 'xviii': '18', 'xix': '19', 'xx': '20',
}

# Ordinal words to arabic numbers
ORDINAL_TO_ARABIC = {
    'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
    'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10',
    'eleventh': '11', 'twelfth': '12', 'thirteenth': '13', 'fourteenth': '14', 'fifteenth': '15',
    '1st': '1', '2nd': '2', '3rd': '3', '4th': '4', '5th': '5',
    '6th': '6', '7th': '7', '8th': '8', '9th': '9', '10th': '10',
}

# Cardinal words to arabic numbers
CARDINAL_TO_ARABIC = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
}

def normalize_numerals(text: str) -> str:
    """
    Normalize Roman numerals, ordinals, and cardinal numbers to Arabic numerals.
    Examples:
        "Mihindu V" -> "mihindu 5"
        "Mihindu fifth" -> "mihindu 5"
        "Mihindu five" -> "mihindu 5"
        "Parakramabahu II" -> "parakramabahu 2"
        "King Parakramabahu the Great" -> "king parakramabahu the great"
    """
    if not text:
        return ""
    
    normalized = text.lower().strip()
    
    # Remove common prefixes/suffixes that don't affect meaning
    normalized = re.sub(r'\bking\s+', '', normalized)  # "King Mihindu V" -> "Mihindu V"
    normalized = re.sub(r'\bthe\s+great\b', '', normalized)  # "Parakramabahu the Great" -> "Parakramabahu"
    normalized = re.sub(r'\bthe\s+second\b', '2', normalized)  # "the second" -> "2"
    normalized = re.sub(r'\bthe\s+first\b', '1', normalized)  # "the first" -> "1"
    
    # Replace Roman numerals (must be at word boundaries, typically after a name)
    # Match Roman numerals at the end of words or standalone
    for roman, arabic in sorted(ROMAN_TO_ARABIC.items(), key=lambda x: -len(x[0])):
        # Match Roman numeral at end of string or followed by space/punctuation
        pattern = rf'\b{roman}\b'
        normalized = re.sub(pattern, arabic, normalized, flags=re.IGNORECASE)
    
    # Replace ordinals (first, second, 1st, 2nd, etc.)
    for ordinal, arabic in sorted(ORDINAL_TO_ARABIC.items(), key=lambda x: -len(x[0])):
        pattern = rf'\b{ordinal}\b'
        normalized = re.sub(pattern, arabic, normalized, flags=re.IGNORECASE)
    
    # Replace cardinal words (one, two, three, etc.)
    for cardinal, arabic in sorted(CARDINAL_TO_ARABIC.items(), key=lambda x: -len(x[0])):
        pattern = rf'\b{cardinal}\b'
        normalized = re.sub(pattern, arabic, normalized, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def evaluate_response(user_answer: str, correct_answer: str, key_phrase: str, is_mcq: bool = True):
    """
    Evaluate user's answer against the correct answer.
    
    For MCQ: Uses exact matching (binary: correct/incorrect)
    For Free-text: Uses SBERT semantic similarity with thresholds
    
    Handles Roman numerals and ordinals:
    "Mihindu V" == "Mihindu 5" == "Mihindu fifth" == "Mihindu five"
    """
    if not user_answer:
        return {
            "score": 0,
            "feedback": "Please select or type an answer!",
            "correct": False,
            "correct_answer": correct_answer or "Refer to context",
        }

    # Normalize both answers for comparison
    user_clean = user_answer.strip().lower()
    correct_clean = correct_answer.strip().lower() if correct_answer else ""
    
    # Additional normalization for Roman numerals and ordinals
    user_normalized = normalize_numerals(user_clean)
    correct_normalized = normalize_numerals(correct_clean)
    key_normalized = normalize_numerals(key_phrase.lower()) if key_phrase else ""

    # === MCQ EVALUATION (Exact Match) ===
    if is_mcq:
        # Direct exact match (basic)
        if user_clean == correct_clean:
            return {
                "score": 100,
                "feedback": "Correct! Well done!",
                "correct": True,
                "correct_answer": correct_answer,
            }
        
        # Normalized match (handles Roman numerals, ordinals)
        if user_normalized == correct_normalized:
            return {
                "score": 100,
                "feedback": "Correct! Well done!",
                "correct": True,
                "correct_answer": correct_answer,
            }
        
        # Check if key phrase matches (fallback for slight variations)
        if key_phrase and len(key_phrase) > 2:
            if key_phrase.lower().strip() == user_clean or key_normalized == user_normalized:
                return {
                    "score": 100,
                    "feedback": "Correct! Well done!",
                    "correct": True,
                    "correct_answer": correct_answer,
                }
        
        # MCQ is binary - no partial credit
        return {
            "score": 0,
            "feedback": "Incorrect. The correct answer was: " + correct_answer,
            "correct": False,
            "correct_answer": correct_answer,
        }

    # === FREE-TEXT EVALUATION (SBERT Semantic Similarity) ===
    target = correct_answer if correct_answer and len(correct_answer) > 2 else "Refer to context"

    # Quick win: exact match (basic)
    if user_clean == correct_clean:
        return {
            "score": 100,
            "feedback": "Perfect! Exact match!",
            "correct": True,
            "correct_answer": correct_answer,
        }
    
    # Quick win: normalized match (handles Roman numerals)
    if user_normalized == correct_normalized:
        return {
            "score": 100,
            "feedback": "Perfect! Exact match!",
            "correct": True,
            "correct_answer": correct_answer,
        }

    # Key phrase check (fast path) - check both regular and normalized
    if key_phrase and len(key_phrase) > 2:
        if key_phrase.lower() in user_clean or key_normalized in user_normalized:
            return {
                "score": 100,
                "feedback": "Excellent! You got it right!",
                "correct": True,
                "correct_answer": correct_answer,
            }

    # SBERT semantic similarity (use normalized versions for better matching)
    # Normalizing helps SBERT compare "Mihindu V" with "Mihindu 5" correctly
    emb1 = sbert_model.encode(user_normalized, convert_to_tensor=True)
    emb2 = sbert_model.encode(normalize_numerals(target.lower()), convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()

    # Thresholds for free-text evaluation
    if similarity >= 0.85:
        return {
            "score": 100,
            "feedback": "Excellent! Your answer is correct.",
            "correct": True,
            "correct_answer": correct_answer,
        }
    if similarity >= 0.70:
        return {
            "score": int(similarity * 100),
            "feedback": "Good answer! You captured the main idea.",
            "correct": True,
            "correct_answer": correct_answer,
        }
    if similarity >= 0.55:
        return {
            "score": int(similarity * 100),
            "feedback": "Partially correct. Some details are missing.",
            "correct": False,
            "correct_answer": correct_answer,
        }

    return {
        "score": int(similarity * 100),
        "feedback": "Incorrect. Please review and try again.",
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
    return evaluate_response(req.user_answer, req.correct_answer, req.key_phrase, is_mcq=True)


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
        is_mcq=True,  # Quiz sets use MCQ format
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


# === GENERATIVE FREE-TEXT QUIZ ENDPOINTS ===
# Questions generated one-by-one, evaluated with SBERT, saved for retake

freetext_sessions_col = db["freetext_sessions"]


class FreeTextStartRequest(BaseModel):
    username: str
    chapter_name: str
    session_id: Optional[str] = None  # For retake


class FreeTextAnswerRequest(BaseModel):
    username: str
    session_id: str
    user_answer: str


class FreeTextFinishRequest(BaseModel):
    username: str
    session_id: str


@router.get("/freetext/chapters")
def get_freetext_chapters():
    """Get available chapters for free-text quiz"""
    return {"chapters": AVAILABLE_CHAPTERS}


@router.post("/freetext/start")
def start_freetext_session(req: FreeTextStartRequest):
    """Start a new free-text quiz session or resume existing one"""
    
    # If session_id provided, resume existing session
    if req.session_id:
        try:
            session_oid = ObjectId(req.session_id)
        except Exception:
            raise HTTPException(400, "Invalid session ID")
        
        session = freetext_sessions_col.find_one({"_id": session_oid, "username": req.username})
        if not session:
            raise HTTPException(404, "Session not found")
        
        # For retake, create a new attempt
        attempt_id = str(ObjectId())
        attempt_entry = {
            "attempt_id": attempt_id,
            "started_at": datetime.utcnow(),
            "answers": [],
            "completed_at": None,
            "summary": None,
        }
        
        freetext_sessions_col.update_one(
            {"_id": session_oid},
            {"$push": {"attempts": attempt_entry}}
        )
        
        # Return first question for retake
        questions = session.get("questions", [])
        first_question = questions[0] if questions else None
        
        return {
            "session_id": str(session_oid),
            "attempt_id": attempt_id,
            "chapter_name": session.get("chapter_name", ""),
            "question_index": 0,
            "current_question": first_question,
            "total_questions": len(questions),
            "is_retake": True,
        }
    
    # Create new session
    first_question = run_ai_generation(req.chapter_name)
    if not first_question:
        raise HTTPException(500, "Failed to generate first question")
    
    # Remove MCQ options for free-text mode
    question_data = {
        "question": first_question["question"],
        "correct_answer": first_question["correct_answer"],
        "key_phrase": first_question.get("key_phrase", ""),
    }
    
    attempt_id = str(ObjectId())
    session_doc = {
        "username": req.username,
        "chapter_name": req.chapter_name,
        "questions": [question_data],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "attempts": [{
            "attempt_id": attempt_id,
            "started_at": datetime.utcnow(),
            "answers": [],
            "completed_at": None,
            "summary": None,
        }],
    }
    
    result = freetext_sessions_col.insert_one(session_doc)
    session_id = str(result.inserted_id)
    
    return {
        "session_id": session_id,
        "attempt_id": attempt_id,
        "chapter_name": req.chapter_name,
        "question_index": 0,
        "current_question": question_data,
        "total_questions": 1,
        "is_retake": False,
    }


@router.post("/freetext/answer")
def submit_freetext_answer(req: FreeTextAnswerRequest):
    """Submit answer and get next question"""
    try:
        session_oid = ObjectId(req.session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID")
    
    session = freetext_sessions_col.find_one({"_id": session_oid, "username": req.username})
    if not session:
        raise HTTPException(404, "Session not found")
    
    # Get current attempt (latest)
    attempts = session.get("attempts", [])
    if not attempts:
        raise HTTPException(400, "No active attempt")
    
    current_attempt = attempts[-1]
    if current_attempt.get("completed_at"):
        raise HTTPException(400, "Attempt already completed")
    
    attempt_id = current_attempt["attempt_id"]
    answers = current_attempt.get("answers", [])
    question_index = len(answers)
    questions = session.get("questions", [])
    
    # Get current question
    if question_index >= len(questions):
        raise HTTPException(400, "No more questions available")
    
    current_question = questions[question_index]
    
    # Evaluate with SBERT (free-text mode)
    evaluation = evaluate_response(
        req.user_answer,
        current_question.get("correct_answer", ""),
        current_question.get("key_phrase", ""),
        is_mcq=False,  # Use SBERT evaluation
    )
    
    # Save answer
    answer_entry = {
        "question_index": question_index,
        "question": current_question.get("question", ""),
        "user_answer": req.user_answer,
        "correct_answer": current_question.get("correct_answer", ""),
        "score": evaluation["score"],
        "correct": evaluation["correct"],
        "feedback": evaluation["feedback"],
        "answered_at": datetime.utcnow(),
    }
    
    freetext_sessions_col.update_one(
        {"_id": session_oid, "attempts.attempt_id": attempt_id},
        {"$push": {"attempts.$.answers": answer_entry}}
    )
    
    return {
        "question_index": question_index,
        "score": evaluation["score"],
        "correct": evaluation["correct"],
        "feedback": evaluation["feedback"],
        "correct_answer": current_question.get("correct_answer", ""),
        "user_answer": req.user_answer,
    }


@router.post("/freetext/next")
def get_next_freetext_question(req: FreeTextAnswerRequest):
    """Generate and return next question"""
    try:
        session_oid = ObjectId(req.session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID")
    
    session = freetext_sessions_col.find_one({"_id": session_oid, "username": req.username})
    if not session:
        raise HTTPException(404, "Session not found")
    
    chapter_name = session.get("chapter_name", "")
    questions = session.get("questions", [])
    
    # Get current attempt
    attempts = session.get("attempts", [])
    if not attempts:
        raise HTTPException(400, "No active attempt")
    
    current_attempt = attempts[-1]
    answers = current_attempt.get("answers", [])
    next_index = len(answers)
    
    # Check if this is a retake - use existing questions
    is_retake = next_index < len(questions)
    
    if is_retake:
        # Return existing question for retake
        next_question = questions[next_index]
    else:
        # Generate new question
        new_question = run_ai_generation(chapter_name)
        if not new_question:
            raise HTTPException(500, "Failed to generate question")
        
        next_question = {
            "question": new_question["question"],
            "correct_answer": new_question["correct_answer"],
            "key_phrase": new_question.get("key_phrase", ""),
        }
        
        # Save new question to session
        freetext_sessions_col.update_one(
            {"_id": session_oid},
            {
                "$push": {"questions": next_question},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
    
    return {
        "question_index": next_index,
        "current_question": next_question,
        "total_questions": len(questions) + (0 if is_retake else 1),
        "is_retake": is_retake,
    }


@router.post("/freetext/finish")
def finish_freetext_session(req: FreeTextFinishRequest):
    """Finish the free-text quiz session and save summary"""
    try:
        session_oid = ObjectId(req.session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID")
    
    session = freetext_sessions_col.find_one({"_id": session_oid, "username": req.username})
    if not session:
        raise HTTPException(404, "Session not found")
    
    # Get current attempt
    attempts = session.get("attempts", [])
    if not attempts:
        raise HTTPException(400, "No active attempt")
    
    current_attempt = attempts[-1]
    attempt_id = current_attempt["attempt_id"]
    answers = current_attempt.get("answers", [])
    
    # Calculate summary
    total_questions = len(answers)
    correct_count = sum(1 for ans in answers if ans.get("correct"))
    total_score = sum(ans.get("score", 0) for ans in answers)
    average_score = int(total_score / total_questions) if total_questions > 0 else 0
    
    summary = {
        "correct_count": correct_count,
        "total_questions": total_questions,
        "average_score": average_score,
    }
    
    # Update attempt with summary
    freetext_sessions_col.update_one(
        {"_id": session_oid, "attempts.attempt_id": attempt_id},
        {
            "$set": {
                "attempts.$.summary": summary,
                "attempts.$.completed_at": datetime.utcnow(),
            }
        }
    )
    
    return {
        "session_id": req.session_id,
        "attempt_id": attempt_id,
        "summary": summary,
        "answers": answers,
    }


@router.get("/freetext/sessions/{username}")
def list_freetext_sessions(username: str):
    """List all free-text quiz sessions for a user"""
    docs = list(freetext_sessions_col.find({"username": username}).sort("created_at", -1))
    payload = []
    
    for doc in docs:
        attempts = doc.get("attempts", [])
        completed_attempts = [a for a in attempts if a.get("completed_at")]
        latest = completed_attempts[-1] if completed_attempts else None
        
        payload.append({
            "session_id": str(doc.get("_id")),
            "chapter_name": doc.get("chapter_name", ""),
            "created_at": to_iso(doc.get("created_at")),
            "questions_count": len(doc.get("questions", [])),
            "attempts_count": len(completed_attempts),
            "latest_attempt": {
                "attempt_id": latest.get("attempt_id"),
                "summary": latest.get("summary"),
                "completed_at": to_iso(latest.get("completed_at")),
            } if latest else None,
        })
    
    return {"sessions": payload}


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
    
    # Normalize Roman numerals, ordinals, and cardinal numbers
    # "Mihindu V" == "Mihindu 5" == "Mihindu fifth" == "Mihindu five"
    user_normalized = normalize_numerals(user_answer)
    correct_normalized = normalize_numerals(correct_answer)
    
    if not user_answer:
        return {
            "score": 0,
            "feedback": "No answer provided. Please provide an answer to continue.",
            "correct": False,
            "similarity_score": 0.0
        }
    
    # Quick win: exact match after normalization
    if user_normalized == correct_normalized:
        return {
            "score": 100,
            "feedback": "Perfect! Exact match!",
            "correct": True,
            "similarity_score": 1.0,
            "correct_answer": req.correct_answer
        }
    
    try:
        # Generate embeddings using normalized versions for better matching
        user_embedding = sbert_model.encode([user_normalized])
        correct_embedding = sbert_model.encode([correct_normalized])
        
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
