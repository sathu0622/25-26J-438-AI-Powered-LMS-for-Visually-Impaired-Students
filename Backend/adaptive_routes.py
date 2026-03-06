import os
import re
import math
import random
import threading
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import util
from dotenv import load_dotenv


# Lightweight 2PL adaptive quiz using a static item bank loaded from QuizDataset.csv

# These will be set by main.py
sbert_model = None
llm = None  # Injected Llama model for distractor generation

# Cache for generated distractors to avoid regenerating
_distractor_cache: Dict[str, List[str]] = {}

# Thread safety for LLM generation
_llm_lock = threading.Lock()


def _clean_option_text(text: str) -> str:
    """
    Clean up LLM artifacts from parsed option text.
    Removes common patterns like 'Answer:', 'Difficulty:', etc.
    """
    if not text:
        return ""
    
    cleaned = text.strip()
    
    # Remove common LLM artifacts (case-insensitive)
    artifacts = [
        r'^answer:\s*', r'^correct\s*answer:\s*', r'^key\s*phrase:\s*',
        r'^system[_\s]*answer:\s*', r'^difficulty:\s*\w*\s*', r'^distractor[_\s]*\d*:\s*',
        r'^option[_\s]*\d*:\s*', r'^choice[_\s]*\d*:\s*',
        r'^\d+\.\s*', r'^[a-d]\)\s*', r'^[a-d]\.\s*',
        r'^\*+\s*', r'^-+\s*'
    ]
    
    # Run iteratively until no more changes (handles nested artifacts like "Option A: Answer: ...")
    prev = ""
    while prev != cleaned:
        prev = cleaned
        for pattern in artifacts:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
    
    # Remove trailing artifacts
    trailing_artifacts = [
        r'\s*difficulty:\s*\w*\s*$', r'\s*\(correct\)\s*$',
        r'\s*\[correct\]\s*$', r'\s*correct\s*$',
    ]
    
    for pattern in trailing_artifacts:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
    
    # Remove any remaining colons at the start
    cleaned = re.sub(r'^:\s*', '', cleaned).strip()
    
    return cleaned

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ITEM_BANK_FILE = os.path.join(BASE_DIR, "QuizDataset.csv")

load_dotenv()
client = MongoClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DATABASE_NAME", "vis_history")]
adaptive_sessions_col = db["adaptive_sessions"]

router = APIRouter(prefix="/adaptive")


class AdaptiveStartRequest(BaseModel):
    username: str
    chapter_name: str


class AdaptiveAnswerRequest(BaseModel):
    username: str
    session_id: str
    item_id: str
    user_answer: str


class AdaptiveFinishRequest(BaseModel):
    username: str
    session_id: str


class AdaptiveItem(BaseModel):
    item_id: str
    chapter_name: str
    question: str
    correct_answer: str
    difficulty: float
    difficulty_label: str
    discrimination: float
    context: str
    key_phrase: Optional[str] = ""
    options: Optional[List[str]] = None  # MCQ options (4 choices)
    correct_index: Optional[int] = None  # Index of correct answer in options


_difficulty_map = {
    "easy": -1.0,
    "medium": 0.0,
    "hard": 1.0,
}
LEVEL_ORDER = ["easy", "medium", "hard"]


def _load_item_bank() -> Dict[str, List[AdaptiveItem]]:
    if not os.path.exists(ITEM_BANK_FILE):
        raise RuntimeError("QuizDataset.csv missing; required for adaptive quiz")

    df = pd.read_csv(
        ITEM_BANK_FILE,
        encoding="utf-8",
        encoding_errors="replace",
    )
    required_cols = {"Chapter", "Question", "CorrectAnswer", "Context", "Difficulty"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError("QuizDataset.csv missing required columns Chapter, Question, CorrectAnswer, Context, Difficulty")

    bank: Dict[str, List[AdaptiveItem]] = {}
    for idx, row in df.iterrows():
        chapter = str(row["Chapter"]).strip()
        difficulty_label = str(row.get("Difficulty", "medium")).strip()
        diff_val = _difficulty_map.get(difficulty_label.lower(), 0.0)
        # Extract key phrase if available (could be same as correct answer or from a separate column)
        key_phrase = str(row.get("KeyPhrase", row.get("CorrectAnswer", ""))).strip()
        item = AdaptiveItem(
            item_id=f"{chapter}::{idx}",
            chapter_name=chapter,
            question=str(row["Question"]).strip(),
            correct_answer=str(row["CorrectAnswer"]).strip(),
            context=str(row.get("Context", "")).strip(),
            difficulty=diff_val,
            difficulty_label=difficulty_label,
            discrimination=1.0,
            key_phrase=key_phrase,
        )
        bank.setdefault(chapter, []).append(item)

    return bank


ITEM_BANK = _load_item_bank()
ADAPTIVE_CHAPTERS = sorted(ITEM_BANK.keys())


def _prob_correct(theta: float, a: float, b: float) -> float:
    return 1.0 / (1.0 + math.exp(-a * (theta - b)))


def _update_theta(theta: float, a: float, b: float, correct: bool, step: float = 0.35) -> float:
    delta = step * a
    return max(-3.0, min(3.0, theta + delta if correct else theta - delta))


def _pick_next_item(chapter: str, level: str, asked: set[str], theta: float) -> Optional[AdaptiveItem]:
    items = ITEM_BANK.get(chapter, [])
    # Prefer items in the current level first
    level_lower = level.lower()
    level_candidates = [i for i in items if i.item_id not in asked and i.difficulty_label.lower() == level_lower]
    if level_candidates:
        level_candidates.sort(key=lambda it: abs(it.difficulty - _difficulty_map.get(level_lower, 0.0)))
        return level_candidates[0]

    # Fallback: any remaining item closest to target difficulty
    target = _difficulty_map.get(level_lower, 0.0)
    remaining = [i for i in items if i.item_id not in asked]
    if not remaining:
        return None
    remaining.sort(key=lambda it: abs(it.difficulty - target))
    return remaining[0]


def _generate_distractors(item: AdaptiveItem) -> List[str]:
    """
    Generate 3 smart distractors for an adaptive quiz item using LLM.
    Returns cached distractors if already generated for this item.
    Uses thread lock to prevent concurrent LLM access issues.
    """
    # Check cache first
    if item.item_id in _distractor_cache:
        return _distractor_cache[item.item_id]
    
    fallback = [f"Option {i}" for i in range(2, 5)]
    
    if llm is None:
        print("⚠️ LLM not available, using fallback distractors")
        _distractor_cache[item.item_id] = fallback
        return fallback
    
    # Try to acquire lock with timeout
    lock_acquired = _llm_lock.acquire(timeout=30)
    if not lock_acquired:
        print("⚠️ LLM busy (lock timeout), using fallback distractors")
        _distractor_cache[item.item_id] = fallback
        return fallback
    
    try:
        # Use LLM to generate smart distractors
        chat_resp = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """You are a History Teacher creating MCQ distractors.
Generate exactly 3 plausible but INCORRECT answers for the given question.
The distractors should be:
- Related to the topic and time period
- Realistic enough to challenge students
- Clearly different from the correct answer

Format your response EXACTLY as:
DISTRACTOR_1: [first wrong answer]
DISTRACTOR_2: [second wrong answer]
DISTRACTOR_3: [third wrong answer]""",
                },
                {
                    "role": "user",
                    "content": f"Question: {item.question}\nCorrect Answer: {item.correct_answer}\nContext: {item.context[:500] if item.context else 'History question'}\n\nGenerate 3 distractors:",
                },
            ],
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
        resp = chat_resp["choices"][0]["message"]["content"]
        
        # Parse distractors
        d1_match = re.search(r"DISTRACTOR_1:?\s*(.*?)(?=\n*DISTRACTOR_2:|$)", resp, re.IGNORECASE | re.DOTALL)
        d2_match = re.search(r"DISTRACTOR_2:?\s*(.*?)(?=\n*DISTRACTOR_3:|$)", resp, re.IGNORECASE | re.DOTALL)
        d3_match = re.search(r"DISTRACTOR_3:?\s*(.*?)(?=$)", resp, re.IGNORECASE | re.DOTALL)
        
        d1 = _clean_option_text(d1_match.group(1).strip()) if d1_match else ""
        d2 = _clean_option_text(d2_match.group(1).strip()) if d2_match else ""
        d3 = _clean_option_text(d3_match.group(1).strip()) if d3_match else ""
        
        distractors = [d for d in [d1, d2, d3] if d and len(d) > 2]
        
        # Ensure we have exactly 3 distractors
        while len(distractors) < 3:
            distractors.append(f"Option {len(distractors) + 2}")
        
        _distractor_cache[item.item_id] = distractors[:3]
        return distractors[:3]
        
    except Exception as e:
        error_str = str(e).lower()
        print(f"Error generating distractors: {e}")
        
        # Check for critical errors that indicate LLM corruption
        if "access violation" in error_str or "segfault" in error_str:
            print("⚠️ LLM critical error detected")
        
        _distractor_cache[item.item_id] = fallback
        return fallback
        
    finally:
        _llm_lock.release()


def _item_with_options(item: AdaptiveItem) -> AdaptiveItem:
    """Add MCQ options to an adaptive item."""
    distractors = _generate_distractors(item)
    
    # Create options array with correct answer and distractors
    options = [item.correct_answer] + distractors
    random.shuffle(options)
    correct_index = options.index(item.correct_answer)
    
    # Return new item with options
    return AdaptiveItem(
        item_id=item.item_id,
        chapter_name=item.chapter_name,
        question=item.question,
        correct_answer=item.correct_answer,
        difficulty=item.difficulty,
        difficulty_label=item.difficulty_label,
        discrimination=item.discrimination,
        context=item.context,
        key_phrase=item.key_phrase,
        options=options,
        correct_index=correct_index,
    )


def _evaluate_answer(user_answer: str, item: AdaptiveItem, is_mcq: bool = True) -> dict:
    """
    Evaluate answer for adaptive quiz.
    
    For MCQ mode: Uses exact matching (binary correct/incorrect)
    For free-text mode: Uses SBERT semantic similarity
    """
    if not user_answer:
        return {
            "correct": False,
            "score": 0,
            "feedback": "Please select or type an answer!",
        }

    if sbert_model is None:
        raise HTTPException(500, "SBERT model not initialized")

    target = item.correct_answer if item.correct_answer and len(item.correct_answer) > 2 else "Refer to context"
    key_phrase = item.key_phrase or ""
    
    # Normalize for comparison
    user_clean = user_answer.strip().lower()
    correct_clean = target.strip().lower()

    # === MCQ EVALUATION (Exact Match) ===
    if is_mcq:
        # Direct exact match
        if user_clean == correct_clean:
            return {
                "correct": True,
                "score": 100,
                "feedback": "Correct! Well done!",
            }
        
        # Check if key phrase matches
        if key_phrase and len(key_phrase) > 2:
            if key_phrase.lower().strip() == user_clean:
                return {
                    "correct": True,
                    "score": 100,
                    "feedback": "Correct! Well done!",
                }
        
        # MCQ is binary - no partial credit
        return {
            "correct": False,
            "score": 0,
            "feedback": f"Incorrect. The correct answer was: {item.correct_answer}",
        }

    # === FREE-TEXT EVALUATION (SBERT) ===
    # Check key phrase first for exact match
    if key_phrase and len(key_phrase) > 2:
        if key_phrase.lower() in user_answer.lower():
            return {
                "correct": True,
                "score": 100,
                "feedback": "🎯 Excellent! You got it right!",
            }

    # Use SBERT semantic similarity
    try:
        emb1 = sbert_model.encode(user_answer, convert_to_tensor=True)
        emb2 = sbert_model.encode(target, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item()
        
        if score > 0.60:
            return {
                "correct": True,
                "score": int(score * 100),
                "feedback": "Your Answer is Correct",
            }
        elif score > 0.50:
            return {
                "correct": True,  # Partially correct still counts as correct for adaptive quiz
                "score": int(score * 100),
                "feedback": "You are Partially Correct, You have missed some details",
            }
        else:
            return {
                "correct": False,
                "score": int(score * 100),
                "feedback": "Your Answer is Incorrect",
            }
    except Exception as e:
        print(f"SBERT evaluation error: {e}")
        # Fallback to simple comparison if SBERT fails
        if user_clean == correct_clean:
            return {
                "correct": True,
                "score": 100,
                "feedback": "Correct (exact match)",
            }
        return {
            "correct": False,
            "score": 0,
            "feedback": "Incorrect",
        }


@router.get("/chapters")
def adaptive_chapters():
    return {"chapters": ADAPTIVE_CHAPTERS}


@router.post("/start")
def adaptive_start(req: AdaptiveStartRequest):
    if req.chapter_name not in ITEM_BANK:
        raise HTTPException(400, "Chapter not found in adaptive bank")

    session_doc = {
        "username": req.username,
        "chapter_name": req.chapter_name,
        "theta": 0.0,
        "asked": [],
        "active": True,
        "current_level": "easy",  # start with easy
        "level_correct": 0,
        "level_total": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    inserted = adaptive_sessions_col.insert_one(session_doc)
    session_id = str(inserted.inserted_id)

    next_item = _pick_next_item(req.chapter_name, session_doc["current_level"], set(), session_doc["theta"])
    if not next_item:
        raise HTTPException(500, "No items available for this chapter")

    # Add MCQ options to the item
    item_with_mcq = _item_with_options(next_item)

    return {
        "session_id": session_id,
        "theta": session_doc["theta"],
        "level": session_doc["current_level"],
        "item": item_with_mcq.dict(exclude={"correct_answer"}),
    }


@router.post("/answer")
def adaptive_answer(req: AdaptiveAnswerRequest):
    try:
        session_obj_id = ObjectId(req.session_id)
    except Exception:
        raise HTTPException(400, "Invalid session id")

    session = adaptive_sessions_col.find_one({"_id": session_obj_id, "username": req.username})
    if not session or not session.get("active", False):
        raise HTTPException(404, "Session not found or inactive")

    chapter = session.get("chapter_name")
    asked_set = set(session.get("asked", []))

    # locate item
    item = next((i for i in ITEM_BANK.get(chapter, []) if i.item_id == req.item_id), None)
    if not item:
        raise HTTPException(400, "Item not found for chapter")

    evaluation = _evaluate_answer(req.user_answer, item, is_mcq=True)
    is_correct = evaluation["correct"]
    new_theta = _update_theta(session.get("theta", 0.0), item.discrimination, item.difficulty, is_correct)

    asked_set.add(item.item_id)
    current_level = session.get("current_level", "easy")
    level_total = session.get("level_total", 0) + 1
    level_correct = session.get("level_correct", 0) + (1 if is_correct else 0)

    # Upgrade rule: after 3 attempts in a level, if at least 2 correct, move up
    level_index = LEVEL_ORDER.index(current_level) if current_level in LEVEL_ORDER else 0
    if level_total >= 3 and level_correct >= 2 and level_index < len(LEVEL_ORDER) - 1:
        current_level = LEVEL_ORDER[level_index + 1]
        level_total = 0
        level_correct = 0

    adaptive_sessions_col.update_one(
        {"_id": session_obj_id},
        {
            "$set": {
                "theta": new_theta,
                "updated_at": datetime.utcnow(),
                "current_level": current_level,
                "level_total": level_total,
                "level_correct": level_correct,
            },
            "$addToSet": {"asked": item.item_id},
        },
    )

    next_item = _pick_next_item(chapter, current_level, asked_set, new_theta)
    
    # Add MCQ options to next item if available
    next_item_with_mcq = _item_with_options(next_item) if next_item else None

    return {
        "correct": is_correct,
        "score": evaluation["score"],
        "feedback": evaluation["feedback"],
        "theta": new_theta,
        "level": current_level,
        "correct_answer": item.correct_answer,
        "probability": _prob_correct(new_theta, item.discrimination, item.difficulty),
        "next_item": next_item_with_mcq.dict(exclude={"correct_answer"}) if next_item_with_mcq else None,
        "done": next_item is None,
    }


@router.post("/finish")
def adaptive_finish(req: AdaptiveFinishRequest):
    try:
        session_obj_id = ObjectId(req.session_id)
    except Exception:
        raise HTTPException(400, "Invalid session id")

    # Calculate final summary before finishing
    session = adaptive_sessions_col.find_one({"_id": session_obj_id, "username": req.username})
    if not session:
        raise HTTPException(404, "Session not found")
    
    # Store final summary for profile display
    final_theta = session.get('theta', 0.0)
    total_questions = len(session.get('asked', []))
    final_level = session.get('current_level', 'easy')
    
    adaptive_sessions_col.update_one(
        {"_id": session_obj_id, "username": req.username},
        {
            "$set": {
                "active": False, 
                "updated_at": datetime.utcnow(),
                "final_summary": {
                    "final_theta": final_theta,
                    "final_level": final_level,
                    "total_questions": total_questions,
                    "completed_at": datetime.utcnow()
                }
            }
        },
    )
    return {"status": "ok"}
