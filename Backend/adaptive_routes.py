import os
import math
import random
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


def _evaluate_answer(user_answer: str, item: AdaptiveItem) -> dict:
    """Evaluate answer using SBERT semantic similarity like quiz_routes.py"""
    if not user_answer:
        return {
            "correct": False,
            "score": 0,
            "feedback": "Please type an answer!",
        }

    if sbert_model is None:
        raise HTTPException(500, "SBERT model not initialized")

    target = item.correct_answer if item.correct_answer and len(item.correct_answer) > 2 else "Refer to context"
    key_phrase = item.key_phrase or ""

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
        ua = user_answer.strip().lower()
        ca = target.strip().lower()
        if ua == ca:
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

    return {
        "session_id": session_id,
        "theta": session_doc["theta"],
        "level": session_doc["current_level"],
        "item": next_item.dict(exclude={"correct_answer"}),
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

    evaluation = _evaluate_answer(req.user_answer, item)
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

    return {
        "correct": is_correct,
        "score": evaluation["score"],
        "feedback": evaluation["feedback"],
        "theta": new_theta,
        "level": current_level,
        "correct_answer": item.correct_answer,
        "probability": _prob_correct(new_theta, item.discrimination, item.difficulty),
        "next_item": next_item.dict(exclude={"correct_answer"}) if next_item else None,
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
