"""
Timed quiz: 20 questions from QuizDataset (mixed chapters), MCQ with 3 distractors, 30 minutes (enforced client-side).
Evaluation runs once when the learner submits all answers or time expires.
"""

import copy
import random
from datetime import datetime
from typing import Dict, List

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient
from dotenv import load_dotenv
import os

from adaptive_routes import ITEM_BANK, AdaptiveItem, _item_with_options

load_dotenv()
client = MongoClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DATABASE_NAME", "vis_history")]
timed_quiz_sessions_col = db["timed_quiz_sessions"]

router = APIRouter(prefix="/timed-quiz", tags=["timed-quiz"])

TIMED_QUESTION_COUNT = 20
TIMED_DURATION_SEC = 30 * 60


def _pick_timed_items(count: int = TIMED_QUESTION_COUNT) -> List[AdaptiveItem]:
    chapters = list(ITEM_BANK.keys())
    if not chapters:
        raise HTTPException(500, "Question dataset is empty")

    total_available = sum(len(ITEM_BANK[ch]) for ch in chapters)
    if total_available < count:
        raise HTTPException(
            500,
            f"Not enough questions in dataset for timed quiz (need {count}, have {total_available})",
        )

    random.shuffle(chapters)
    used: set[str] = set()
    picked: List[AdaptiveItem] = []

    while len(picked) < count:
        added = False
        for ch in chapters:
            if len(picked) >= count:
                break
            candidates = [i for i in ITEM_BANK[ch] if i.item_id not in used]
            if not candidates:
                continue
            choice = random.choice(candidates)
            picked.append(choice)
            used.add(choice.item_id)
            added = True
        if not added:
            break

    while len(picked) < count:
        pool = [i for ch in chapters for i in ITEM_BANK[ch] if i.item_id not in used]
        if not pool:
            raise HTTPException(500, "Could not select 20 unique questions for timed quiz")
        choice = random.choice(pool)
        picked.append(choice)
        used.add(choice.item_id)

    return picked


def _serialize_item_server(item: AdaptiveItem) -> dict:
    return {
        "item_id": item.item_id,
        "chapter_name": item.chapter_name,
        "question": item.question,
        "options": item.options or [],
        "correct_index": item.correct_index if item.correct_index is not None else 0,
        "correct_answer": item.correct_answer,
    }


def _public_question(item: AdaptiveItem) -> dict:
    return {
        "item_id": item.item_id,
        "chapter_name": item.chapter_name,
        "question": item.question,
        "options": item.options or [],
    }


class TimedQuizStartRequest(BaseModel):
    username: str


class TimedQuizAnswerItem(BaseModel):
    item_id: str
    selected_index: int = Field(default=-1, ge=-1, le=3)


class TimedQuizEvaluateRequest(BaseModel):
    username: str
    session_id: str
    answers: List[TimedQuizAnswerItem]


class TimedQuizRetakeRequest(BaseModel):
    username: str
    template_session_id: str


@router.post("/start")
def timed_quiz_start(req: TimedQuizStartRequest):
    raw_items = _pick_timed_items(TIMED_QUESTION_COUNT)
    server_items: List[dict] = []
    client_questions: List[dict] = []

    for raw in raw_items:
        with_options = _item_with_options(raw)
        server_items.append(_serialize_item_server(with_options))
        client_questions.append(_public_question(with_options))

    doc = {
        "username": req.username,
        "status": "in_progress",
        "created_at": datetime.utcnow(),
        "items": server_items,
    }
    inserted = timed_quiz_sessions_col.insert_one(doc)
    session_id = str(inserted.inserted_id)

    return {
        "session_id": session_id,
        "duration_seconds": TIMED_DURATION_SEC,
        "total_questions": TIMED_QUESTION_COUNT,
        "questions": client_questions,
    }


@router.post("/retake")
def timed_quiz_retake(req: TimedQuizRetakeRequest):
    """
    Clone a completed session's frozen items into a new in-progress quiz (same wording and MCQ shuffle).
    """
    try:
        tid = ObjectId(req.template_session_id)
    except Exception:
        raise HTTPException(400, "Invalid template session id")

    src = timed_quiz_sessions_col.find_one(
        {"_id": tid, "username": req.username, "status": "completed"}
    )
    if not src:
        raise HTTPException(404, "Completed timed quiz not found for retake")

    items: List[dict] = copy.deepcopy(src.get("items") or [])
    if not items:
        raise HTTPException(500, "Saved timed quiz data is incomplete")

    client_questions: List[dict] = [
        {
            "item_id": it["item_id"],
            "chapter_name": it.get("chapter_name", ""),
            "question": it.get("question", ""),
            "options": list(it.get("options") or []),
        }
        for it in items
    ]

    doc = {
        "username": req.username,
        "status": "in_progress",
        "created_at": datetime.utcnow(),
        "items": items,
        "retake_from_session_id": str(tid),
    }
    inserted = timed_quiz_sessions_col.insert_one(doc)
    session_id = str(inserted.inserted_id)

    return {
        "session_id": session_id,
        "duration_seconds": TIMED_DURATION_SEC,
        "total_questions": len(items),
        "questions": client_questions,
    }


@router.post("/evaluate")
def timed_quiz_evaluate(req: TimedQuizEvaluateRequest):
    try:
        oid = ObjectId(req.session_id)
    except Exception:
        raise HTTPException(400, "Invalid session id")

    session = timed_quiz_sessions_col.find_one({"_id": oid, "username": req.username})
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("status") != "in_progress":
        raise HTTPException(400, "This quiz was already submitted")

    items: List[dict] = session.get("items") or []
    answers_map: Dict[str, int] = {a.item_id: a.selected_index for a in req.answers}

    results: List[dict] = []
    correct_count = 0

    for it in items:
        iid = it["item_id"]
        sel = answers_map.get(iid, -1)
        correct_idx = int(it.get("correct_index", 0))
        options: List[str] = it.get("options") or []
        is_correct = sel == correct_idx and 0 <= sel < len(options)
        if is_correct:
            correct_count += 1
        sel_text = options[sel] if 0 <= sel < len(options) else ""
        corr_text = it.get("correct_answer", "")
        results.append(
            {
                "item_id": iid,
                "chapter_name": it.get("chapter_name", ""),
                "question": it.get("question", ""),
                "correct": is_correct,
                "selected_index": sel if sel >= 0 else None,
                "correct_index": correct_idx,
                "selected_text": sel_text,
                "correct_answer": corr_text,
            }
        )

    average_score = round((correct_count / len(items)) * 100, 1) if items else 0.0

    timed_quiz_sessions_col.update_one(
        {"_id": oid},
        {
            "$set": {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "correct_count": correct_count,
                "total_questions": len(items),
                "average_score": average_score,
                "detail": results,
            }
        },
    )

    return {
        "correct_count": correct_count,
        "total_questions": len(items),
        "average_score": average_score,
        "results": results,
    }


@router.get("/health")
def timed_quiz_health():
    chapters = len(ITEM_BANK)
    items = sum(len(v) for v in ITEM_BANK.values())
    return {"ok": True, "chapters": chapters, "items": items}

