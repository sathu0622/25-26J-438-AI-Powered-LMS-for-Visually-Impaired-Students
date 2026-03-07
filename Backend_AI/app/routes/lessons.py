from fastapi import APIRouter, HTTPException
from typing import List

from app.models.models import Lesson, Topic
from app.services.lesson_data import (
    get_grades,
    get_lessons_by_grade,
    get_lesson_by_id,
    get_topics_by_lesson_id
)

router = APIRouter(prefix="/api", tags=["lessons"])


@router.get("/grades", response_model=List[int])
def list_grades():
    return get_grades()


@router.get("/lessons/{grade}", response_model=List[Lesson])
def list_lessons_by_grade(grade: int):
    lessons = get_lessons_by_grade(grade)
    if not lessons:
        raise HTTPException(status_code=404, detail="Grade not found")
    return lessons


@router.get("/lesson/{lesson_id}", response_model=Lesson)
def get_lesson(lesson_id: int):
    lesson = get_lesson_by_id(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return lesson


@router.get("/lesson/{lesson_id}/topics", response_model=List[Topic])
def get_topics(lesson_id: int):
    topics = get_topics_by_lesson_id(lesson_id)
    if not topics:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return topics
