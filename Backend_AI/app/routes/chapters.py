from fastapi import APIRouter, HTTPException
from app.services.chapter_service import ChapterService

router = APIRouter(prefix="/api", tags=["chapters"])
chapter_service = ChapterService()


@router.get("/grades")
async def get_grades():
    """Get available grades"""
    return {
        "grades": [10, 11],
        "message": "Select Grade 10 or Grade 11"
    }


@router.get("/chapters/{grade}")
async def get_chapters(grade: int):
    """Get all chapters for a specific grade"""
    if grade not in [10, 11]:
        raise HTTPException(status_code=400, detail="Grade must be 10 or 11")
    
    chapters = chapter_service.get_chapters_by_grade(grade)
    
    if not chapters:
        raise HTTPException(status_code=404, detail=f"No chapters found for Grade {grade}")
    
    return {
        "grade": grade,
        "chapters": chapters,
        "total_chapters": len(chapters)
    }


@router.get("/chapters/{grade}/{chapter_idx}")
async def get_chapter_details(grade: int, chapter_idx: int):
    """Get chapter name and topic count"""
    if grade not in [10, 11]:
        raise HTTPException(status_code=400, detail="Grade must be 10 or 11")
    
    chapters = chapter_service.get_chapters_by_grade(grade)
    
    if chapter_idx >= len(chapters):
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    return chapters[chapter_idx]


@router.get("/chapters/{grade}/{chapter_idx}/topics")
async def get_chapter_topics(grade: int, chapter_idx: int):
    """Get all topics in a chapter"""
    if grade not in [10, 11]:
        raise HTTPException(status_code=400, detail="Grade must be 10 or 11")
    
    chapters = chapter_service.get_chapters_by_grade(grade)
    
    if chapter_idx >= len(chapters):
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    chapter_name = chapters[chapter_idx]['chapter_name']
    topics = chapter_service.get_topics_by_chapter(grade, chapter_idx)
    
    return {
        "grade": grade,
        "chapter_id": chapter_idx,
        "chapter_name": chapter_name,
        "topics": topics,
        "total_topics": len(topics)
    }


@router.get("/chapters/{grade}/{chapter_idx}/topics/{topic_idx}")
async def get_topic_detail(grade: int, chapter_idx: int, topic_idx: int):
    """Get a specific topic content"""
    if grade not in [10, 11]:
        raise HTTPException(status_code=400, detail="Grade must be 10 or 11")
    
    topic = chapter_service.get_topic_by_id(grade, chapter_idx, topic_idx)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return topic
