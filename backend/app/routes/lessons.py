from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.services.lesson_data import get_lessons_by_grade, get_lesson_by_id, get_subsection_by_id
from app.services.ai_history_service import ai_history_service
from app.services.model_service import ai_model_service
from app.services.audio_generator import audio_generator
from typing import List, Optional
from pydantic import BaseModel
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic model for student profile
class StudentProfile(BaseModel):
    learning_style: Optional[str] = "auditory"  # auditory, visual, kinesthetic
    pace: Optional[str] = "normal"  # slow, normal, fast
    interests: Optional[list] = []

# Pydantic model for audio generation request
class AudioGenerationRequest(BaseModel):
    emotion_intensity: Optional[float] = 1.0
    include_effects: Optional[bool] = True
    effects_only: Optional[bool] = False

@router.get("/grades")
async def get_grades():
    """Get available grades"""
    return {"grades": [10, 11]}

@router.get("/chapters/{grade}")
async def get_chapters_by_grade(grade: int):
    """
    Get AI-generated chapters for a specific grade using the trained model
    This endpoint uses the AI_History_Teacher_System.pth model
    """
    chapters = ai_history_service.get_chapters_by_grade(grade)
    if not chapters:
        return {"error": "Grade not found", "grade": grade}, 404
    
    return {
        "grade": grade,
        "model_used": "AI_History_Teacher_System.pth" if ai_history_service.model_loaded else "AI-Generated Curriculum",
        "chapters": chapters,
        "total_chapters": len(chapters)
    }

@router.get("/chapter/{grade}/{chapter_id}")
async def get_chapter(grade: int, chapter_id: str):
    """Get a specific chapter with all its topics"""
    chapter = ai_history_service.get_chapter_by_id(grade, chapter_id)
    if not chapter:
        return {"error": "Chapter not found"}, 404
    
    return {
        "grade": grade,
        "chapter": chapter,
        "topics_count": len(chapter.get("topics", []))
    }

@router.get("/chapter/{grade}/{chapter_id}/topics")
async def get_chapter_topics(grade: int, chapter_id: str):
    """Get all topics for a chapter"""
    topics = ai_history_service.get_chapter_topics(grade, chapter_id)
    if not topics:
        return {"error": "Chapter or topics not found"}, 404
    
    return {
        "grade": grade,
        "chapter_id": chapter_id,
        "topics": topics,
        "total_topics": len(topics)
    }

@router.get("/topic/{grade}/{chapter_id}/{topic_id}")
async def get_topic(grade: int, chapter_id: str, topic_id: str):
    """Get a specific topic with its content and lesson details"""
    topic = ai_history_service.get_topic_by_id(grade, chapter_id, topic_id)
    if not topic:
        return {"error": "Topic not found"}, 404
    
    return {
        "grade": grade,
        "chapter_id": chapter_id,
        "topic": topic
    }

@router.post("/topic/{grade}/{chapter_id}/{topic_id}/generate-audio")
async def generate_topic_audio(
    grade: int, 
    chapter_id: str, 
    topic_id: str,
    request: AudioGenerationRequest
):
    """
    Generate audio for a specific topic with emotional TTS and sound effects
    Based on the Jupyter notebook implementation
    """
    # Get topic data
    topic = ai_history_service.get_topic_by_id(grade, chapter_id, topic_id)
    if not topic:
        return {"error": "Topic not found"}, 404
    
    # Get chapter for intro
    chapter = ai_history_service.get_chapter_by_id(grade, chapter_id)
    if not chapter:
        return {"error": "Chapter not found"}, 404
    
    chapter_title = chapter.get('title', 'Chapter')
    
    try:
        # Generate audio
        audio_path = audio_generator.generate_topic_audio(
            topic_data=topic,
            chapter_title=chapter_title,
            emotion_intensity=request.emotion_intensity,
            include_effects=request.include_effects,
            effects_only=request.effects_only
        )
        
        # Return file info
        return {
            "success": True,
            "audio_file": os.path.basename(audio_path),
            "audio_url": f"/api/audio/{os.path.basename(audio_path)}",
            "emotion": topic.get('emotion', 'neutral'),
            "duration_estimate": "varies"
        }
    
    except Exception as e:
        import traceback
        logger.error(f"❌ Audio generation error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, 500

@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files"""
    audio_output_folder = audio_generator.audio_output_folder
    file_path = audio_output_folder / filename
    
    if not file_path.exists():
        return {"error": "Audio file not found"}, 404
    
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=filename
    )

@router.get("/ai/chapters/{grade}")
async def get_ai_chapters(grade: int):
    """Get AI-recommended chapters for a specific grade (Legacy endpoint)"""
    chapters_data = ai_history_service.get_chapters_by_grade(grade)
    if not chapters_data:
        return {
            "grade": grade,
            "error": "Grade not found",
            "chapters": []
        }
    
    return {
        "grade": grade,
        "difficulty": "intermediate" if grade == 10 else "advanced",
        "model_source": "AI_History_Teacher_System.pth",
        "chapters": chapters_data
    }

@router.post("/ai/personalized-lessons/{grade}")
async def get_personalized_lessons(grade: int, profile: StudentProfile):
    """Get personalized lesson recommendations based on student profile"""
    profile_dict = profile.dict() if hasattr(profile, 'dict') else profile
    recommendations = ai_model_service.get_personalized_lessons(grade, profile_dict)
    return recommendations

@router.get("/ai/chapter-difficulty/{grade}/{chapter_id}")
async def check_chapter_difficulty(grade: int, chapter_id: str):
    """Check if a chapter is appropriate for the student's grade level"""
    evaluation = ai_model_service.evaluate_chapter_difficulty(grade, chapter_id)
    return evaluation

@router.get("/lessons/{grade}")
async def get_lessons(grade: int):
    """Get all lessons for a specific grade (Legacy endpoint)"""
    lessons = get_lessons_by_grade(grade)
    if not lessons:
        return {"error": "Grade not found"}, 404
    return {
        "grade": grade,
        "lessons": lessons
    }

@router.get("/lesson/{lesson_id}")
async def get_lesson(lesson_id: str):
    """Get a specific lesson"""
    lesson = get_lesson_by_id(lesson_id)
    if not lesson:
        return {"error": "Lesson not found"}, 404
    return lesson

@router.get("/lesson/{lesson_id}/subsections")
async def get_subsections(lesson_id: str):
    """Get all subsections for a lesson"""
    lesson = get_lesson_by_id(lesson_id)
    if not lesson:
        return {"error": "Lesson not found"}, 404
    return {
        "lesson_id": lesson_id,
        "lesson_title": lesson["title"],
        "subsections": lesson["subsections"]
    }

@router.get("/subsection/{lesson_id}/{subsection_id}")
async def get_subsection(lesson_id: str, subsection_id: str):
    """Get a specific subsection with its content"""
    subsection = get_subsection_by_id(lesson_id, subsection_id)
    if not subsection:
        return {"error": "Subsection not found"}, 404
    return {
        "lesson_id": lesson_id,
        "subsection": subsection
    }

@router.post("/generate-audio/{lesson_id}/{subsection_id}")
async def generate_audio(lesson_id: str, subsection_id: str):
    """Generate audio for a subsection"""
    lesson = get_lesson_by_id(lesson_id)
    subsection = get_subsection_by_id(lesson_id, subsection_id)
    
    if not lesson or not subsection:
        return {"error": "Lesson or subsection not found"}, 404
    
    from app.services.audio_service import AudioService
    audio_service = AudioService()
    
    content = subsection.get("content", "")
    filename = f"{lesson['title'].replace(' ', '_')}_{subsection['title'].replace(' ', '_')}"
    
    filepath = audio_service.generate_audio(content, filename)
    
    if filepath:
        return {
            "success": True,
            "audio_url": f"/audio/{filename}.mp3",
            "duration": subsection.get("duration", 5)
        }
    else:
        return {"error": "Failed to generate audio"}, 500

