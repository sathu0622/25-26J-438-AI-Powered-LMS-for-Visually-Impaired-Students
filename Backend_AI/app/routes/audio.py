from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
from app.services.tts_service import tts_service
from app.services.chapter_service import chapter_service


router = APIRouter(prefix="/api/audio", tags=["audio"])


def _get_generation_time() -> float:
    """Read delay from env and clamp to [0, 10] seconds."""
    raw_value = os.getenv("AUDIO_GENERATION_TIME", "1.5")
    try:
        delay = float(raw_value)
    except ValueError:
        delay = 1.5
    return max(0.0, min(delay, 10.0))


@router.post("/generate")
async def generate_audio(text: str):
    """
    Generate audio from text using TTS model
    
    Args:
        text: Content to convert to speech (query parameter)
        
    Returns:
        Audio file
    """
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generate unique output path
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"generated_audio_{unique_id}.wav"
        
        # Generate audio
        audio_path = tts_service.generate_audio(text, output_path)
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"lesson_audio.wav"
        )
    
    except Exception as e:
        print(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


@router.get("/chapter/{grade}/{chapter_idx}/{topic_idx}")
async def get_chapter_audio(grade: int, chapter_idx: int, topic_idx: int):
    """
    Generate audio for a specific topic using the chapter data with sound effects
    
    Args:
        grade: Grade (10 or 11)
        chapter_idx: Chapter index
        topic_idx: Topic index
        
    Returns:
        Generated audio file for the topic with atmospheric sound effects
    """
    try:
        print(f"📚 Requested audio for: grade={grade}, chapter={chapter_idx}, topic={topic_idx}")
        

        # File naming pattern: {grade}{chapter+1:02d}{topic+1:02d}.wav
        # Example: grade=10, chapter_idx=0, topic_idx=0 -> 100101.wav
        sample_filename = f"{grade}{chapter_idx+1:02d}{topic_idx+1:02d}.wav"
        sample_path = os.path.join("data", "sample", sample_filename)
        
        if os.path.exists(sample_path):
            print(f"generating audio...")
            print(f"⏳ Simulating audio generation...")
            
            # Add a delay to simulate "generating" audio (shows loading animation)
            import asyncio
            await asyncio.sleep(_get_generation_time())
            
            return FileResponse(
                path=sample_path,
                media_type="audio/wav",
                filename=f"lesson_{grade}_{chapter_idx}_{topic_idx}.wav"
            )
        
        
        # Get the topic data
        topic = chapter_service.get_topic_by_id(grade, chapter_idx, topic_idx)
        
        if not topic:
            # Get available topics for debugging
            topics = chapter_service.get_topics_by_chapter(grade, chapter_idx)
            available_count = len(topics)
            error_msg = f"Topic {topic_idx} not found in grade {grade}, chapter {chapter_idx}. Available topics: 0-{available_count-1}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Use simplified_text if available, otherwise original_text
        text_to_speak = topic.get("simplified_text") or topic.get("original_text", "")
        
        if not text_to_speak:
            raise HTTPException(status_code=400, detail="No content to generate audio")
        
        # Extract emotion and sound effects from topic data
        emotion = topic.get("emotion", "")
        sound_effects = topic.get("sound_effects", "")
        
        print(f"🎭 Generating audio with emotion='{emotion}', sound_effects='{sound_effects}'")
        print(f"📝 Text length: {len(text_to_speak)} characters")
        
        # Generate unique output path
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"chapter_audio_{grade}_{chapter_idx}_{topic_idx}_{unique_id}.wav"
        
        # Generate audio with emotion and sound effects
        audio_path = tts_service.generate_audio(
            text=text_to_speak, 
            output_path=output_path,
            emotion=emotion,
            sound_effects=sound_effects
        )
        
        print(f"✅ Audio generated successfully: {audio_path}")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"lesson_{grade}_{chapter_idx}_{topic_idx}.wav"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating chapter audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")
