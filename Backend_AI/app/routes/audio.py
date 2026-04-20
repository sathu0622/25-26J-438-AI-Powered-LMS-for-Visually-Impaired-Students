from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
from app.services.tts_service import tts_service
from local_main import generate_audio_for_selection
from datetime import datetime, timedelta


router = APIRouter(prefix="/api/audio", tags=["audio"])

# Cache for recent audio generation requests to prevent duplicates
# Format: {(grade, chapter_idx, topic_idx): (file_path, timestamp)}
_audio_cache = {}
CACHE_DURATION = timedelta(seconds=30)  # Cache for 30 seconds


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
async def get_chapter_audio(
    grade: int,
    chapter_idx: int,
    topic_idx: int,
    sound_mode: str = "With Effects",
    emotion_intensity: float = 1.0,
    use_model: bool = True,
):
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
        cache_key = (grade, chapter_idx, topic_idx)
        now = datetime.now()
        
        # Check if we have a cached version
        if cache_key in _audio_cache:
            cached_path, cached_time = _audio_cache[cache_key]
            if now - cached_time < CACHE_DURATION and os.path.exists(cached_path):
                # Return cached audio without regenerating
                return FileResponse(
                    path=cached_path,
                    media_type="audio/wav",
                    filename=f"lesson_{grade}_{chapter_idx}_{topic_idx}.wav"
                )
        
        # Generate new audio
        print(f"📚 Requested audio for: grade={grade}, chapter={chapter_idx}, topic={topic_idx}")

        audio_path = generate_audio_for_selection(
            grade=grade,
            chapter_idx=chapter_idx,
            topic_idx=topic_idx,
            sound_mode=sound_mode,
            emotion_intensity=emotion_intensity,
            use_model=use_model,
        )
        
        # Cache the result
        _audio_cache[cache_key] = (audio_path, now)
        
        print(f"✅ Audio generated successfully: {audio_path}")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"lesson_{grade}_{chapter_idx}_{topic_idx}.wav"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating chapter audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")
