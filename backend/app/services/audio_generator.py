"""
Audio Generation Service - Simplified Version
Generates TTS audio with emotional tones
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path
from gtts import gTTS
from pydub import AudioSegment
import logging

# Python 3.13 compatibility
try:
    import audioop_lts as audioop
    sys.modules['audioop'] = audioop
except ImportError:
    pass

logger = logging.getLogger(__name__)


class EmotionalTTS:
    """Generate TTS with emotional modulation"""
    
    def __init__(self):
        self.emotion_map = {
            'neutral': {'speed': 1.0, 'pitch': 1.0, 'volume': 1.0},
            'inspirational': {'speed': 0.9, 'pitch': 1.1, 'volume': 1.2},
            'awe': {'speed': 0.85, 'pitch': 1.15, 'volume': 1.1},
            'vibrancy': {'speed': 1.1, 'pitch': 1.15, 'volume': 1.3},
            'harmony': {'speed': 0.95, 'pitch': 1.05, 'volume': 1.1},
            'wonder': {'speed': 0.9, 'pitch': 1.2, 'volume': 1.2},
            'reverence': {'speed': 0.8, 'pitch': 0.9, 'volume': 0.95},
            'justice': {'speed': 1.0, 'pitch': 1.05, 'volume': 1.15},
            'prosperity': {'speed': 1.05, 'pitch': 1.1, 'volume': 1.2},
            'warmth': {'speed': 0.95, 'pitch': 1.0, 'volume': 1.15},
            'hope': {'speed': 0.9, 'pitch': 1.15, 'volume': 1.25},
            'resilience': {'speed': 1.0, 'pitch': 1.0, 'volume': 1.1},
            'somber': {'speed': 0.85, 'pitch': 0.95, 'volume': 0.9},
            'respect': {'speed': 0.9, 'pitch': 0.95, 'volume': 1.05},
        }
    
    def generate_speech(self, text, emotion='neutral', intensity=1.0):
        """Generate TTS speech with emotion"""
        try:
            if not text or len(text.strip()) == 0:
                return None
            
            # Get emotion parameters
            emotion_params = self.emotion_map.get(emotion, self.emotion_map['neutral'])
            
            # Apply intensity multiplier
            speed = emotion_params['speed'] * intensity
            
            # Generate MP3 using gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            temp_file.close()
            
            # Load as AudioSegment
            audio = AudioSegment.from_file(temp_file.name, format='mp3')
            
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
            
            return audio
            
        except Exception as e:
            logger.error(f"[ERROR] Error in generate_speech: {str(e)}")
            logger.error(traceback.format_exc())
            return None


class AudioGenerator:
    """Generate audio for topics"""
    
    def __init__(self):
        self.tts = EmotionalTTS()
        self.output_dir = Path(__file__).parent.parent.parent / "audio_output"
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_topic_audio(self, topic_data, chapter_title, emotion_intensity=1.0, include_effects=False, effects_only=False):
        """Generate audio for a topic"""
        try:
            # Extract text from topic_data
            topic_name = topic_data.get('name', 'Untitled Topic')
            topic_content = topic_data.get('content', '')
            
            logger.info("[AUDIO] Generating introductory speech part 1...")
            
            # Intro part 1
            intro1_text = f"Welcome to the lesson on {topic_name}"
            audio1 = self.tts.generate_speech(intro1_text, emotion='inspirational', intensity=emotion_intensity)
            
            logger.info("[AUDIO] Generating introductory speech part 2...")
            
            # Intro part 2
            intro2_text = f"This is Chapter: {chapter_title}"
            audio2 = self.tts.generate_speech(intro2_text, emotion='inspirational', intensity=emotion_intensity)
            
            logger.info("[AUDIO] Generating main lesson speech...")
            
            # Main content (in chunks to avoid too long strings)
            content_chunks = [topic_content[i:i+500] for i in range(0, len(topic_content), 500)]
            audio_parts = []
            
            if audio1:
                audio_parts.append(audio1)
            if audio2:
                audio_parts.append(audio2)
            
            for chunk in content_chunks:
                if chunk.strip():
                    audio_chunk = self.tts.generate_speech(chunk, emotion='neutral', intensity=emotion_intensity)
                    if audio_chunk:
                        audio_parts.append(audio_chunk)
            
            logger.info("[AUDIO] Generating closing speech...")
            
            # Closing
            closing_text = "Thank you for listening to this lesson. Continue exploring history!"
            audio_closing = self.tts.generate_speech(closing_text, emotion='warmth', intensity=emotion_intensity)
            if audio_closing:
                audio_parts.append(audio_closing)
            
            # Combine all parts
            if not audio_parts:
                logger.error("[ERROR] No audio parts were generated")
                return {'success': False, 'error': 'No audio generated'}
            
            logger.info("[COMBINE] Combining audio parts...")
            
            combined_audio = audio_parts[0]
            for audio_part in audio_parts[1:]:
                combined_audio += audio_part
            
            # Export
            output_filename = f"{topic_name.replace(' ', '_')}_{int(emotion_intensity*10)}.wav"
            output_path = self.output_dir / output_filename
            
            logger.info(f"[EXPORT] Exporting to: {output_path}")
            
            combined_audio.export(str(output_path), format='wav')
            
            logger.info(f"[OK] Audio saved to: {output_path}")
            
            return {
                'success': True,
                'audio_file': output_filename,
                'audio_url': f"/audio/{output_filename}",
                'emotion': 'inspirational',
                'duration_estimate': f"{len(combined_audio) / 1000:.1f}s"
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error generating audio: {str(e)}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}


# Create global instance
audio_generator = AudioGenerator()
