import pyttsx3
from typing import Optional
import os
from datetime import datetime

class AudioService:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.audio_dir = 'audio_files'
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)

    def generate_audio(self, text: str, filename: str) -> Optional[str]:
        """
        Generate audio from text using TTS
        """
        try:
            filepath = os.path.join(self.audio_dir, f"{filename}.mp3")
            
            # Only generate if file doesn't exist
            if not os.path.exists(filepath):
                self.engine.save_to_file(text, filepath)
                self.engine.runAndWait()
            
            return filepath
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None

    def generate_lesson_audio(self, lesson_title: str, subsection_title: str, content: str) -> Optional[str]:
        """
        Generate audio for a lesson subsection
        """
        filename = f"{lesson_title.replace(' ', '_')}_{subsection_title.replace(' ', '_')}_{int(datetime.now().timestamp())}"
        return self.generate_audio(content, filename)
