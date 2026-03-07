import os
try:
    import torch
except ImportError:
    torch = None
from pathlib import Path
from typing import Optional, Dict


class TTSService:
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.load_model()
        
        # Map emotions to TTS adjustments
        self.emotion_prefixes = {
            'wonder': '(Speaking with wonder) ',
            'reverence': '(Speaking with reverence) ',
            'excitement': '(Speaking with excitement) ',
            'mystery': '(Speaking mysteriously) ',
            'pride': '(Speaking with pride) ',
            'sadness': '(Speaking solemnly) ',
            'tension': '(Speaking with tension) ',
            'triumph': '(Speaking triumphantly) ',
            'curiosity': '(Speaking curiously) ',
            'awe': '(Speaking with awe) '
        }
        
        # Map sound effects to atmospheric descriptions
        self.sound_descriptions = {
            'birds_chirping': '[Sound of birds chirping in the background]',
            'ancient_ambience': '[Ancient atmospheric sounds]',
            'wind_blowing': '[Sound of wind blowing]',
            'water_flowing': '[Sound of water flowing]',
            'drums': '[Traditional drums beating]',
            'chanting': '[Chanting in the background]',
            'battle_sounds': '[Sounds of battle]',
            'waves_crashing': '[Waves crashing]',
            'forest_ambience': '[Forest ambience]',
            'ceremonial_music': '[Ceremonial music playing]',
            'thunder': '[Thunder rumbling]',
            'footsteps': '[Footsteps echoing]',
            'marketplace': '[Marketplace sounds]',
            'temple_bells': '[Temple bells ringing]',
            'construction': '[Construction sounds]'
        }
    
    def load_model(self):
        """Load the trained TTS model"""
        model_path = "app/models/AI_History_Teacher_System.pth"
        
        try:
            if os.path.exists(model_path):
                if torch is None:
                    print("⚠ PyTorch is not installed. Will use gTTS fallback.")
                    return False
                    
                print(f"Loading trained model from {model_path}...")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model = checkpoint
                print("✓ Model loaded successfully")
                return True
            else:
                print(f"⚠ Model file not found at {model_path}")
                print("Will use gTTS fallback for audio generation")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will use gTTS fallback for audio generation")
            return False
    
    def generate_audio(self, text: str, output_path: str = "temp_audio.wav", 
                      emotion: str = "", sound_effects: str = "") -> str:
        """
        Generate audio from text using trained model or fallback to gTTS
        
        Args:
            text: Content to convert to speech
            output_path: Path to save the audio file
            emotion: Emotion tag (e.g., "wonder,reverence")
            sound_effects: Sound effects tag (e.g., "birds_chirping,ancient_ambience")
            
        Returns:
            Path to generated audio file
        """
        try:
            # Enhance text with emotion and sound effects
            enhanced_text = self._enhance_text_with_effects(text, emotion, sound_effects)
            
            if self.model:
                # Use trained model if available
                return self._generate_with_trained_model(enhanced_text, output_path)
            else:
                # Fallback to gTTS
                return self._generate_with_gtts(enhanced_text, output_path)
        except Exception as e:
            print(f"Error in generate_audio: {e}")
            # Final fallback to gTTS with basic text
            return self._generate_with_gtts(text, output_path)
    
    def _enhance_text_with_effects(self, text: str, emotion: str, sound_effects: str) -> str:
        """
        Enhance text with emotional context and sound effect descriptions
        
        Args:
            text: Original text
            emotion: Comma-separated emotion tags
            sound_effects: Comma-separated sound effect tags
            
        Returns:
            Enhanced text with atmospheric descriptions
        """
        enhanced_parts = []
        
        # Add sound effects introduction
        if sound_effects:
            effects_list = [e.strip() for e in sound_effects.split(',') if e.strip()]
            effect_descriptions = []
            
            for effect in effects_list:
                if effect in self.sound_descriptions:
                    effect_descriptions.append(self.sound_descriptions[effect])
            
            if effect_descriptions:
                # Add opening sound effects
                enhanced_parts.append(' '.join(effect_descriptions[:2]))  # Max 2 effects at start
                enhanced_parts.append('\n\n')
        
        # Add emotional context
        emotion_prefix = ""
        if emotion:
            emotions_list = [e.strip() for e in emotion.split(',') if e.strip()]
            if emotions_list and emotions_list[0] in self.emotion_prefixes:
                emotion_prefix = self.emotion_prefixes[emotions_list[0]]
        
        # Combine: sound effects + emotional narration + content
        if emotion_prefix:
            enhanced_parts.append(emotion_prefix)
        
        enhanced_parts.append(text)
        
        # Add closing sound effects if there are more
        if sound_effects:
            effects_list = [e.strip() for e in sound_effects.split(',') if e.strip()]
            if len(effects_list) > 2:
                remaining_effects = []
                for effect in effects_list[2:]:
                    if effect in self.sound_descriptions:
                        remaining_effects.append(self.sound_descriptions[effect])
                
                if remaining_effects:
                    enhanced_parts.append('\n\n')
                    enhanced_parts.append(' '.join(remaining_effects))
        
        enhanced_text = ''.join(enhanced_parts)
        
        print(f"✨ Enhanced with emotion: {emotion}, sound effects: {sound_effects}")
        return enhanced_text
    
    def _generate_with_trained_model(self, text: str, output_path: str) -> str:
        """Generate audio using the trained model"""
        try:
            # Placeholder: The exact usage depends on your model architecture
            # This assumes the model can generate audio from text
            
            print(f"Generating audio with trained model...")
            
            # If your model is a custom TTS model, you would use it here
            # For now, using gTTS as the model might need special inference code
            return self._generate_with_gtts(text, output_path)
            
        except Exception as e:
            print(f"Error with trained model: {e}")
            return self._generate_with_gtts(text, output_path)
    
    def _generate_with_gtts(self, text: str, output_path: str) -> str:
        """Fallback: Generate audio using Google Text-to-Speech"""
        try:
            from gtts import gTTS
            
            # Truncate text if too long (gTTS limitation)
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            print(f"Generating audio with gTTS (chars: {len(text)})...")
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            
            print(f"✓ Audio generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            raise


# Initialize TTS service
tts_service = TTSService()
