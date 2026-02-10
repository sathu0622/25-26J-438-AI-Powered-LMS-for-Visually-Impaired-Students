"""
Model Service - Manages AI model and curriculum loading
"""
import torch
from pathlib import Path


class AIModelService:
    """Service for managing AI models and curriculum recommendations"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize or load the model"""
        try:
            backend_dir = Path(__file__).parent.parent.parent
            model_path = backend_dir / "AI_History_Teacher_System.pth"
            
            if not model_path.exists():
                print(f"[INFO] Model file not found at {model_path}")
                print(f"[INFO] Using predefined chapter recommendations")
                return
            
            try:
                print(f"[INFO] Loading model from {model_path}...")
                self.model = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model_loaded = True
                print(f"[INFO] Model loaded successfully")
            except Exception as load_error:
                print(f"[WARN] Standard model load failed: {str(load_error)[:100]}...")
                print(f"[INFO] Using predefined chapter recommendations instead")
                print(f"[TIP] To enable model inference, define custom model classes in this file")
        
        except Exception as e:
            print(f"[WARN] Error during model initialization: {e}")
            print(f"[INFO] App will continue with predefined chapters")
    
    def get_chapter_recommendation(self, grade: int, chapter_num: int) -> dict:
        """Get chapter recommendation based on model or defaults"""
        # Default recommendations
        recommendations = {
            10: {
                1: {"title": "Sources of Studying History", "difficulty": "beginner"},
                2: {"title": "Ancient Settlements", "difficulty": "beginner"},
                3: {"title": "Evolution of Political Power in Sri Lanka", "difficulty": "intermediate"},
            },
            11: {
                1: {"title": "Industrial Revolution", "difficulty": "intermediate"},
                2: {"title": "Modern Era", "difficulty": "advanced"},
            }
        }
        
        if grade in recommendations and chapter_num in recommendations[grade]:
            return recommendations[grade][chapter_num]
        
        return {"title": f"Chapter {chapter_num}", "difficulty": "unknown"}


# Create global instance
ai_model_service = AIModelService()
