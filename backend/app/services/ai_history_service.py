"""
AI History Teacher Model Service - Uses CSV dataset and trained model
This service loads lessons from CSV files or the trained AI model
Based on the Jupyter notebook implementation
"""
import torch
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


class AIHistoryModelService:
    """
    Service to load and use the trained AI_History_Teacher_System model
    Generates chapters and lessons based on Grade 10/11 history curriculum
    Loads data from CSV files (preferred) or .pth model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        self.chapters_data = {10: {}, 11: {}}
        self.backend_dir = Path(__file__).parent.parent.parent
        self.data_folder = self.backend_dir / "data"
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """Load curriculum from CSV files or trained model"""
        print("=" * 60)
        print("[CURRICULUM] AI History Teacher System - Loading Curriculum")
        print("=" * 60)
        
        # Priority 1: Try loading from CSV files
        if self._load_from_csv():
            print("[OK] Successfully loaded curriculum from CSV files")
            return True
        
        # Priority 2: Try loading from PyTorch model
        if self._load_from_pytorch_model():
            print("[OK] Successfully loaded curriculum from .pth model")
            return True
        
        # Fallback: Use predefined curriculum
        print("[WARN] Using fallback predefined curriculum")
        self.initialize_default_chapters()
        return False
    
    def _load_from_csv(self) -> bool:
        """Load curriculum data from CSV files (matching notebook implementation)"""
        try:
            # Load Grade 10
            grade10_csv = self.data_folder / "grade10_dataset.csv"
            # Load Grade 11
            grade11_csv = self.data_folder / "grade11_dataset.csv"
            
            # Load Grade 10 with latin-1 encoding
            if os.path.exists(grade10_csv):
                print(f"[DATA] Loading Grade 10 data from: {grade10_csv}")
                df_g10 = pd.read_csv(grade10_csv, encoding='latin-1')
                self._parse_csv_to_chapters(df_g10, 10)
            
            # Load Grade 11 with latin-1 encoding
            if os.path.exists(grade11_csv):
                print(f"[DATA] Loading Grade 11 data from: {grade11_csv}")
                df_g11 = pd.read_csv(grade11_csv, encoding='latin-1')
                self._parse_csv_to_chapters(df_g11, 11)
            
            # Check if we loaded any data
            if self.chapters_data[10] or self.chapters_data[11]:
                return True
            
            return False
            
        except Exception as e:
            print(f"[WARN] Could not load from CSV: {e}")
            return False
    
    def _parse_csv_to_chapters(self, df: pd.DataFrame, grade: int):
        """Parse CSV DataFrame into chapters structure"""
        try:
            for idx, row in df.iterrows():
                # Extract chapter number from 'chapter' column (e.g., "1.Sources of Studying History")
                chapter_str = str(row.get('chapter', f'Chapter_{idx}'))
                # Split on dot to get chapter number
                chapter_parts = chapter_str.split('.')
                chapter_num = chapter_parts[0].strip()
                
                # Extract topic from 'Grade/Topic' column (e.g., "Grade 10: Classification of Sources")
                grade_topic = str(row.get('Grade/Topic', 'Untitled'))
                topic_title = grade_topic.replace(f'Grade {grade}: ', '')
                
                # Get content from various text fields
                original_text = str(row.get('original_text', ''))
                simplified_text = str(row.get('simplified_text', ''))
                narrative_text = str(row.get('narrative_text', ''))
                emotion = str(row.get('emotion', 'neutral'))
                sound_effects = str(row.get('sound_effects', ''))
                
                # Create chapter key
                chapter_key = str(chapter_num)
                
                # Initialize chapter if it doesn't exist
                if chapter_key not in self.chapters_data[grade]:
                    # Extract chapter title (everything after the dot)
                    chapter_title = '.'.join(chapter_parts[1:]).strip() if len(chapter_parts) > 1 else f'Chapter {chapter_num}'
                    self.chapters_data[grade][chapter_key] = {
                        'chapter_num': chapter_num,
                        'title': chapter_title,
                        'topics': []
                    }
                
                # Create topic ID from chapter and row index
                topic_id = f"{grade}_ch{chapter_num}_{idx}"
                
                # Add topic to chapter
                topic_data = {
                    'id': topic_id,
                    'title': topic_title,
                    'name': topic_title,
                    'content': original_text[:500] if original_text else 'No content available',
                    'simplified_content': simplified_text[:500] if simplified_text else '',
                    'narrative_content': narrative_text[:500] if narrative_text else '',
                    'full_content': original_text,
                    'emotion': emotion,
                    'sound_effects': sound_effects,
                    'chapter_title': self.chapters_data[grade][chapter_key]['title']
                }
                self.chapters_data[grade][chapter_key]['topics'].append(topic_data)
                print(f"[DATA] Loaded Grade {grade} Chapter {chapter_num}: {topic_title}")
            
            # Print summary
            for chapter_key, chapter in self.chapters_data[grade].items():
                chapter_num_str = str(chapter['chapter_num']).split('.')[-1] if '.' in str(chapter['chapter_num']) else str(chapter['chapter_num'])
                print(f"  [+] Chapter {chapter_num_str}: {chapter['title']} ({len(chapter['topics'])} topics)")
            
        except Exception as e:
            print(f"[WARN] Error parsing CSV: {e}")
    
    def _load_from_pytorch_model(self) -> bool:
        """Load curriculum from the trained .pth model"""
        try:
            model_path = self.backend_dir / "AI_History_Teacher_System.pth"
            
            if not os.path.exists(model_path):
                print(f"[WARN] Model file not found: {model_path}")
                return False
            
            print(f"[INFO] Loading model from: {model_path}")
            
            # Try to load the model
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"[OK] Model loaded successfully!")
            
            self.model_loaded = True
            
            # Try to extract chapters from the model
            if isinstance(checkpoint, dict) and 'chapters' in checkpoint:
                self.chapters_data = checkpoint.get('chapters_data', {})
                print(f"[OK] Extracted {len(self.chapters_data)} chapters from model")
            
            return True
            
        except Exception as e:
            print(f"[WARN] Could not extract chapters from model: {e}")
            return False
    
    def initialize_default_chapters(self):
        """Initialize with default/fallback curriculum"""
        # Grade 10 chapters
        self.chapters_data[10] = {
            '1': {
                'chapter_num': 1,
                'title': 'Sources of Studying History',
                'topics': [
                    {'id': '10_ch1_1', 'title': 'Importance of Historical Sources', 'name': 'Importance of Historical Sources', 'content': 'Understanding how historians use sources to study the past'},
                    {'id': '10_ch1_2', 'title': 'Primary and Secondary Sources', 'name': 'Primary and Secondary Sources', 'content': 'Learning the difference between primary and secondary historical sources'},
                    {'id': '10_ch1_3', 'title': 'Evaluating Historical Evidence', 'name': 'Evaluating Historical Evidence', 'content': 'Developing critical thinking skills to analyze historical sources'}
                ]
            },
            '2': {
                'chapter_num': 2,
                'title': 'Ancient Settlements',
                'topics': [
                    {'id': '10_ch2_1', 'title': 'Early Human Settlements', 'name': 'Early Human Settlements', 'content': 'Understanding how humans settled in different regions'},
                    {'id': '10_ch2_2', 'title': 'Development of Villages', 'name': 'Development of Villages', 'content': 'Learning about the transition from nomadic to settled societies'},
                    {'id': '10_ch2_3', 'title': 'Ancient Sri Lankan Settlements', 'name': 'Ancient Sri Lankan Settlements', 'content': 'Exploring the early settlements in Sri Lanka'}
                ]
            }
        }
        
        # Grade 11 chapters
        self.chapters_data[11] = {
            '1': {
                'chapter_num': 1,
                'title': 'Industrial Revolution',
                'topics': [
                    {'id': '11_ch1_1', 'title': 'Origins of Industrial Revolution', 'name': 'Origins of Industrial Revolution', 'content': 'Understanding the causes and beginning of the Industrial Revolution in Britain'},
                    {'id': '11_ch1_2', 'title': 'Technological Innovations', 'name': 'Technological Innovations', 'content': 'Learning about key inventions that powered the Industrial Revolution'},
                    {'id': '11_ch1_3', 'title': 'Social Changes', 'name': 'Social Changes', 'content': 'Exploring how the Industrial Revolution transformed society'}
                ]
            }
        }
    
    def get_chapters_by_grade(self, grade: int) -> List[Dict]:
        """Get all chapters for a specific grade"""
        chapters = []
        if grade in self.chapters_data:
            for chapter_key, chapter in self.chapters_data[grade].items():
                chapters.append({
                    'id': f"g{grade}_ch{chapter['chapter_num']}",
                    'number': chapter['chapter_num'],
                    'title': chapter['title'],
                    'topics_count': len(chapter.get('topics', []))
                })
        
        if chapters:
            print(f"[OK] Retrieved {len(chapters)} chapters for Grade {grade} from CSV data")
        else:
            print(f"[WARN] No chapters found for grade {grade}")
        
        return chapters
    
    def get_chapter_details(self, grade: int, chapter_num: str) -> Optional[Dict]:
        """Get detailed information about a specific chapter"""
        chapter_key = str(chapter_num)
        if grade in self.chapters_data and chapter_key in self.chapters_data[grade]:
            return self.chapters_data[grade][chapter_key]
        return None
    
    def get_topics_by_chapter(self, grade: int, chapter_num: str) -> List[Dict]:
        """Get all topics for a specific chapter"""
        chapter_details = self.get_chapter_details(grade, chapter_num)
        if chapter_details:
            return chapter_details.get('topics', [])
        return []
    
    def get_topic_by_id(self, grade: int, chapter_num: str, topic_id: str) -> Optional[Dict]:
        """Get a specific topic by ID"""
        topics = self.get_topics_by_chapter(grade, chapter_num)
        for topic in topics:
            if topic['id'] == topic_id:
                return topic
        return None
    
    def get_chapter_by_id(self, grade: int, chapter_id: str) -> Optional[Dict]:
        """Get a specific chapter by ID (with full details)"""
        # Extract chapter number from chapter_id (e.g., 'g10_ch1' -> '1')
        chapter_num = chapter_id.split('_ch')[-1] if '_ch' in chapter_id else chapter_id
        chapter_key = str(chapter_num)
        
        if grade in self.chapters_data and chapter_key in self.chapters_data[grade]:
            chapter = self.chapters_data[grade][chapter_key]
            return {
                'id': f"g{grade}_ch{chapter['chapter_num']}",
                'number': chapter['chapter_num'],
                'title': chapter['title'],
                'description': f"Learn about {chapter['title']}",
                'estimated_duration': len(chapter.get('topics', [])) * 5,
                'topics': chapter.get('topics', []),
                'learning_objectives': []
            }
        return None
    
    def get_chapter_topics(self, grade: int, chapter_id: str) -> List[Dict]:
        """Get all topics for a specific chapter (alternative method name)"""
        # Extract chapter number from chapter_id
        chapter_num = chapter_id.split('_ch')[-1] if '_ch' in chapter_id else chapter_id
        return self.get_topics_by_chapter(grade, chapter_num)


# Create global instance
ai_history_service = AIHistoryModelService()
