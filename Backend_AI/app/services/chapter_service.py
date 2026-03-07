import pandas as pd
import os
from typing import List, Dict, Optional


class ChapterService:
    def __init__(self):
        self.grade10_df = None
        self.grade11_df = None
        self.load_datasets()
    
    def load_datasets(self):
        """Load Grade 10 and Grade 11 datasets from CSV"""
        # Try multiple possible paths for the datasets
        possible_paths_10 = [
            "data/grade10_dataset.csv",
            "../data/grade10_dataset.csv",
            "../../data/grade10_dataset.csv",
        ]
        
        possible_paths_11 = [
            "data/grade11_dataset.csv",
            "../data/grade11_dataset.csv",
            "../../data/grade11_dataset.csv",
        ]
        
        # Load Grade 10
        for path in possible_paths_10:
            if os.path.exists(path):
                try:
                    self.grade10_df = pd.read_csv(path, encoding='latin-1')
                    print(f"✓ Loaded Grade 10 dataset from {path}")
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        # Load Grade 11
        for path in possible_paths_11:
            if os.path.exists(path):
                try:
                    self.grade11_df = pd.read_csv(path, encoding='latin-1')
                    print(f"✓ Loaded Grade 11 dataset from {path}")
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        if self.grade10_df is None:
            print("⚠ Grade 10 dataset not found")
        if self.grade11_df is None:
            print("⚠ Grade 11 dataset not found")
    
    def get_chapters_by_grade(self, grade: int) -> List[Dict]:
        """Get unique chapters for a grade"""
        df = self.grade10_df if grade == 10 else self.grade11_df
        
        if df is None or 'chapter' not in df.columns:
            return []
        
        chapters = []
        unique_chapters = df['chapter'].unique()
        
        for idx, chapter in enumerate(unique_chapters):
            # Count topics in this chapter
            topic_count = len(df[df['chapter'] == chapter])
            
            chapters.append({
                'id': idx,
                'chapter_name': str(chapter),
                'grade': grade,
                'topic_count': topic_count
            })
        
        return chapters
    
    def get_topics_by_chapter(self, grade: int, chapter_idx: int) -> List[Dict]:
        """Get all topics/rows for a specific chapter"""
        df = self.grade10_df if grade == 10 else self.grade11_df
        
        if df is None or 'chapter' not in df.columns:
            return []
        
        # Get the chapter name by index
        chapters = df['chapter'].unique()
        if chapter_idx >= len(chapters):
            return []
        
        chapter_name = chapters[chapter_idx]
        chapter_data = df[df['chapter'] == chapter_name]
        
        topics = []
        # Use enumerate to create sequential indices instead of pandas indices
        for topic_idx, (_, row) in enumerate(chapter_data.iterrows()):
            topic = {
                'id': topic_idx,  # Use sequential index
                'topic_name': str(row.get('Grade/Topic', '')),
                'chapter': str(chapter_name),
                'grade': grade
            }
            
            # Add available content fields
            if 'original_text' in row.index:
                topic['original_text'] = str(row.get('original_text', ''))
            if 'simplified_text' in row.index:
                topic['simplified_text'] = str(row.get('simplified_text', ''))
            if 'narrative_text' in row.index:
                topic['narrative_text'] = str(row.get('narrative_text', ''))
            if 'summary' in row.index:
                topic['summary'] = str(row.get('summary', ''))
            if 'emotion' in row.index:
                topic['emotion'] = str(row.get('emotion', ''))
            if 'sound_effects' in row.index:
                topic['sound_effects'] = str(row.get('sound_effects', ''))
            
            topics.append(topic)
        
        return topics
    
    def get_topic_by_id(self, grade: int, chapter_idx: int, topic_idx: int) -> Optional[Dict]:
        """Get a specific topic"""
        topics = self.get_topics_by_chapter(grade, chapter_idx)
        
        if 0 <= topic_idx < len(topics):
            return topics[topic_idx]
        return None


# Initialize chapter service
chapter_service = ChapterService()
