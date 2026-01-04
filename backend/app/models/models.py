from pydantic import BaseModel
from typing import List, Optional

class Subsection(BaseModel):
    id: str
    title: str
    duration: int  # in minutes
    audio_url: Optional[str] = None
    description: str

class Lesson(BaseModel):
    id: str
    title: str
    description: str
    grade: int  # 10 or 11
    subsections: List[Subsection]
    thumbnail: Optional[str] = None

class Grade(BaseModel):
    grade: int
    lessons: List[Lesson]
