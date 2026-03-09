from pydantic import BaseModel
from typing import List


class Topic(BaseModel):
    id: int
    title: str
    content: str
    duration: str


class Lesson(BaseModel):
    id: int
    title: str
    description: str
    duration: str
    grade: int
    topics: List[Topic]
