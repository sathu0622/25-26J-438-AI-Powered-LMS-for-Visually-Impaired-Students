from pydantic import BaseModel

class EvaluationRequest(BaseModel):
    question: str
    student_answer: str

class EvaluationResponse(BaseModel):
    question: str
    student_answer: str
    model_answer: str
    final_score: float
    semantic_similarity: float
    keyword_match: float
    jaccard_similarity: float
    error_penalty: str
    status: str
    feedback: str
