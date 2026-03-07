
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter()

client = MongoClient(os.getenv('MONGO_URL'))
db = client[os.getenv('DATABASE_NAME')]
users_col = db['users']
quiz_sets_col = db['quiz_sets']  # For generative quizzes
adaptive_sessions_col = db['adaptive_sessions']  # For adaptive quizzes
past_paper_sessions_col = db['past_paper_sessions']  # For past paper quizzes

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class UserProfileResponse(BaseModel):
    username: str
    total_quizzes: int
    generative_quizzes: int
    adaptive_quizzes: int
    past_paper_quizzes: int
    average_score: float
    recent_activity: List[Dict]
    quiz_history: Dict[str, List[Dict]]

# Simple password hashing
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@router.post('/register')
def register_user(req: RegisterRequest):
    if users_col.find_one({'username': req.username}):
        raise HTTPException(status_code=400, detail='Username already exists')
    users_col.insert_one({
        'username': req.username,
        'password': hash_password(req.password),
        'quiz_history': []
    })
    return {'message': 'User registered successfully'}

@router.post('/login')
def login_user(req: LoginRequest):
    user = users_col.find_one({'username': req.username})
    if not user or user['password'] != hash_password(req.password):
        raise HTTPException(status_code=401, detail='Invalid username or password')
    return {'message': 'Login successful'}

# Quiz history endpoint
class QuizHistoryRequest(BaseModel):
    username: str
    quiz_result: dict  # e.g. {"chapter": "Chapter 1", "score": 80, "feedback": "Good job!"}

@router.post('/add_quiz_history')
def add_quiz_history(req: QuizHistoryRequest):
    user = users_col.find_one({'username': req.username})
    if not user:
        raise HTTPException(status_code=404, detail='User not found')
    users_col.update_one(
        {'username': req.username},
        {'$push': {'quiz_history': req.quiz_result}}
    )
    return {'message': 'Quiz history added'}

# Past Paper Quiz Performance Endpoints
class PastPaperQuestionResult(BaseModel):
    question: str
    user_answer: str
    correct_answer: str
    score: float
    correct: bool
    year: str

class SavePastPaperResultRequest(BaseModel):
    username: str
    chapter_name: str
    questions: List[PastPaperQuestionResult]
    total_score: float
    correct_count: int
    total_questions: int

@router.post('/past-paper/save-result')
def save_past_paper_result(req: SavePastPaperResultRequest):
    """Save past paper quiz performance to database"""
    user = users_col.find_one({'username': req.username})
    if not user:
        raise HTTPException(status_code=404, detail='User not found')
    
    # Create a new past paper session record
    session_data = {
        'username': req.username,
        'chapter_name': req.chapter_name,
        'questions': [q.dict() for q in req.questions],
        'total_score': req.total_score,
        'correct_count': req.correct_count,
        'total_questions': req.total_questions,
        'completed_at': datetime.utcnow(),
        'created_at': datetime.utcnow()
    }
    
    result = past_paper_sessions_col.insert_one(session_data)
    
    return {
        'message': 'Past paper quiz result saved successfully',
        'session_id': str(result.inserted_id)
    }

# Add quiz history endpoint later

@router.get('/profile/{username}', response_model=UserProfileResponse)
def get_user_profile(username: str):
    """Get comprehensive user profile with quiz history from both systems"""
    user = users_col.find_one({'username': username})
    if not user:
        raise HTTPException(status_code=404, detail='User not found')
    
    # Fetch generative quiz history
    generative_quizzes = list(quiz_sets_col.find({'username': username}).sort('created_at', -1))
    
    # Fetch adaptive quiz history
    adaptive_sessions = list(adaptive_sessions_col.find({'username': username, 'active': False}).sort('created_at', -1))
    
    # Fetch past paper quiz history
    past_paper_sessions = list(past_paper_sessions_col.find({'username': username}).sort('completed_at', -1))
    
    # Process generative quiz data
    generative_history = []
    generative_total_score = 0
    generative_count = 0
    
    for quiz_set in generative_quizzes:
        for attempt in quiz_set.get('attempts', []):
            if attempt.get('completed_at'):
                summary = attempt.get('summary', {})
                score_percentage = (summary.get('average_score', 0))
                
                generative_history.append({
                    'quiz_id': str(quiz_set['_id']),
                    'chapter_name': quiz_set.get('chapter_name', ''),
                    'score': score_percentage,
                    'correct_answers': summary.get('correct_count', 0),
                    'total_questions': summary.get('total_questions', 0),
                    'completed_at': attempt['completed_at'].isoformat() if attempt.get('completed_at') else None,
                    'quiz_type': 'Generative'
                })
                generative_total_score += score_percentage
                generative_count += 1
    
    # Process adaptive quiz data
    adaptive_history = []
    adaptive_total_score = 0
    adaptive_count = 0
    
    for session in adaptive_sessions:
        if session.get('asked'):
            # Calculate adaptive quiz stats
            total_questions = len(session.get('asked', []))
            # For adaptive, we don't store individual answer scores, so let's use theta as performance indicator
            theta = session.get('theta', 0.0)
            # Convert theta to percentage score (theta ranges from -3 to +3, center at 0)
            score_percentage = max(0, min(100, 50 + (theta / 3.0) * 50))
            
            adaptive_history.append({
                'session_id': str(session['_id']),
                'chapter_name': session.get('chapter_name', ''),
                'score': round(score_percentage, 1),
                'theta': round(theta, 2),
                'final_level': session.get('current_level', 'easy'),
                'total_questions': total_questions,
                'completed_at': session.get('updated_at', session.get('created_at')).isoformat() if session.get('updated_at') or session.get('created_at') else None,
                'quiz_type': 'Adaptive'
            })
            adaptive_total_score += score_percentage
            adaptive_count += 1
    
    # Process past paper quiz data
    past_paper_history = []
    past_paper_total_score = 0
    past_paper_count = 0
    
    for session in past_paper_sessions:
        score_percentage = session.get('total_score', 0)
        past_paper_history.append({
            'session_id': str(session['_id']),
            'chapter_name': session.get('chapter_name', ''),
            'score': round(score_percentage, 1),
            'correct_count': session.get('correct_count', 0),
            'total_questions': session.get('total_questions', 0),
            'completed_at': session.get('completed_at').isoformat() if session.get('completed_at') else None,
            'quiz_type': 'PastPaper'
        })
        past_paper_total_score += score_percentage
        past_paper_count += 1
    
    # Calculate overall statistics
    total_quizzes = generative_count + adaptive_count + past_paper_count
    average_score = 0.0
    if total_quizzes > 0:
        average_score = (generative_total_score + adaptive_total_score + past_paper_total_score) / total_quizzes
    
    # Combine and sort recent activity
    recent_activity = (generative_history + adaptive_history + past_paper_history)
    recent_activity.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
    recent_activity = recent_activity[:10]  # Last 10 activities
    
    return UserProfileResponse(
        username=username,
        total_quizzes=total_quizzes,
        generative_quizzes=generative_count,
        adaptive_quizzes=adaptive_count,
        past_paper_quizzes=past_paper_count,
        average_score=round(average_score, 1),
        recent_activity=recent_activity,
        quiz_history={
            'generative': generative_history,
            'adaptive': adaptive_history,
            'past_paper': past_paper_history
        }
    )

@router.get('/profile/{username}/stats')
def get_user_stats(username: str):
    """Get quick user statistics"""
    user = users_col.find_one({'username': username})
    if not user:
        raise HTTPException(status_code=404, detail='User not found')
    
    # Count completed quizzes
    generative_count = quiz_sets_col.count_documents({
        'username': username,
        'attempts.completed_at': {'$exists': True}
    })
    
    adaptive_count = adaptive_sessions_col.count_documents({
        'username': username,
        'active': False
    })
    
    past_paper_count = past_paper_sessions_col.count_documents({
        'username': username
    })
    
    return {
        'username': username,
        'total_quizzes': generative_count + adaptive_count + past_paper_count,
        'generative_quizzes': generative_count,
        'adaptive_quizzes': adaptive_count,
        'past_paper_quizzes': past_paper_count
    }
