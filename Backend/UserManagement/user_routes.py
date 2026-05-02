
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
freetext_sessions_col = db['freetext_sessions']  # For free text quizzes
timed_quiz_sessions_col = db['timed_quiz_sessions']  # Timed mixed-chapter MCQ quiz

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChapterStats(BaseModel):
    chapter_name: str
    total_quizzes: int
    average_score: float
    best_score: float
    last_attempted: Optional[str]
    quiz_types: Dict[str, int]  # Count by quiz type

class SavedQuiz(BaseModel):
    id: str
    chapter_name: str
    quiz_type: str  # 'generative', 'freetext', 'adaptive', 'past_paper'
    created_at: str
    total_questions: int
    attempts_count: int
    last_score: Optional[float]
    can_retake: bool

class UserProfileResponse(BaseModel):
    username: str
    total_quizzes: int
    generative_quizzes: int
    adaptive_quizzes: int
    past_paper_quizzes: int
    freetext_quizzes: int
    timed_quizzes: int
    average_score: float
    recent_activity: List[Dict]
    quiz_history: Dict[str, List[Dict]]
    chapter_stats: List[ChapterStats]
    saved_quizzes: List[SavedQuiz]

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
    
    # Fetch all quiz data
    generative_quizzes = list(quiz_sets_col.find({'username': username}).sort('created_at', -1))
    adaptive_sessions = list(adaptive_sessions_col.find({'username': username}).sort('created_at', -1))
    past_paper_sessions = list(past_paper_sessions_col.find({'username': username}).sort('completed_at', -1))
    freetext_sessions = list(freetext_sessions_col.find({'username': username}).sort('created_at', -1))
    
    # Process generative quiz data
    generative_history = []
    generative_total_score = 0
    generative_count = 0
    chapter_data = {}  # For chapter-wise stats
    
    for quiz_set in generative_quizzes:
        chapter_name = quiz_set.get('chapter_name', '')
        if chapter_name not in chapter_data:
            chapter_data[chapter_name] = {'scores': [], 'dates': [], 'types': {'generative': 0, 'adaptive': 0, 'past_paper': 0, 'freetext': 0, 'timed_quiz': 0}}
        chapter_data[chapter_name]['types']['generative'] += 1
        
        for attempt in quiz_set.get('attempts', []):
            if attempt.get('completed_at'):
                summary = attempt.get('summary', {})
                score_percentage = (summary.get('average_score', 0))
                
                generative_history.append({
                    'quiz_id': str(quiz_set['_id']),
                    'chapter_name': chapter_name,
                    'score': score_percentage,
                    'correct_answers': summary.get('correct_count', 0),
                    'total_questions': summary.get('total_questions', 0),
                    'completed_at': attempt['completed_at'].isoformat() if attempt.get('completed_at') else None,
                    'quiz_type': 'Generative'
                })
                chapter_data[chapter_name]['scores'].append(score_percentage)
                chapter_data[chapter_name]['dates'].append(attempt['completed_at'])
                generative_total_score += score_percentage
                generative_count += 1
    
    # Process adaptive quiz data
    adaptive_history = []
    adaptive_total_score = 0
    adaptive_count = 0
    
    for session in adaptive_sessions:
        chapter_name = session.get('chapter_name', '')
        if chapter_name not in chapter_data:
            chapter_data[chapter_name] = {'scores': [], 'dates': [], 'types': {'generative': 0, 'adaptive': 0, 'past_paper': 0, 'freetext': 0, 'timed_quiz': 0}}
        chapter_data[chapter_name]['types']['adaptive'] += 1
        
        if session.get('asked'):
            total_questions = len(session.get('asked', []))
            theta = session.get('theta', 0.0)
            score_percentage = max(0, min(100, 50 + (theta / 3.0) * 50))
            
            adaptive_history.append({
                'session_id': str(session['_id']),
                'chapter_name': chapter_name,
                'score': round(score_percentage, 1),
                'theta': round(theta, 2),
                'final_level': session.get('current_level', 'easy'),
                'total_questions': total_questions,
                'completed_at': session.get('updated_at', session.get('created_at')).isoformat() if session.get('updated_at') or session.get('created_at') else None,
                'quiz_type': 'Adaptive'
            })
            chapter_data[chapter_name]['scores'].append(score_percentage)
            adaptive_total_score += score_percentage
            adaptive_count += 1
    
    # Process past paper quiz data
    past_paper_history = []
    past_paper_total_score = 0
    past_paper_count = 0
    
    for session in past_paper_sessions:
        chapter_name = session.get('chapter_name', '')
        if chapter_name not in chapter_data:
            chapter_data[chapter_name] = {'scores': [], 'dates': [], 'types': {'generative': 0, 'adaptive': 0, 'past_paper': 0, 'freetext': 0, 'timed_quiz': 0}}
        chapter_data[chapter_name]['types']['past_paper'] += 1
        
        score_percentage = session.get('total_score', 0)
        past_paper_history.append({
            'session_id': str(session['_id']),
            'chapter_name': chapter_name,
            'score': round(score_percentage, 1),
            'correct_count': session.get('correct_count', 0),
            'total_questions': session.get('total_questions', 0),
            'completed_at': session.get('completed_at').isoformat() if session.get('completed_at') else None,
            'quiz_type': 'PastPaper'
        })
        chapter_data[chapter_name]['scores'].append(score_percentage)
        past_paper_total_score += score_percentage
        past_paper_count += 1
    
    # Process free text quiz data
    freetext_history = []
    freetext_total_score = 0
    freetext_count = 0
    
    for session in freetext_sessions:
        chapter_name = session.get('chapter_name', '')
        if chapter_name not in chapter_data:
            chapter_data[chapter_name] = {'scores': [], 'dates': [], 'types': {'generative': 0, 'adaptive': 0, 'past_paper': 0, 'freetext': 0, 'timed_quiz': 0}}
        chapter_data[chapter_name]['types']['freetext'] += 1
        
        # Get the latest completed attempt
        attempts = session.get('attempts', [])
        completed_attempts = [a for a in attempts if a.get('completed_at')]
        if completed_attempts:
            latest_attempt = completed_attempts[-1]
            summary = latest_attempt.get('summary', {})
            score_percentage = summary.get('average_score', 0)
            
            freetext_history.append({
                'session_id': str(session['_id']),
                'chapter_name': chapter_name,
                'score': score_percentage,
                'total_questions': summary.get('total_questions', 0),
                'completed_at': latest_attempt.get('completed_at').isoformat() if latest_attempt.get('completed_at') else None,
                'quiz_type': 'FreeText'
            })
            chapter_data[chapter_name]['scores'].append(score_percentage)
            freetext_total_score += score_percentage
            freetext_count += 1
    
    # Timed quiz (mixed chapters, 20 MCQs)
    timed_quiz_sessions = list(
        timed_quiz_sessions_col.find({'username': username, 'status': 'completed'}).sort('completed_at', -1)
    )
    timed_history = []
    timed_total_score = 0
    timed_count = 0
    timed_chapter_label = 'Timed quiz (mixed chapters)'

    for session in timed_quiz_sessions:
        if timed_chapter_label not in chapter_data:
            chapter_data[timed_chapter_label] = {
                'scores': [],
                'dates': [],
                'types': {'generative': 0, 'adaptive': 0, 'past_paper': 0, 'freetext': 0, 'timed_quiz': 0},
            }
        chapter_data[timed_chapter_label]['types']['timed_quiz'] += 1
        score_percentage = float(session.get('average_score') or 0)
        completed = session.get('completed_at')
        timed_history.append({
            'session_id': str(session['_id']),
            'chapter_name': timed_chapter_label,
            'score': round(score_percentage, 1),
            'correct_count': session.get('correct_count', 0),
            'total_questions': session.get('total_questions', 20),
            'completed_at': completed.isoformat() if completed else None,
            'quiz_type': 'TimedQuiz',
        })
        chapter_data[timed_chapter_label]['scores'].append(score_percentage)
        if completed:
            chapter_data[timed_chapter_label]['dates'].append(completed)
        timed_total_score += score_percentage
        timed_count += 1
    
    # Calculate chapter statistics
    chapter_stats = []
    for chapter_name, data in chapter_data.items():
        if data['scores']:
            avg_score = sum(data['scores']) / len(data['scores'])
            best_score = max(data['scores'])
            last_date = max(data['dates']) if data['dates'] else None
            total_quizzes = sum(data['types'].values())
            
            chapter_stats.append(ChapterStats(
                chapter_name=chapter_name,
                total_quizzes=total_quizzes,
                average_score=round(avg_score, 1),
                best_score=round(best_score, 1),
                last_attempted=last_date.isoformat() if last_date else None,
                quiz_types=data['types']
            ))
    
    chapter_stats.sort(key=lambda x: x.last_attempted or '', reverse=True)
    
    # Collect saved quizzes (all quiz sets/sessions that can be retaken)
    saved_quizzes = []
    
    # Generative quiz sets
    for quiz_set in generative_quizzes:
        attempts = quiz_set.get('attempts', [])
        completed_attempts = [a for a in attempts if a.get('completed_at')]
        last_score = None
        if completed_attempts:
            last_attempt = completed_attempts[-1]
            summary = last_attempt.get('summary', {})
            last_score = summary.get('average_score')
        
        saved_quizzes.append(SavedQuiz(
            id=str(quiz_set['_id']),
            chapter_name=quiz_set.get('chapter_name', ''),
            quiz_type='generative',
            created_at=quiz_set.get('created_at').isoformat() if quiz_set.get('created_at') else '',
            total_questions=len(quiz_set.get('questions', [])),
            attempts_count=len(completed_attempts),
            last_score=last_score,
            can_retake=True
        ))
    
    # Free text sessions
    for session in freetext_sessions:
        attempts = session.get('attempts', [])
        completed_attempts = [a for a in attempts if a.get('completed_at')]
        last_score = None
        if completed_attempts:
            last_attempt = completed_attempts[-1]
            summary = last_attempt.get('summary', {})
            last_score = summary.get('average_score')
        
        saved_quizzes.append(SavedQuiz(
            id=str(session['_id']),
            chapter_name=session.get('chapter_name', ''),
            quiz_type='freetext',
            created_at=session.get('created_at').isoformat() if session.get('created_at') else '',
            total_questions=len(session.get('questions', [])),
            attempts_count=len(completed_attempts),
            last_score=last_score,
            can_retake=True
        ))

    timed_sets_label = 'Timed quiz (mixed chapters)'
    timed_saved = list(timed_quiz_sessions_col.find({
        'username': username,
        'status': 'completed',
    }).sort('completed_at', -1))
    for ts in timed_saved:
        completed = ts.get('completed_at')
        created_fallback = ts.get('created_at')
        ts_stamp = completed or created_fallback
        saved_quizzes.append(SavedQuiz(
            id=str(ts['_id']),
            chapter_name=timed_sets_label,
            quiz_type='timed',
            created_at=ts_stamp.isoformat() if ts_stamp else '',
            total_questions=int(ts.get('total_questions', 0) or len(ts.get('items', [])) or 0),
            attempts_count=1,
            last_score=float(ts['average_score']) if ts.get('average_score') is not None else None,
            can_retake=True,
        ))

    # Sort saved quizzes by creation date (most recent first)
    saved_quizzes.sort(key=lambda x: x.created_at, reverse=True)
    
    # Calculate overall statistics
    total_quizzes = generative_count + adaptive_count + past_paper_count + freetext_count + timed_count
    average_score = 0.0
    if total_quizzes > 0:
        average_score = (
            generative_total_score
            + adaptive_total_score
            + past_paper_total_score
            + freetext_total_score
            + timed_total_score
        ) / total_quizzes
    
    # Combine and sort recent activity
    recent_activity = (
        generative_history + adaptive_history + past_paper_history + freetext_history + timed_history
    )
    recent_activity.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
    recent_activity = recent_activity[:10]  # Last 10 activities
    
    return UserProfileResponse(
        username=username,
        total_quizzes=total_quizzes,
        generative_quizzes=generative_count,
        adaptive_quizzes=adaptive_count,
        past_paper_quizzes=past_paper_count,
        freetext_quizzes=freetext_count,
        timed_quizzes=timed_count,
        average_score=round(average_score, 1),
        recent_activity=recent_activity,
        quiz_history={
            'generative': generative_history,
            'adaptive': adaptive_history,
            'past_paper': past_paper_history,
            'freetext': freetext_history,
            'timed': timed_history,
        },
        chapter_stats=chapter_stats,
        saved_quizzes=saved_quizzes
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
    
    timed_count = timed_quiz_sessions_col.count_documents({
        'username': username,
        'status': 'completed',
    })
    
    return {
        'username': username,
        'total_quizzes': generative_count + adaptive_count + past_paper_count + timed_count,
        'generative_quizzes': generative_count,
        'adaptive_quizzes': adaptive_count,
        'past_paper_quizzes': past_paper_count,
        'timed_quizzes': timed_count,
    }
