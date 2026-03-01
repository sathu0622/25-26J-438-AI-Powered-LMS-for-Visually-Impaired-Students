
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import hashlib
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter()

client = MongoClient(os.getenv('MONGO_URL'))
db = client[os.getenv('DATABASE_NAME')]
users_col = db['users']

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

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

# Add quiz history endpoint later
