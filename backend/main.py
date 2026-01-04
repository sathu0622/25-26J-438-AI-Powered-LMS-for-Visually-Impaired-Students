"""
O/L History Answer Evaluation System - FastAPI Backend
"""
import os
import torch
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading and cleanup"""
    logger.info("Application startup: Loading models...")
    load_models()
    yield
    logger.info("Application shutdown: Cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete")

# Initialize FastAPI app with lifespan
app = FastAPI(title="O/L History Answer Evaluation System", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_MODEL_PATH = r"C:\Users\MSI\Desktop\Research\Model\ol_history_model\final_lora_model"
HF_TOKEN = os.getenv("HF_TOKEN")


# Global variables for models
tokenizer = None
model = None
sbert = None

# Request/Response models
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

# =======================
# Model Loading
# =======================

def load_models():
    """Load LLaMA + LoRA model and Sentence-BERT model"""
    global tokenizer, model, sbert
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading base model with 4-bit quantization...")
        logger.info("⚠️ Optimized for 6GB VRAM - generation will take 1-2 minutes")
        
        # 4-bit quantization without CPU offload (avoids bitsandbytes bug)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load with sequential device mapping (layers distributed across GPU/CPU automatically)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="sequential",  # Changed from "auto" to avoid meta tensor issues
            torch_dtype=torch.float16,
            token=HF_TOKEN,
            low_cpu_mem_usage=True
        )
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        model.eval()
        
        logger.info("Loading Sentence-BERT...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("✅ All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# =======================
# Evaluation Functions
# =======================

def generate_correct_answer(question: str) -> str:
    """Generate correct answer using the fine-tuned model"""
    torch.manual_seed(42)
    
    prompt = f"""You are an expert Sri Lankan O/L History teacher.
Answer the question clearly and factually in detailed, exam-style narrative suitable for a Grade 11 student.
Include all key historical points, but keep language simple.
Do NOT add unnecessary commentary or endnotes.

Question:
{question}

Answer:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the first available device
    if hasattr(model, 'hf_device_map'):
        first_device = list(model.hf_device_map.values())[0]
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    logger.info("🔄 Generating answer (1-2 minutes)...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.5,
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("✅ Answer generation completed")
    return response.split("Answer:")[-1].strip()

def semantic_similarity(correct: str, student: str) -> float:
    """Calculate semantic similarity using Sentence-BERT"""
    emb1 = sbert.encode(correct, convert_to_tensor=True)
    emb2 = sbert.encode(student, convert_to_tensor=True)
    return round(float(util.cos_sim(emb1, emb2)) * 100, 2)

def keyword_overlap_score(correct: str, student: str) -> float:
    """Calculate keyword overlap using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
        tfidf = vectorizer.fit_transform([correct, student])
        features = vectorizer.get_feature_names_out()
        
        correct_words = set(features[i] for i, v in enumerate(tfidf[0].toarray()[0]) if v > 0)
        student_words = set(features[i] for i, v in enumerate(tfidf[1].toarray()[0]) if v > 0)
        
        if not correct_words:
            return 0.0
        
        return round(len(correct_words & student_words) / len(correct_words) * 100, 2)
    except:
        return 0.0

def jaccard_similarity(correct: str, student: str) -> float:
    """Calculate Jaccard similarity"""
    def tokenize(text):
        text = re.sub(r'[^a-z\s]', '', text.lower())
        return set(text.split())
    
    a, b = tokenize(correct), tokenize(student)
    if not a or not b:
        return 0.0
    
    return round(len(a & b) / len(a | b) * 100, 2)

def length_penalty(correct: str, student: str) -> float:
    """Apply length penalty to score"""
    r = len(student.split()) / max(len(correct.split()), 1)
    if 0.5 <= r <= 1.5:
        return 1.0
    elif r < 0.5:
        return 0.8
    return 0.9

HISTORICAL_ERRORS = [
    "british", "factory", "industrial", "ignored agriculture",
    "little impact", "did not contribute", "traveling abroad"
]

def detect_historical_errors(answer: str) -> float:
    """Detect historical errors in answer"""
    errors = sum(1 for e in HISTORICAL_ERRORS if e in answer.lower())
    if errors >= 3:
        return 0.3
    elif errors == 2:
        return 0.5
    elif errors == 1:
        return 0.8
    return 1.0

def calculate_final_score(correct: str, student: str) -> tuple:
    """Calculate final score and component scores"""
    semantic = semantic_similarity(correct, student)
    keyword = keyword_overlap_score(correct, student)
    jaccard = jaccard_similarity(correct, student)
    
    length_factor = length_penalty(correct, student)
    error_factor = detect_historical_errors(student)
    
    final = (semantic * 0.65 + keyword * 0.35) * length_factor * error_factor
    
    if semantic >= 60 and keyword >= 50 and error_factor == 1.0:
        final += 5
    
    return round(final, 2), semantic, keyword, jaccard, error_factor

def generate_feedback(score: float) -> str:
    """Generate feedback based on score"""
    if score >= 70:
        return "Excellent answer with correct historical understanding."
    elif score >= 50:
        return "Good answer, but some important points can be improved."
    elif score >= 40:
        return "Basic understanding shown, but key facts are missing."
    else:
        return "Incorrect or weak answer. Please revise the lesson."

def evaluate_student_answer(question: str, student_answer: str) -> dict:
    """Main evaluation function"""
    correct_answer = generate_correct_answer(question)
    
    final, semantic, keyword, jaccard, error_penalty = calculate_final_score(
        correct_answer, student_answer
    )
    
    status = "PASS" if final >= 60 else "NEEDS IMPROVEMENT" if final >= 50 else "FAIL"
    
    return {
        "question": question,
        "student_answer": student_answer,
        "model_answer": correct_answer,
        "final_score": final,
        "semantic_similarity": semantic,
        "keyword_match": keyword,
        "jaccard_similarity": jaccard,
        "error_penalty": f"{int(error_penalty*100)}%",
        "status": status,
        "feedback": generate_feedback(final)
    }

# =======================
# API Endpoints
# =======================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "O/L History Answer Evaluation System API",
        "status": "running",
        "endpoints": {
            "evaluate": "/evaluate (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model is not None and sbert is not None
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(request: EvaluationRequest):
    """Evaluate student answer against model-generated correct answer"""
    if not request.question or not request.student_answer:
        raise HTTPException(
            status_code=400,
            detail="Question and student_answer cannot be empty"
        )
    
    if model is None or sbert is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please wait for startup to complete."
        )
    
    try:
        logger.info(f"Evaluating answer for question: {request.question[:50]}...")
        result = evaluate_student_answer(request.question, request.student_answer)
        return EvaluationResponse(**result)
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during evaluation: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)