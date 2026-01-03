from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import random
import re
import os
import uvicorn

# --- CONFIGURATION ---

OLLAMA_MODEL_NAME = "history-tutor" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SBERT_PATH = os.path.join(
    BASE_DIR,
    "Model",
    "sbert_history_evaluator_v4"
)

SYLLABUS_FILE = os.path.join(
    BASE_DIR,
    "HistorySyllubusDataSet.csv"
)

# --- 1. LOAD MODELS ---
print("\n🚀 Booting up History Tutor (Ollama Edition)...")

# A. Check if Ollama is running
try:
    print(f"   -> Connecting to Ollama...")
    # Simple test to see if the server is up
    ollama.list() 
    print("       Connection Established!")
except:
    print("      Error: Ollama is not running!")
    print("       Please open the 'Ollama' app on your computer first.")
    exit()

# B. Load SBERT 
print(f"   -> Loading SBERT...")
try:
    if os.path.exists(SBERT_PATH):
        sbert_model = SentenceTransformer(SBERT_PATH)
    else:
        sbert_model = SentenceTransformer('all-distilroberta-v1')
    print("      ✅ Evaluator Ready.")
except Exception as e:
    print(f"      ❌ SBERT Load Error: {e}")
    exit()

# --- 2. LOAD SYLLABUS ---
print("📚 Loading Syllabus...")
if os.path.exists(SYLLABUS_FILE):
    df_syl = pd.read_csv(SYLLABUS_FILE)
    if 'Context' in df_syl.columns: df_syl['Context'] = df_syl['Context'].astype(str)
    elif 'Text Content' in df_syl.columns: df_syl['Context'] = df_syl['Text Content'].astype(str)
    else: 
        col = df_syl.select_dtypes(include=['object']).columns[0]
        df_syl['Context'] = df_syl[col].astype(str)
    
    df_syl['Chapter_Clean'] = df_syl['Chapter'].astype(str).str.strip()
    AVAILABLE_CHAPTERS = sorted(df_syl['Chapter_Clean'].unique().tolist())
    print(f"      ✅ Loaded {len(AVAILABLE_CHAPTERS)} Chapters.")
else:
    print("      ❌ Error: Syllabus CSV missing.")
    df_syl = None
    AVAILABLE_CHAPTERS = []

# --- 3. MEMORY ---
question_history = {}
app = FastAPI(title="History Tutor API")

# CORS Middleware (Required for index.html)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. ENDPOINTS ---
class ChapterRequest(BaseModel):
    chapter_name: str

class AnswerRequest(BaseModel):
    user_answer: str
    correct_answer: str
    key_phrase: str

@app.get("/chapters")
def get_chapters():
    return {"chapters": AVAILABLE_CHAPTERS}

@app.post("/generate_question")
def generate_question(req: ChapterRequest):
    if df_syl is None: raise HTTPException(500, "Syllabus not loaded")
    
    subset = df_syl[df_syl['Chapter_Clean'] == req.chapter_name]
    if subset.empty: raise HTTPException(404, "Chapter not found")

    # Pick Random Context
    all_indices = subset.index.tolist()
    idx = random.choice(all_indices)
    context = df_syl.loc[idx]['Context']

    # Anti-Duplicate Logic
    history_key = (req.chapter_name, idx)
    previous_qs = question_history.get(history_key, [])
    avoid_instruction = ""
    if previous_qs:
        avoid_list = "; ".join(previous_qs[-3:])
        avoid_instruction = f"IMPORTANT: You have already asked: [{avoid_list}]. You MUST generate a completely DIFFERENT question."

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a History Examiner. Output Question, Answer, Key Phrase.<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: "{context[:3000]}"
{avoid_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
QUESTION:"""

    # (Ollama)
    try:
        response_obj = ollama.generate(model=OLLAMA_MODEL_NAME, prompt=prompt)
        resp = response_obj['response']
    except Exception as e:
        print(f"Ollama generation failed: {e}")
        raise HTTPException(500, "AI Generation Failed")
    
    # Parsing
    def extract(key):
        m = re.search(rf"{key}[:\-]\s*(.*?)(?=\n[A-Z_]+:|$)", resp, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    q = extract("QUESTION") or resp.split('\n')[0]
    a = extract("ANSWER")
    k = extract("KEY_PHRASE")

    if not a or len(a) < 2:
        if "?" in resp: a = resp.split("?")[-1].strip()
        else: a = "Refer to context."

    if history_key not in question_history: question_history[history_key] = []
    question_history[history_key].append(q)

    return {"question": q, "correct_answer": a, "key_phrase": k}

@app.post("/evaluate_answer")
def evaluate_answer(req: AnswerRequest):
    if req.key_phrase and len(req.key_phrase) > 2:
        if req.key_phrase.lower() in req.user_answer.lower():
            return {"score": 100, "feedback": "🎯 Excellent! (Key Match)", "correct": True}

    if req.correct_answer:
        emb1 = sbert_model.encode(req.user_answer, convert_to_tensor=True)
        emb2 = sbert_model.encode(req.correct_answer, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item()

        if score > 0.75: return {"score": int(score*100), "feedback": "✅ Correct", "correct": True}
        if score > 0.60: return {"score": int(score*100), "feedback": "⚠️ Partially Correct", "correct": True}
        return {"score": int(score*100), "feedback": "❌ Incorrect", "correct": False}

    return {"score": 0, "feedback": "Error", "correct": False}

if __name__ == "__main__":
    print("\n✅ Server running on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)