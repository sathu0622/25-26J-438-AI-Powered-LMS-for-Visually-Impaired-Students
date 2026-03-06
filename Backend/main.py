from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from dotenv import load_dotenv
import pandas as pd
import os
import uvicorn
from Quiz import quiz_routes
from Quiz.quiz_routes import load_past_paper_data
from UserManagement import user_routes
import adaptive_routes

# --- CONFIGURATION ---

HF_MODEL_REPO = "KavindyaD/history-tuter-gguf"  # Hugging face repo name
HF_MODEL_FILENAME = "History_Tutor_Llama3.gguf"  # Model file name in the repo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load env early so HF_TOKEN is available even when started from a different cwd
load_dotenv(os.path.join(BASE_DIR, ".env"))

SBERT_PATH = os.path.join(
    BASE_DIR,
    "Model",
    "sbert_sts_history_v18"
)

SYLLABUS_FILE = os.path.join(
    BASE_DIR,
    "HistorySyllubusDataSet.csv"
)

print("\n🚀 Booting up History Tutor")

load_dotenv()  # load HF_TOKEN and other secrets from .env

print("   -> Loading LLM from Hugging Face (GGUF)...")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN env var is required to pull the model from Hugging Face.")

llm = Llama.from_pretrained(
    repo_id=HF_MODEL_REPO,
    filename=HF_MODEL_FILENAME,
    hf_token=HF_TOKEN,
    n_ctx=4096,
)
print("      ✅ LLM Ready.")

print(f"   -> Loading SBERT from {SBERT_PATH}...")
if not os.path.exists(SBERT_PATH):
    raise RuntimeError(
        "SBERT model directory sbert_sts_history_v18 is missing. "
        f"Expected at: {SBERT_PATH}"
    )
sbert_model = SentenceTransformer(SBERT_PATH)
print("      ✅ Evaluator Ready (sbert_sts_history_v18).")

print(" Loading Syllabus...")
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

# Load past paper data
print(" Loading Past Paper Questions...")
load_past_paper_data()

# --- 3. STATE ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Share state with Quiz module
quiz_routes.sbert_model = sbert_model
quiz_routes.df_syl = df_syl
quiz_routes.AVAILABLE_CHAPTERS = AVAILABLE_CHAPTERS
quiz_routes.prefetch_cache = {}
quiz_routes.question_history = {}
quiz_routes.llm = llm

# Share SBERT model and LLM with Adaptive module
adaptive_routes.sbert_model = sbert_model
adaptive_routes.llm = llm  # For generating smart MCQ distractors

# Mount routers
app.include_router(quiz_routes.router)
app.include_router(user_routes.router)
app.include_router(adaptive_routes.router)

if __name__ == "__main__":
    print("\n✅ Server running on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)