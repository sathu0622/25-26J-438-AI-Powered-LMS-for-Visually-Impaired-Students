from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ollama
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import uvicorn
from Quiz import quiz_routes
from UserManagement import user_routes

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

print("\n🚀 Booting up History Tutor (Self-Correcting Edition)...")
try:
    ollama.list()
    print("      ✅ Ollama Connected.")
except:
    print("      ❌ Error: Ollama is not running!")
    exit()

print(f"   -> Loading SBERT...")
if os.path.exists(SBERT_PATH):
    sbert_model = SentenceTransformer(SBERT_PATH)
else:
    sbert_model = SentenceTransformer('all-distilroberta-v1')
print("      ✅ Evaluator Ready.")

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

# --- 3. STATE ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Share state with Quiz module
quiz_routes.sbert_model = sbert_model
quiz_routes.df_syl = df_syl
quiz_routes.AVAILABLE_CHAPTERS = AVAILABLE_CHAPTERS
quiz_routes.prefetch_cache = {}
quiz_routes.question_history = {}

# Mount routers
app.include_router(quiz_routes.router)
app.include_router(user_routes.router)

if __name__ == "__main__":
    print("\n✅ Server running on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)