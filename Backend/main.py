from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import random
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import os

# --- CONFIGURATION ---
# Path to the adapter folder (local)
ADAPTER_PATH = "Model/llama3_history_QA"
# Base model identifier (Unsloth Llama 3 base)
BASE_MODEL_ID = "unsloth/llama-3-8b"
SBERT_MODEL_PATH = "Model/sbert_history_evaluator_v4"
SYLLABUS_PATH = "HistorySyllubusDataSet.csv"

app = FastAPI()

# --- GLOBAL VARIABLES ---
df_syl = None
model = None
tokenizer = None
eval_model = None

# --- STARTUP: LOAD RESOURCES ---
@app.on_event("startup")
def load_resources():
    global df_syl, model, tokenizer, eval_model

    print("Loading Syllabus...")
    try:
        df_syl = pd.read_csv(SYLLABUS_PATH)
        # Normalize context column
        if 'Text Content' in df_syl.columns:
            df_syl['Context'] = df_syl['Text Content'].astype(str)
        elif 'Context' in df_syl.columns:
            df_syl['Context'] = df_syl['Context'].astype(str)
        else:
            col = df_syl.select_dtypes(include=['object']).columns[0]
            df_syl['Context'] = df_syl[col].astype(str)

        df_syl['Chapter_Clean'] = df_syl['Chapter'].astype(str).str.strip()
        print(f"Loaded {len(df_syl)} rows.")
    except Exception as e:
        print(f"Error loading syllabus: {e}")

    print("Loading Generator (Low VRAM Mode)...")
    try:
        # 1. Configure Quantization (8-bit + CPU Offload)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # 2. Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            offload_folder="offload", # Required for CPU offloading
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

        # 3. Load Adapter
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("Generator loaded successfully.")
    except Exception as e:
        print(f"Error loading Generator: {e}")

    print("Loading Evaluator (SBERT)...")
    try:
        eval_model = SentenceTransformer(SBERT_MODEL_PATH)
        print("Evaluator loaded.")
    except Exception as e:
        print(f"Error loading Evaluator, falling back: {e}")
        eval_model = SentenceTransformer('all-distilroberta-v1')

# --- DATA MODELS ---
class QuestionRequest(BaseModel):
    chapter_name: str

class QuestionResponse(BaseModel):
    question: str
    context: str
    correct_answer: str
    key_phrase: str

class EvaluateRequest(BaseModel):
    user_answer: str
    correct_answer: str
    key_phrase: str

class EvaluateResponse(BaseModel):
    score: float
    feedback: str

# --- ENDPOINTS ---
@app.post("/generate_question", response_model=QuestionResponse)
def generate_question(req: QuestionRequest):
    if df_syl is None or model is None:
        raise HTTPException(status_code=503, detail="Resources not loaded")

    subset = df_syl[df_syl['Chapter_Clean'] == req.chapter_name]
    if subset.empty:
        raise HTTPException(status_code=404, detail="Chapter not found")

    # Select random context
    idx = random.choice(subset.index.tolist())
    context = df_syl.loc[idx]['Context']

    # Prompt Engineering
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict Examiner. Output the Question, Answer, and Key Phrase.<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: "{context[:3500]}"
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
QUESTION:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, use_cache=True)
    
    resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Extraction Logic
    def extract(key):
        # Regex looking for Key: Value
        m = re.search(rf"{key}[:\\-]\\s*(.*?)(?=\\n[A-Z_]+:|$)", resp, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    q = extract("QUESTION") or resp.split('\\n')[0]
    a = extract("ANSWER")
    k = extract("KEY_PHRASE")

    if not a or len(a) < 2:
        if "?" in resp: a = resp.split("?")[-1].strip()
        else: a = "Refer to context."

    return QuestionResponse(
        question=q,
        context=context,
        correct_answer=a,
        key_phrase=k
    )

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    if eval_model is None:
        raise HTTPException(status_code=503, detail="Evaluator not loaded")

    # 1. Key Phrase Match
    if req.key_phrase and len(req.key_phrase) > 2 and req.key_phrase.lower() in req.user_answer.lower():
        return EvaluateResponse(score=1.0, feedback="🎯 Excellent! (Key Match)")

    # 2. Semantic Similarity
    emb1 = eval_model.encode(req.user_answer, convert_to_tensor=True)
    emb2 = eval_model.encode(req.correct_answer, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()

    if score < 0.50:
        feedback = "❌ Incorrect"
    elif score > 0.75:
        feedback = "✅ Correct"
    elif score > 0.60:
        feedback = "⚠️ Partially Correct"
    else:
        feedback = "❌ Incorrect"

    return EvaluateResponse(score=score, feedback=feedback)

@app.get("/")
def root():
    return {"message": "History Tutor Backend (Low VRAM) is Running"}