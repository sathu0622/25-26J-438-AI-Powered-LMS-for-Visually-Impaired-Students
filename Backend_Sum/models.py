# models.py
import torch
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from peft import PeftModel
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import TYPE_MODEL_PATH, T5_MODEL_DIR, QA_MODEL_NAME, DEVICE

print(f"Loading models on device: {DEVICE}")

def load_all_models():
    """Load all ML models."""
    print("Loading models...")
    
    # Resource type model
    type_model = load_model(str(TYPE_MODEL_PATH), compile=False)
    print(f"✓ Resource type model loaded")
    
    # Summarization model
    base_model_name = "google/flan-t5-base"
    summ_tokenizer = T5Tokenizer.from_pretrained(base_model_name)
    base_summ_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    summ_model = PeftModel.from_pretrained(base_summ_model, str(T5_MODEL_DIR))
    summ_model.to(DEVICE)
    summ_model.eval()
    print(f"✓ T5 summarization model loaded ({DEVICE})")
    
    # Q&A model
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    qa_model.to(DEVICE)
    qa_model.eval()
    print(f"✓ Q&A model loaded ({DEVICE})")
    
    print("Model loading complete!\n")
    
    return {
        "type_model": type_model,
        "summ_tokenizer": summ_tokenizer,
        "summ_model": summ_model,
        "qa_tokenizer": qa_tokenizer,
        "qa_model": qa_model
    }