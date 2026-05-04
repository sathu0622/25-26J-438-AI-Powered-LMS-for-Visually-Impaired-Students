import os
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

LORA_MODEL_PATH = "Abishaan/braille_ol_history"

# Hugging Face Token (from .env)
HF_TOKEN = os.getenv("HF_TOKEN")

BRAILLE_LABELS = list("abcdefghijklmnopqrstuvwxyz")