import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from config import BASE_MODEL, LORA_MODEL_PATH, HF_TOKEN
from logger_config import logger

tokenizer = None
model = None
sbert = None

def load_models():
    """Load LLaMA + LoRA + Sentence-BERT"""
    global tokenizer, model, sbert
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading base model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="sequential",
            torch_dtype=torch.float16,
            token=HF_TOKEN,
            low_cpu_mem_usage=True
        )

        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        model.eval()

        logger.info("Loading Sentence-BERT...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
