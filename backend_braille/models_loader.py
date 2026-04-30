import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from config import BASE_MODEL, LORA_MODEL_PATH, HF_TOKEN
from logger_config import logger
from rag_retriever import load_rag_data

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
        bnb_config = BitsAndBytesConfig(    #4-bit quantization (memory optimization)
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True, #Improves compression efficiency.
            bnb_4bit_quant_type="nf4"   #Uses NF4 quantization.
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",    # Helps load large models on limited GPU memory
            torch_dtype=torch.float16,  #Speeds up inference and reduces memory usage
            token=HF_TOKEN,
            low_cpu_mem_usage=True  #Optimizes CPU memory while loading the model
        )

        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)  #final combined model
        model.eval()

        logger.info("Loading Sentence-BERT...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

        # NEW: build RAG index
        load_rag_data(sbert)

        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
