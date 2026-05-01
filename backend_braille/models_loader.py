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


def _bnb_max_memory():
    """Leave VRAM headroom for LoRA, Sentence-BERT, and CUDA; CPU bucket enables partial offload."""
    max_memory = {}
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(idx).total_memory
            gb = max(int(total / (1024**3) * 0.88), 1)
            max_memory[idx] = f"{gb}GiB"
    max_memory["cpu"] = "96GiB"
    return max_memory


def load_models():
    """Load LLaMA + LoRA + Sentence-BERT"""
    global tokenizer, model, sbert
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading base model with 4-bit quantization...")
        # Mixed GPU/CPU placement requires fp32 CPU offload flag or the 4-bit loader raises.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=_bnb_max_memory(),
            dtype=torch.float16,
            token=HF_TOKEN,
            low_cpu_mem_usage=True,
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
