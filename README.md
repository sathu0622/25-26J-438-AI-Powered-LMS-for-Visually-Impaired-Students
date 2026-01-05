# 📘 AI-Powered Document Processing & Summarization Backend

## 📌 Project Overview

This project implements an **AI-powered backend system** for extracting and summarizing historical content from **Books, Magazines, and Newspapers**.  
The system is specially designed to support **visually impaired students** by generating **clear, structured, and voice-friendly summaries**.

The backend automatically:
- Extracts text using OCR
- Detects the resource type
- Applies context-aware summarization using a **LoRA fine-tuned FLAN-T5 model**

---

## 🎯 Key Features

- 📄 Supports **PDF and Image documents**
- 🔍 OCR-based text extraction (without grammar correction)
- 🧠 Automatic resource type detection (Book / Magazine / Newspaper)
- 📰 Article splitting for newspapers
- ✍️ Context-aware summarization
- 🚀 FastAPI backend
- 💾 Lightweight LoRA model (~25–100MB)

---

## 🧠 System Architecture

User Upload (PDF / Image)
          ↓
FastAPI Backend
          ↓
OCR (Tesseract)
          ↓
Resource Type Detection (CNN)
          ↓
Article Segmentation (Newspapers)
          ↓
FLAN-T5 + LoRA Summarization
          ↓
JSON Output (Voice-ready)


---

## 🛠️ Technologies Used

### Backend
- Python 3.9+
- FastAPI
- PyTorch
- TensorFlow / Keras
- OpenCV
- Tesseract OCR
- Poppler (PDF to Image)

### AI Models
- CNN (Keras) – Resource type classification
- FLAN-T5-Base (Google)
- LoRA (PEFT) – Parameter-efficient fine-tuning

---

## 📂 Project Structure


project-root/
│
├── Model/
│ ├── book_magazine_newspaper_model_super_finetuned_FIXED.keras
│ └── final/ # FLAN-T5 + LoRA adapter
│
├── main.py # FastAPI backend
├── README.md
└── requirements.txt



---

## ⚙️ Model Details

### 🔹 Resource Type Classifier
- CNN-based image classification model
- Classes:
  - Books
  - Magazine
  - Newspapers
- Input size: 224 × 224

### 🔹 Summarization Model
- Base model: `google/flan-t5-base`
- Fine-tuning method: LoRA (PEFT)
- Trainable parameters: ~2.7%
- Checkpoint size: ~25–100MB
- Evaluation Results:
  - ROUGE-1: 49.38
  - ROUGE-2: 20.85
  - ROUGE-L: 30.80

---

## 🧪 OCR Module

- Uses **Tesseract OCR**
- Supports scanned images and PDFs
- Converts PDFs to images before OCR
- No grammar correction (intentional design)
- Preserves raw extracted content

---

## ✍️ Summarization Strategy

Summarization behavior depends on the detected resource type:

| Resource Type | Summary Style |
|--------------|--------------|
| Newspaper | Short and factual |
| Magazine | Medium-length descriptive |
| Book | Long and detailed |

Example prompts:


---

## 🔌 API Endpoints

### ✅ Health Check


Response:
```json
{
  "status": "ok",
  "message": "Simplified Document Processor API v2.0"
}
```

📤 Process Document
POST /process
Supported file types:
PDF
JPG / JPEG
PNG
TIFF
BMP

{
  "resource_type": "newspapers",
  "confidence": 0.92,
  "extracted_text": "...",
  "summaries": [
    "Summary 1",
    "Summary 2"
  ],
  "num_articles": 4,
  "text_length": 12540
}

How to Run the Server
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Install OCR Tools

Install Tesseract OCR

Install Poppler for PDF support

3️⃣ Start the Server
python main.py

👩‍🦯 Accessibility Use Case
Automatically understands document type
Produces clean summaries suitable for TTS
Enables independent learning for visually impaired users
Reduces cognitive load through structured summarization

🧑‍🎓 Research Contribution
This project contributes:
Resource-type–aware summarization
Lightweight fine-tuned language model using LoRA
OCR-driven historical document understanding
Accessibility-focused AI system design

🚀 Future Enhancements
Voice-based interaction
Multilingual OCR and summarization
Audio navigation by sections
Mobile and web application integration

