# ── Base image: slim Python 3.10 (matches your local Python version)
FROM python:3.10-slim

# ── System dependencies needed by pydub / soundfile / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory inside container
WORKDIR /app

# ── Copy and install Python dependencies first (layer-cached)
COPY requirements.production.txt .
RUN pip install --no-cache-dir -r requirements.production.txt

# ── Download NLTK data needed at runtime
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Copy the rest of the application source
COPY . .

# ── Azure App Service / Container Apps injects PORT env var
ENV PORT=8004
ENV HOST=0.0.0.0

# ── Expose the port (documentation only; Azure uses $PORT)
EXPOSE 8004

# ── Start the FastAPI app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
