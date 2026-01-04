from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(
    title="AI History Teacher API",
    description="Backend for AI-powered history learning app",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routes
from app.routes.lessons import router as lessons_router

app.include_router(lessons_router, prefix="/api", tags=["lessons"])

# Serve audio files
audio_dir = os.path.join(os.path.dirname(__file__), "..", "audio_files")
if os.path.exists(audio_dir):
    app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

@app.get("/")
async def root():
    """API Health Check"""
    return {
        "message": "AI History Teacher API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
