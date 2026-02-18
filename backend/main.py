from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routes.lessons import router as lessons_router
from app.routes.chapters import router as chapters_router
from app.routes.audio import router as audio_router

app = FastAPI(title="AI History Teacher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(lessons_router)
app.include_router(chapters_router)
app.include_router(audio_router)


@app.get("/")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
