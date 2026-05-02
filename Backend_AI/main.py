from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import socket
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.routes.lessons import router as lessons_router
from app.routes.chapters import router as chapters_router
from app.routes.audio import router as audio_router

app = FastAPI(title="AI History Teacher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(lessons_router)
app.include_router(chapters_router)
app.include_router(audio_router)


@app.get("/")
def health_check():
    return {"status": "ok"}


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _find_available_port(host: str, starting_port: int, max_attempts: int = 20) -> int:
    port = starting_port
    for _ in range(max_attempts):
        if _is_port_available(host, port):
            return port
        port += 1
    return starting_port


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    requested_port = int(os.getenv("PORT", "8003"))
    reload_enabled = os.getenv("RELOAD", "false").lower() == "true"

    selected_port = _find_available_port(host, requested_port)
    if selected_port != requested_port:
        print(f"⚠️ Port {requested_port} is unavailable. Starting on {selected_port} instead.")

    uvicorn.run("main:app", host=host, port=selected_port, reload=reload_enabled)
