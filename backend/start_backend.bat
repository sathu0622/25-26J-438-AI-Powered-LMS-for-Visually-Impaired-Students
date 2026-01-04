@echo off
echo Starting O/L History Answer Evaluation Backend...
echo.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)
echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python main.py

