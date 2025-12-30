@echo off
echo Starting Document Processor Backend...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start the server
echo.
echo Starting FastAPI server...
echo Server will be available at http://localhost:8000
echo.
python main.py

pause

