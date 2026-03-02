@echo off
echo Starting O/L History Answer Evaluation Frontend...
echo.
cd /d "%~dp0"
if not exist node_modules (
    echo Installing dependencies...
    call npm install
)
echo.
echo Starting Vite development server on http://localhost:3000
echo.
call npm run dev
