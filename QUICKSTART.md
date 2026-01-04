# Quick Start Guide

Follow these steps to get the application running quickly.

## Step 1: Backend Setup (First Time Only)

1. Open Command Prompt or PowerShell
2. Navigate to the backend folder:
   ```bash
   cd backend
   ```
3. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note:** This will take several minutes as it downloads large packages.

## Step 2: Start Backend Server

**Option A - Using the batch file (Easiest):**
- Double-click `start_backend.bat` in the backend folder

**Option B - Manual:**
```bash
cd backend
venv\Scripts\activate
python main.py
```

Wait for the message: "✅ All models loaded successfully"
The backend is now running at: http://localhost:8000

## Step 3: Frontend Setup (First Time Only)

1. Open a **NEW** Command Prompt or PowerShell window
2. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
3. **If upgrading from Create React App, clean old dependencies first:**
   ```bash
   # Windows PowerShell:
   Remove-Item -Recurse -Force node_modules
   Remove-Item -Force package-lock.json
   ```

4. Install dependencies:
   ```bash
   npm install
   ```
   
   **Note:** This may take a few minutes. Make sure you have Node.js 14.18+ (or 16+ recommended).

## Step 4: Start Frontend Server

**Option A - Using the batch file (Easiest):**
- Double-click `start_frontend.bat` in the frontend folder

**Option B - Manual:**
```bash
cd frontend
npm run dev
```

The browser should automatically open at: http://localhost:3000

## Step 5: Use the Application

1. Enter a history question
2. Enter a student's answer
3. Click "Evaluate Answer"
4. Wait for results (10-30 seconds depending on your hardware)

## Important Notes

- **First Run:** The backend will download the LLaMA model (~16GB) and Sentence-BERT model. This can take 10-30 minutes depending on your internet speed.
- **GPU vs CPU:** GPU is much faster. CPU will work but is slower.
- **Keep Both Servers Running:** You need both backend and frontend running simultaneously.
- **Model Path:** Make sure your model is at: `C:\Users\MSI\Desktop\Research\Model\ol_history_model\final_lora_model`

## Troubleshooting

**Backend won't start:**
- Check if port 8000 is already in use
- Verify Python version (need 3.8+)
- Check that model path exists

**Frontend won't start:**
- Check if Node.js is installed: `node --version`
- Delete `node_modules` folder and run `npm install` again

**"Models not loaded" error:**
- Wait a few minutes for models to load on first startup
- Check backend console for error messages
- Verify Hugging Face token is correct

## Stopping the Servers

- Backend: Press `Ctrl+C` in the backend terminal
- Frontend: Press `Ctrl+C` in the frontend terminal

