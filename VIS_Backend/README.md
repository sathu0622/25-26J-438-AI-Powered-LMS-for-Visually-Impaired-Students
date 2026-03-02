# VIS Backend (Node + Express)

Backend for the AI-Powered LMS for Visually Impaired Students. Provides the **Text-to-Speech (TTS)** API using Google Cloud Text-to-Speech (en-IN / en-GB).

## Requirements

- Node.js 18+
- A Google Cloud project with **Cloud Text-to-Speech API** enabled and a **service account key** (JSON).

## Setup

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Google Cloud credentials** — **→ Full steps: [docs/GOOGLE_CREDENTIALS.md](docs/GOOGLE_CREDENTIALS.md)**

   - Create a project (or use an existing one) at [Google Cloud Console](https://console.cloud.google.com).
   - Enable **Cloud Text-to-Speech API**: APIs & Services → Enable APIs → search “Text-to-Speech” → Enable.
   - Create a service account: IAM & Admin → Service Accounts → Create. Grant no extra roles (TTS only needs the API).
   - Create a JSON key for that service account and download it.

3. **Environment**

   - Copy `.env.example` to `.env`.
   - Set `GOOGLE_APPLICATION_CREDENTIALS` to the **absolute path** of your service account JSON file, e.g.:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-key.json
     ```
   - Optionally set `PORT` (default is `8000`).

## Run

```bash
npm start
```

Development with auto-restart:

```bash
npm run dev
```

Server runs at `http://localhost:8000` (or your `PORT`).

## API

### Health

- **GET /health**  
  Returns `{ "status": "ok", "service": "vis-backend" }`.

### Text-to-Speech

- **POST /api/tts**

  - **Body (JSON):**
    - `text` (optional): plain text to speak.
    - `ssml` (optional): SSML string for pronunciation/speed/emphasis.
    - `lang` (optional): `en-IN` (default) or `en-GB`.
  - **Response (200):**
    - `audio_base64`: base64-encoded MP3.
    - `content_type`: `"audio/mp3"`.

  If `GOOGLE_APPLICATION_CREDENTIALS` is not set or invalid, the API returns **503** and the frontend falls back to browser or mock TTS.

## Frontend

In the frontend (VIS_Frontend):

1. Set `VITE_API_URL=http://localhost:8000` (or your backend URL).
2. Set `VITE_USE_GOOGLE_TTS=true` to use this backend for speech.
3. Run the frontend; all TTS will go through this backend when enabled.
