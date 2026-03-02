# Text-to-Speech (TTS) API Contract

The frontend uses a **central TTS service** that can use Google Cloud Text-to-Speech when the backend provides a TTS endpoint. This document describes the contract for the backend.

## When Google TTS is used

- Set `VITE_USE_GOOGLE_TTS=true` in the frontend environment.
- The frontend will call `POST {VITE_API_URL_VOICE}/api/tts` for speech when this is enabled (default: `http://localhost:5000`).

## Endpoint: `POST /api/tts`

### Request

- **Headers:** `Content-Type: application/json`
- **Body (JSON):**
  - `text` (optional): Plain text to speak. Use when `ssml` is not provided.
  - `ssml` (optional): SSML string (e.g. from [Google Cloud TTS SSML](https://cloud.google.com/text-to-speech/docs/ssml)). Used for tuning pronunciation, speed, and emphasis.
  - `lang` (optional): Language code. Frontend sends `en-IN` or `en-GB`. Default: `en-IN`.

Example:

```json
{
  "ssml": "<speak><prosody rate=\"95%\">Hello, welcome to the app.</prosody></speak>",
  "lang": "en-IN"
}
```

### Response

- **Status:** `200 OK`
- **Body (JSON):**
  - `audio_base64` (or `audioBase64`): Base64-encoded audio (e.g. MP3).
  - `content_type` (or `contentType`): MIME type, e.g. `audio/mp3`.

Example:

```json
{
  "audio_base64": "...",
  "content_type": "audio/mp3"
}
```

### Backend implementation notes

- Use [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech/docs) with a service account or API key (keep credentials on the backend only).
- Prefer **en-IN** or **en-GB** voices for natural speech for visually impaired users.
- If the endpoint is not implemented (e.g. 404/501), the frontend falls back to the browser Speech API or mock TTS.
