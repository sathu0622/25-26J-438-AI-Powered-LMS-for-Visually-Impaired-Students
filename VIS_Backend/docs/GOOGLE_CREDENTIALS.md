# How to Get Google Cloud Credentials for Text-to-Speech

This guide walks you through getting a **service account JSON key** so the VIS Backend can use Google Cloud Text-to-Speech (en-IN / en-GB).

---

## Prerequisites

- A **Google account** (Gmail).
- A web browser.

---

## Step 1: Open Google Cloud Console

1. Go to **[Google Cloud Console](https://console.cloud.google.com)**.
2. Sign in with your Google account.

---

## Step 2: Create or Select a Project

1. At the top of the page, click the **project dropdown** (it may say "Select a project" or show a project name).
2. Click **"New Project"**.
   - **Project name:** e.g. `vis-tts` or `my-lms-tts`.
   - **Organization:** leave default if you have one, or "No organization".
3. Click **"Create"** and wait until the project is created.
4. Make sure the new project is **selected** in the top bar (click the project dropdown and choose it).

---

## Step 3: Enable the Text-to-Speech API

1. In the left sidebar, go to **"APIs & Services"** → **"Library"**  
   (or open: [API Library](https://console.cloud.google.com/apis/library)).
2. In the search box, type **"Cloud Text-to-Speech API"**.
3. Click **"Cloud Text-to-Speech API"** in the results.
4. Click **"Enable"**.
5. Wait until it says the API is enabled.

---

## Step 4: Create a Service Account

1. In the left sidebar, go to **"APIs & Services"** → **"Credentials"**  
   (or open: [Credentials](https://console.cloud.google.com/apis/credentials)).
2. Click **"+ Create Credentials"** at the top.
3. Select **"Service account"**.
4. **Service account details:**
   - **Service account name:** e.g. `vis-tts-backend`.
   - **Service account ID:** will fill automatically (e.g. `vis-tts-backend`).
   - **Description (optional):** e.g. `Used by VIS Backend for TTS`.
5. Click **"Create and Continue"**.
6. **Grant access (optional):** you can skip this — click **"Continue"**.
7. Click **"Done"**.

---

## Step 5: Create and Download the JSON Key

1. On the **Credentials** page, find the **"Service accounts"** section.
2. Click the **service account** you just created (e.g. `vis-tts-backend@...`).
3. Open the **"Keys"** tab.
4. Click **"Add key"** → **"Create new key"**.
5. Choose **"JSON"**.
6. Click **"Create"**.
7. A JSON file will **download** to your computer (e.g. `vis-tts-xxxxx.json`).
   - **Keep this file private.** Do not commit it to Git or share it publicly.

---

## Step 6: Place the Key and Set the Environment Variable

1. **Move the JSON file** to a safe folder, for example:
   - Windows: `C:\keys\vis-tts-key.json`
   - Mac/Linux: `~/keys/vis-tts-key.json`
2. **Use the full path** in your backend `.env`:
   - Copy `VIS_Backend/.env.example` to `VIS_Backend/.env`.
   - Edit `.env` and set:
     ```env
     GOOGLE_APPLICATION_CREDENTIALS=C:\keys\vis-tts-key.json
     ```
     (On Mac/Linux use something like `/Users/yourname/keys/vis-tts-key.json`.)
3. **Restart the backend** so it picks up the new env:
   ```bash
   cd VIS_Backend
   npm start
   ```

---

## Step 7: Enable Billing (If Required)

- Google Cloud may ask you to **enable billing** (e.g. free trial or pay-as-you-go) to use the Text-to-Speech API.
- Text-to-Speech has a [free tier](https://cloud.google.com/text-to-speech/pricing); you may not be charged for light use.
- If you see "Billing account required" or "Quota exceeded", go to [Billing](https://console.cloud.google.com/billing) and add a billing account or check quotas.

---

## Quick Checklist

| Step | Action |
|------|--------|
| 1 | Open [Google Cloud Console](https://console.cloud.google.com) and sign in |
| 2 | Create or select a project |
| 3 | Enable **Cloud Text-to-Speech API** (APIs & Services → Library) |
| 4 | Create a **service account** (APIs & Services → Credentials → Create Credentials → Service account) |
| 5 | Create a **JSON key** for that service account (Keys tab → Add key → Create new key → JSON) |
| 6 | Set **GOOGLE_APPLICATION_CREDENTIALS** in `VIS_Backend/.env` to the **full path** of the JSON file |
| 7 | Restart the backend (`npm start`) |

---

## Troubleshooting

- **"Could not load the default credentials"**  
  - Check that `GOOGLE_APPLICATION_CREDENTIALS` in `.env` is the **full path** to the JSON file.
  - On Windows, use backslashes or forward slashes: `C:/keys/vis-tts-key.json` or `C:\keys\vis-tts-key.json`.

- **"Permission denied" or 403**  
  - Ensure **Cloud Text-to-Speech API** is enabled for the same project that owns the service account.

- **Backend returns 503 on `/api/tts`**  
  - Usually means the backend could not use Google credentials; re-check the path and that the JSON file exists.

- **.env not loading**  
  - Ensure the file is named exactly `.env` and is in the `VIS_Backend` folder (same folder as `package.json`).