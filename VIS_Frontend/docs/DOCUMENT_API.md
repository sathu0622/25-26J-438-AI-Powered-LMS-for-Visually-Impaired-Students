# Document API Contract (Python Backend)

The frontend calls your **Python backend** at `VITE_API_URL` (default `http://localhost:8000`) for document upload, summarization, and Q&A.

## 404 with `x-powered-by: Express`?

If the response is **404** and the response headers include **`x-powered-by: Express`**, the request is hitting a **Node/Express** app, not your Python (Uvicorn) backend. So something else is using port 8000.

**Fix:** Free port 8000 and run only your Python backend there.

- **Windows (PowerShell):**  
  ```powershell
  # See what is using port 8000
  Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object OwningProcess
  # Then kill that process (replace PID with the number from above)
  Stop-Process -Id PID -Force
  ```
- Then start your Python backend: `python main.py` (from your Backend folder). It should bind to 8000 and show `Uvicorn running on http://0.0.0.0:8000`.  
- Do **not** run any other server (e.g. another Node/Express app) on port 8000.

## 404 on `/process`? (wrong path)

If you see **404 (Not Found)** on `:8000/process`, either:

1. **Your backend uses a path prefix** (e.g. `/api/process` instead of `/process`).  
   In `.env` set:
   ```env
   VITE_API_DOCUMENT_PREFIX=/api
   ```
   Then the frontend will call `/api/process`, `/api/summarize-article`, `/api/ask-question`.

2. **The route does not exist yet.** Your Python backend must expose these endpoints (with or without a prefix):

## Required endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST   | `{prefix}/process`         | Upload and process a document (FormData with `file`). |
| POST   | `{prefix}/summarize-article` | Get summary for one article (form: `document_id`, `article_id`). |
| POST   | `{prefix}/ask-question`    | Ask a question (JSON body: `document_id`, `article_id`, `question`, etc.). |

- **Prefix** = value of `VITE_API_DOCUMENT_PREFIX` (default `""`, so paths are `/process`, `/summarize-article`, `/ask-question`).

## Request/response shapes

- **POST /process**  
  - Body: `FormData` with key `file` (PDF or image).  
  - Response JSON: `{ document_id: string, summaries?: [{ summary: string }], article_list?: ArticleInfo[] }`.

- **POST /summarize-article**  
  - Body: form or JSON `document_id`, `article_id`.  
  - Response: `{ summary: string, article_heading?: string }`.

- **POST /ask-question**  
  - Body: JSON `document_id`, `article_id`, `question`, optional `max_answer_len`, `score_threshold`.  
  - Response: `{ answer: string, confidence?: number, article_heading?: string, context_preview?: string }`.

After adding the prefix or implementing these routes, restart the frontend dev server so env changes apply.
