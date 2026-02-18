# AI History Teacher Backend

FastAPI backend serving Grade 10 and Grade 11 history curricula.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Datasets

Place your CSV datasets in the `data/` folder:
- `data/grade10_dataset.csv` - Grade 10 history content
- `data/grade11_dataset.csv` - Grade 11 history content

**Expected CSV columns:**
- `chapter` - Chapter name (e.g., "1. Prehistoric Era")
- `Grade/Topic` - Topic/lesson name
- `original_text` - Full original content
- `simplified_text` - Simplified content (for TTS)
- `narrative_text` - Narrative format
- `emotion` - Emotion tone (e.g., "wonder", "respect")
- `sound_effects` - Associated sound effects

### 3. Add Your Trained Model

Copy your trained TTS model to:
- `app/models/AI_History_Teacher_System.pth`

### 4. Run the Server

```bash
python main.py
```

Server starts at: `http://localhost:8000`

## API Endpoints

### Get Available Grades
```
GET /api/grades
```

### Get Chapters for a Grade
```
GET /api/chapters/{grade}
```
Example: `GET /api/chapters/10`

### Get Topics in a Chapter
```
GET /api/chapters/{grade}/{chapter_idx}/topics
```
Example: `GET /api/chapters/10/0/topics`

### Get Specific Topic Content
```
GET /api/chapters/{grade}/{chapter_idx}/topics/{topic_idx}
```

## Project Structure

```
backend/
├── main.py                     # FastAPI app entry point
├── requirements.txt            # Python dependencies
├── data/                       # CSV datasets (add your files here)
│   ├── grade10_dataset.csv
│   └── grade11_dataset.csv
└── app/
    ├── __init__.py
    ├── models/                 # Data models and trained .pth file
    │   ├── __init__.py
    │   ├── models.py
    │   └── AI_History_Teacher_System.pth
    ├── routes/                 # API endpoints
    │   ├── __init__.py
    │   ├── lessons.py
    │   └── chapters.py
    └── services/               # Business logic
        ├── __init__.py
        ├── lesson_data.py
        ├── chapter_service.py
        └── tts_service.py
```

## Development

### Enable Hot Reload
The server automatically reloads on file changes when run with `python main.py`.

### Access API Documentation
When server is running, visit: `http://localhost:8000/docs`

## Next Steps

1. **Add TTS Integration**: Update `app/services/tts_service.py` to load and use your `.pth` model
2. **Audio Generation**: Create endpoint to generate audio from topic content
3. **Caching**: Cache generated audio files to avoid regeneration
4. **Database**: Migrate from CSV to database (PostgreSQL, MongoDB)
