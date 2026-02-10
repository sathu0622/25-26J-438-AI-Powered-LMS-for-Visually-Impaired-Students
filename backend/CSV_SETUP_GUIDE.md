# CSV Data Setup Guide

## 📊 Overview

The AI History Teacher System now loads curriculum data from CSV files matching your Jupyter notebook structure!

## 📁 Required CSV Files

You need to upload these CSV files to the `backend/data/` folder:

1. **grade10_dataset.csv** - Grade 10 curriculum data
2. **grade11_dataset.csv** - Grade 11 curriculum data

## 📋 CSV File Structure

Your CSV files should have these columns (matching the notebook):

| Column | Description | Example |
|--------|-------------|---------|
| `chapter` | Chapter number and title | "1. Ancient Civilizations" |
| `Grade/Topic` | Topic name | "Mesopotamia: The Cradle of Civilization" |
| `emotion` | Emotional tone for TTS | "inspirational", "awe", "wonder" |
| `sound_effects` | Comma-separated sound effects | "chime, soft_background_music, ambient_ancient" |
| `simplified_text` | Full lesson content | "Mesopotamia, located between..." |

The system will automatically extract:
- **chapter_num**: From the chapter column using regex `r'(\d+)\.'`
- **chapter_title**: From the chapter column after the number
- **sound_effects_list**: By splitting sound_effects on commas

## 🎵 Sound Effect Files

Upload your sound effect MP3 files to the `backend/sounds/` folder:

Example sound files:
- `chime.mp3`
- `soft_background_music.mp3`
- `ambient_ancient.mp3`
- `desert_wind.mp3`
- `church_bells.mp3`
- `marketplace.mp3`
- etc.

## 🎭 Supported Emotions (from notebook)

The system supports 14 emotional tones:
1. inspirational
2. awe
3. vibrancy
4. harmony
5. wonder
6. reverence
7. justice
8. prosperity
9. warmth
10. hope
11. resilience
12. somber
13. respect
14. neutral

## 📂 Folder Structure

```
backend/
├── data/                    # ← Upload CSV files here
│   ├── grade10_dataset.csv
│   └── grade11_dataset.csv
├── sounds/                  # ← Upload sound effect files here
│   ├── chime.mp3
│   ├── soft_background_music.mp3
│   └── ...
└── audio_output/            # Generated TTS audio files
```

## 🚀 How to Upload Files

### Method 1: Drag & Drop (Recommended)
1. Open VS Code Explorer
2. Navigate to `backend/data/` folder
3. Drag your CSV files into the folder
4. Navigate to `backend/sounds/` folder
5. Drag your sound effect files into the folder

### Method 2: Manual Copy
1. Copy `grade10_dataset.csv` to `backend/data/`
2. Copy `grade11_dataset.csv` to `backend/data/`
3. Copy all sound effect MP3 files to `backend/sounds/`

## ✅ Verification Steps

After uploading the CSV files:

1. **Install pandas** (if not already installed):
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start the backend server**:
   ```bash
   python main.py
   ```

3. **Check the console output** - you should see:
   ```
   ============================================================
   🎓 AI History Teacher System - Loading Curriculum
   ============================================================
   📊 Loading Grade 10 data from: backend/data/grade10_dataset.csv
     ✓ Chapter 1: Ancient Civilizations (X topics)
     ✓ Chapter 2: ... (X topics)
   ✅ Successfully loaded curriculum from CSV files
   ```

4. **Test the API** - visit:
   - http://localhost:8000/chapters/10 - Should show Grade 10 chapters from CSV
   - http://localhost:8000/chapters/11 - Should show Grade 11 chapters from CSV

## 🔧 How It Works

The system uses the same preprocessing logic from your notebook:

```python
# Extract chapter number
df['chapter_num'] = df['chapter'].str.extract(r'(\d+)\.')

# Extract chapter title
df['chapter_title'] = df['chapter'].str.split('.').str[1].str.strip()

# Split sound effects
df['sound_effects_list'] = df['sound_effects'].str.split(', ')
```

Then groups topics by chapter number to create the hierarchy:
- Chapter 1: Ancient Civilizations
  - Topic 1: Mesopotamia
  - Topic 2: Egypt
  - etc.

## 📊 Example CSV Row

```csv
chapter,Grade/Topic,emotion,sound_effects,simplified_text
"1. Ancient Civilizations","Mesopotamia: The Cradle of Civilization","wonder","chime, soft_background_music, ambient_ancient","Mesopotamia, located between the Tigris and Euphrates rivers..."
```

## 🔄 Fallback Behavior

If CSV files are not found, the system will:
1. Try to load from `AI_History_Teacher_System.pth` model
2. If model not found, use predefined fallback curriculum
3. Console will show which data source is being used

## 🐛 Troubleshooting

**CSV not loading?**
- Check file names are exactly: `grade10_dataset.csv`, `grade11_dataset.csv`
- Verify encoding is `latin-1` (system handles this automatically)
- Check the console for error messages

**Chapters not showing correctly?**
- Verify `chapter` column format: "1. Chapter Title"
- Ensure chapter numbers are in format: `1.`, `2.`, etc.
- Check that `Grade/Topic` column has topic names

**Sound effects not working?**
- Verify sound file names match those in the CSV `sound_effects` column
- Ensure files are in `backend/sounds/` folder
- Check file extensions are `.mp3`, `.wav`, or `.ogg`

## 📝 Next Steps

1. ✅ Folders created: `data/`, `sounds/`, `audio_output/`
2. ✅ Code updated to load from CSV files
3. ✅ pandas dependency added to requirements.txt
4. ⏳ **Your turn:** Upload CSV and sound files
5. ⏳ Install dependencies: `pip install -r requirements.txt`
6. ⏳ Start backend and verify data loads correctly

Once files are uploaded, the Grade 10/11 chapters will display the actual curriculum from your CSV files! 🎉
