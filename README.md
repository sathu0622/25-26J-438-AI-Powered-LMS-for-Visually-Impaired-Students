# AI History Teacher - Educational Learning Platform

A modern, responsive web application for Grade 10 & 11 History learning with AI-powered audio lessons and voice-based navigation.

## Project Structure

```
project-root/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── models/            # Data models
│   │   ├── routes/            # API endpoints
│   │   ├── services/          # Business logic (audio, lessons)
│   │   └── __init__.py
│   ├── main.py                # FastAPI application
│   └── requirements.txt        # Python dependencies
│
└── frontend/                   # React Vite frontend
    ├── src/
    │   ├── components/        # Reusable React components
    │   ├── screens/           # Screen/Page components
    │   ├── services/          # API and voice services
    │   ├── hooks/             # Custom React hooks
    │   ├── styles/            # Global styles
    │   ├── App.jsx            # Main app component
    │   └── main.jsx           # Entry point
    ├── package.json
    ├── vite.config.js
    └── index.html
```

## Features

### 🎓 Learning Features
- **Grade Selection**: Choose between Grade 10 or Grade 11 History
- **Lesson Organization**: Browse lessons organized by grade level
- **Subsection Topics**: Detailed topics within each lesson
- **AI Audio Lessons**: Text-to-speech powered audio lectures
- **Audio Player**: Play, pause, forward, backward controls with progress tracking

### 🎤 Voice-Based Navigation
- **Voice Recognition**: Use voice commands to navigate the app
- **Voice Synthesis**: App provides voice feedback and announcements
- **Supported Commands**:
  - "Grade 10" / "Grade 11" - Select grade
  - "Play" / "Pause" - Control audio playback
  - "Next" / "Previous" - Skip or go back
  - "Back" / "Home" - Navigate screens

### 📱 Design Features
- **Responsive Design**: Mobile-first, works on phones, tablets, and desktops
- **Modern UI**: Clean, educational aesthetic with soft colors
- **Accessibility**: Large buttons, clear typography for students
- **Animations**: Smooth transitions and engaging interactions

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Server**: Uvicorn
- **Audio**: pyttsx3 (Text-to-Speech)
- **APIs**: RESTful architecture

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Routing**: React Router v6
- **HTTP Client**: Axios
- **Web APIs**: Web Speech API (for voice)

## Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server**:
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`

## API Endpoints

### Grades
- `GET /api/grades` - Get available grades

### Lessons
- `GET /api/lessons/{grade}` - Get all lessons for a grade
- `GET /api/lesson/{lesson_id}` - Get specific lesson details
- `GET /api/lesson/{lesson_id}/subsections` - Get lesson subsections

### Subsections
- `GET /api/subsection/{lesson_id}/{subsection_id}` - Get subsection details
- `POST /api/generate-audio/{lesson_id}/{subsection_id}` - Generate audio for subsection

## Content Structure

### Grade 10 Lessons
1. **Ancient Civilizations**
   - Ancient Egypt
   - Ancient Greece
   - Roman Empire

2. **Medieval Period**
   - Feudalism
   - Knights and Castles

### Grade 11 Lessons
1. **Age of Exploration**
   - Motives for Exploration
   - Famous Explorers
   - Colonial Impact

2. **Industrial Revolution**
   - Start of Industrial Revolution
   - Industrial Society Changes
   - Global Spread of Industry

## Usage

1. **Start the backend server** (if not already running)
2. **Start the frontend development server**
3. **Open browser** to `http://localhost:5173`
4. **Select a grade** (Grade 10 or 11)
5. **Choose a lesson** to view available topics
6. **Select a subsection** to listen to audio lesson
7. **Use voice commands** or buttons to control playback

## Voice Commands

The app supports the following voice commands:
- **Navigation**: "grade ten", "grade eleven", "back", "home"
- **Audio Control**: "play", "pause", "next", "previous"

**Note**: Voice recognition requires:
- Modern browser with Web Speech API support
- Microphone permission
- Internet connection for optimal performance

## Customization

### Adding New Lessons
Edit `backend/app/services/lesson_data.py`:

```python
LESSON_DATA = {
    10: [
        {
            "id": "lesson_id",
            "title": "Lesson Title",
            "description": "Description",
            "grade": 10,
            "subsections": [
                {
                    "id": "subsection_id",
                    "title": "Topic Title",
                    "duration": 10,
                    "description": "Topic description",
                    "content": "Full lesson content..."
                }
            ]
        }
    ]
}
```

### Styling
- Global styles: `frontend/src/styles/global.css`
- Component styles: `frontend/src/components/styles.css`
- Screen styles: `frontend/src/screens/styles.css`
- Audio player: `frontend/src/components/AudioPlayer.css`

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari 14+, Chrome Mobile)

**Voice Recognition Support**: Chrome, Edge, Safari (requires explicit permission)

## Performance Optimization

- Lazy loading of lessons and audio
- Efficient state management
- Optimized CSS with CSS-in-JS
- Responsive image handling
- Audio caching

## Accessibility

- ARIA labels for voice commands
- High contrast colors
- Large touch targets for mobile
- Clear typography (Poppins font)
- Semantic HTML structure

## Future Enhancements

- [ ] User authentication and progress tracking
- [ ] More history lessons and grades
- [ ] Quiz and assessment features
- [ ] Offline mode with service workers
- [ ] Multiple language support
- [ ] Advanced analytics dashboard
- [ ] Teacher management portal
- [ ] Personalized learning paths

## Troubleshooting

### Backend Issues
- **Port already in use**: Change port in `main.py` or kill process using port 8000
- **Import errors**: Ensure all dependencies in `requirements.txt` are installed
- **Audio generation fails**: Check if pyttsx3 is properly installed

### Frontend Issues
- **API connection error**: Ensure backend is running on `http://localhost:8000`
- **Voice recognition not working**: Check browser compatibility and microphone permissions
- **Styling issues**: Clear browser cache (Ctrl+Shift+Del)

## Environment Variables

Create `.env` files if needed:

**Backend (.env in backend/)**:
```
FASTAPI_ENV=development
API_HOST=0.0.0.0
API_PORT=8000
```

**Frontend (.env in frontend/)**:
```
VITE_API_URL=http://localhost:8000/api
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues, questions, or suggestions, please create an issue in the repository.

---

**Happy Learning! 📚🎧**
This is using pre-built models + rule-based audio processing