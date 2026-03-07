# Architecture & Features Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User's Browser                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          React Frontend (Vite)                        │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ Home Screen │ Lesson List │ Audio Player │     │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ Voice Recognition │ Audio Controls│ Routing    │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓ HTTP                              │
├─────────────────────────────────────────────────────────────┤
│                    Network (Internet)                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │     Python Backend (FastAPI on Uvicorn)             │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │         API Routes (Lessons, Audio)            │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │   Services (Audio Generation, Data Mgmt)      │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Data Layer (In-Memory / Future: Database)    │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Frontend Architecture

### Component Tree

```
App
├── HomeScreen
│   ├── Header
│   ├── VoiceControl
│   └── Grade Selection Buttons
├── LessonListScreen
│   ├── Header
│   ├── VoiceControl
│   └── Lesson Cards Grid
├── LessonSubsectionsScreen
│   ├── Header
│   ├── VoiceControl
│   └── Subsection Cards List
└── AudioPlayerScreen
    ├── Header
    ├── VoiceControl
    ├── AudioPlayer
    └── Subsection Details

Components
├── Header (Navigation)
├── VoiceControl (Voice Button)
├── LessonCard (Lesson Display)
├── SubsectionCard (Topic Display)
└── AudioPlayer (Audio Controls)

Hooks
└── useVoiceCommand (Voice Logic)

Services
├── api.js (API Calls)
└── voiceService.js (Voice Recognition)
```

### Data Flow

```
User Interaction
       ↓
Event Handler
       ↓
Voice Command / Button Click
       ↓
State Update (useState)
       ↓
API Call (axios)
       ↓
Backend Response
       ↓
Update State
       ↓
Re-render Component
       ↓
Display to User
```

## Backend Architecture

### Request Flow

```
HTTP Request
       ↓
FastAPI Router
       ↓
Route Handler (@router.get, @router.post)
       ↓
Business Logic (Services)
       ↓
Data Access (lesson_data.py)
       ↓
Response JSON
       ↓
HTTP Response
```

### Module Structure

```
app/
├── models/
│   └── models.py (Data Models - Lesson, Subsection, Grade)
├── routes/
│   └── lessons.py (API Endpoints)
├── services/
│   ├── lesson_data.py (Sample Data & Queries)
│   └── audio_service.py (TTS Audio Generation)
└── __init__.py

main.py (FastAPI App Setup)
```

## Voice-Based Navigation Flow

```
┌─────────────────────────────────┐
│   User Clicks Voice Button      │
└────────────────┬────────────────┘
                 ↓
      ┌──────────────────────┐
      │ Browser Permission   │
      │ Microphone Access    │
      └──────────┬───────────┘
                 ↓
      ┌──────────────────────────────┐
      │ Web Speech API Initialization │
      │ speech.start()               │
      └──────────┬───────────────────┘
                 ↓
      ┌──────────────────────────────┐
      │ Listening for Voice Input     │
      │ (UI shows animated button)    │
      └──────────┬───────────────────┘
                 ↓
      ┌──────────────────────────────┐
      │ Voice Recognition Processing  │
      │ onresult event triggered     │
      └──────────┬───────────────────┘
                 ↓
      ┌──────────────────────────────┐
      │ Command Matching              │
      │ "grade 10", "play", etc      │
      └──────────┬───────────────────┘
                 ↓
      ┌──────────────────────────────┐
      │ Callback Function Triggered   │
      │ Navigate or Control Audio     │
      └──────────┬───────────────────┘
                 ↓
      ┌──────────────────────────────┐
      │ Voice Feedback                │
      │ Speech synthesis response     │
      └──────────┬───────────────────┘
                 ↓
      ┌──────────────────────────────┐
      │ Listening Stopped             │
      │ speech.stop()                │
      └──────────────────────────────┘
```

## Audio Processing Pipeline

```
User Selects Subsection
         ↓
Frontend Requests Audio
         ↓
Backend Receives Request (POST /api/generate-audio)
         ↓
Check if Audio Exists
    ↙    ↖
YES      NO
 ↓        ↓
Return    Extract Content
Existing  ↓
URL       pyttsx3 TTS Processing
 ↓        ↓
         Save MP3 File
          ↓
        Return Audio URL
          ↓
Frontend Receives URL
         ↓
AudioPlayer Component Updates
         ↓
<audio> tag plays MP3
         ↓
User Controls: Play/Pause/Forward/Backward
```

## Data Models & Relationships

```
Grade (10 or 11)
    ↓
    ├─ Lesson 1
    │   ├─ Subsection 1
    │   │   └─ Content (Text for TTS)
    │   ├─ Subsection 2
    │   │   └─ Content (Text for TTS)
    │   └─ Subsection N
    │
    ├─ Lesson 2
    │   └─ ...
    │
    └─ Lesson N

Lesson Properties:
- id: "grade10_lesson1"
- title: "Ancient Civilizations"
- description: "Learn about..."
- grade: 10
- thumbnail: "🏛️"

Subsection Properties:
- id: "ancient_egypt"
- title: "Ancient Egypt"
- duration: 8 (minutes)
- description: "..."
- content: "Full lesson text..."
- audio_url: "optional generated audio"
```

## Voice Command Processing

### Command Recognition Pattern

```
User says: "play"
         ↓
Web Speech API transcript: "play"
         ↓
voiceService.processCommand("play")
         ↓
Find Matching Keyword
         ↓
commands = {
  'play': 'play',
  'pause': 'pause',
  'next': 'next',
  ...
}
         ↓
Match Found: 'play'
         ↓
Trigger Callback: commandCallbacks['play'](transcript)
         ↓
Execute Registered Function
         ↓
Update UI / Trigger Action
```

## State Management Pattern

```
Component State (useState):

HomeScreen:
- No local state (stateless)

LessonListScreen:
- lessons: Array of lessons
- loading: Boolean
- error: String
- isListening: Boolean (from hook)

LessonSubsectionsScreen:
- subsections: Array of subsections
- lesson: Object
- loading: Boolean
- error: String
- isListening: Boolean (from hook)

AudioPlayerScreen:
- subsection: Object
- audioUrl: String
- loading: Boolean
- generatingAudio: Boolean
- error: String
- isListening: Boolean (from hook)

Audio Player (Internal):
- currentTime: Number
- isAudioPlaying: Boolean
- audioRef: Reference to <audio> element
```

## Styling Architecture

### CSS Variables (root)
```css
--primary-blue: #4A90E2
--secondary-green: #7ED321
--light-bg: #F5F9FF
--white: #FFFFFF
--text-dark: #2C3E50
--text-light: #7F8C8D
--shadow: 0 2px 8px rgba(...)
--border-radius: 16px
--transition: all 0.3s ease
```

### Responsive Breakpoints
```css
Mobile: < 480px
Tablet: 480px - 768px
Desktop: > 768px
```

### Design System
```
Colors:
- Primary: Blue (#4A90E2)
- Secondary: Green (#7ED321)
- Backgrounds: Light Blue (#F5F9FF)
- Text: Dark Gray (#2C3E50)
- Accents: Soft Red (#FF6B6B)

Typography:
- Font Family: Poppins, Inter
- Headers: 700 weight
- Body: 400 weight
- Small: 12-14px
- Medium: 14-16px
- Large: 18-24px

Spacing:
- Small: 8px
- Medium: 16px
- Large: 20-24px
- XLarge: 40px

Shadows:
- Light: 0 2px 8px
- Dark: 0 4px 16px

Borders:
- Radius: 16px
- Transitions: 0.3s ease
```

## Performance Optimizations

### Frontend
- Code splitting with React Router
- Lazy loading of lessons
- Memoization of heavy components
- CSS animations (GPU-accelerated)
- Debouncing voice input
- Audio caching in browser

### Backend
- In-memory caching of lessons
- Audio file caching
- Efficient string matching
- Minimal JSON payloads
- Gzip compression ready

## Security Measures

### Frontend
- No sensitive data in localStorage
- XSS protection via React (automatic escaping)
- CSP headers ready
- HTTPS recommended for production

### Backend
- CORS validation
- Input sanitization
- Error handling (no stack traces exposed)
- Rate limiting ready
- Authentication hooks available

## Scalability Considerations

### Current Limitations
- In-memory data storage
- Local file audio storage
- No database
- Single server instance

### Future Improvements
- PostgreSQL/MongoDB database
- Redis caching layer
- Cloud storage (AWS S3, Azure Blob)
- Load balancing
- Microservices architecture
- GraphQL API option
- WebSocket for real-time features

## Browser Compatibility

### Voice Recognition Support
- Chrome 25+
- Edge 79+
- Safari 14.1+
- Firefox (limited)
- Opera

### Voice Synthesis Support
- All modern browsers
- Polyfill options available

### Minimum Requirements
- ES6 support
- Fetch API
- CSS Grid & Flexbox
- Web Audio API (for future)

## Testing Strategy

### Frontend Testing
```javascript
// Unit tests with Vitest/Jest
// Component tests with React Testing Library
// E2E tests with Cypress/Playwright

Examples:
- HomeScreen renders grade buttons
- LessonCard handles click events
- AudioPlayer controls work correctly
- Voice commands trigger navigation
```

### Backend Testing
```python
# Unit tests with pytest
# Integration tests with httpx

Examples:
- GET /api/grades returns correct data
- GET /api/lessons/{grade} validates grade
- Audio generation creates file
- Error handling for invalid requests
```

## Monitoring & Analytics

### Frontend Metrics
- Page load time
- Time to interactive
- Voice command success rate
- Audio playback errors
- User flow funnels

### Backend Metrics
- API response times
- Error rates
- Audio generation time
- Audio file cache hit rate
- Database query performance

---

**Architecture Version**: 1.0.0  
**Last Updated**: 2024  
**Technology Stack**: React 18, FastAPI, Python 3.10, Vite 5
