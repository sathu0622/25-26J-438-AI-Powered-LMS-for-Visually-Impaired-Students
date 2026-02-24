# Project Structure - Clean Architecture

## Complete File Structure

```
VIS_Frontend/
├── src/
│   └── app/
│       ├── App.tsx ........................ Main entry point (223 lines - CLEAN!)
│       │
│       ├── components/
│       │   ├── document/ ................. Document module
│       │   │   ├── DocumentModule.tsx .... Container (orchestration)
│       │   │   ├── DocumentUpload.tsx .... UI component (presentational)
│       │   │   ├── DocumentProcessing.tsx  UI component (presentational)
│       │   │   ├── DocumentSummary.tsx .. UI component (presentational)
│       │   │   └── DocumentQA.tsx ....... UI component (presentational)
│       │   │
│       │   ├── braille/ ................. Braille module (can be refactored)
│       │   │   ├── BrailleUpload.tsx
│       │   │   └── BrailleEvaluation.tsx
│       │   │
│       │   ├── quiz/ .................... Quiz module (can be refactored)
│       │   │   ├── QuizStart.tsx
│       │   │   ├── QuizQuestion.tsx
│       │   │   └── QuizFeedback.tsx
│       │   │
│       │   ├── history/ ................ History module (can be refactored)
│       │   │   ├── HistoryHome.tsx
│       │   │   ├── LessonList.tsx
│       │   │   └── LessonPlayer.tsx
│       │   │
│       │   ├── ui/ ...................... Reusable UI components
│       │   │   ├── button.tsx
│       │   │   ├── card.tsx
│       │   │   ├── input.tsx
│       │   │   └── ... (shadcn components)
│       │   │
│       │   ├── Navigation.tsx
│       │   ├── HomePage.tsx
│       │   ├── VoiceCommandSystem.tsx
│       │   ├── VoiceButton.tsx
│       │   ├── AudioPlayer.tsx
│       │   └── MockSpeechIndicator.tsx
│       │
│       ├── services/ .................... API & Business Logic Layer
│       │   ├── api.ts ................... Base API configuration
│       │   └── documentService.ts ....... Document API calls
│       │
│       ├── hooks/ ....................... Custom Hooks & State Management
│       │   ├── useDocumentModule.ts ..... Document state management
│       │   ├── useSpeechRecognition.ts
│       │   └── useSpeechSynthesis.ts
│       │
│       ├── data/ ........................ Static data
│       │   ├── historyData.ts
│       │   └── quizData.ts
│       │
│       ├── utils/ ....................... Utility functions
│       │   ├── mockSpeech.ts
│       │   └── speech.ts
│       │
│       └── styles/ ...................... Stylesheet ....................... CSS stylesheets
│           ├── index.css
│           ├── fonts.css
│           ├── theme.css
│           └── tailwind.css
│
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
├── postcss.config.mjs
│
├── ARCHITECTURE.md ..................... Architecture documentation
└── REFACTORING_NOTES.md ............... This refactoring summary
```

## Layer Breakdown

### 📍 PRESENTATION LAYER (components/)
**Responsibility:** Render UI, respond to user input via callbacks

```
DocumentUpload.tsx
├── Input: onUpload callback
├── Output: User selects file
└── Logic: Only UI rendering

DocumentSummary.tsx
├── Input: summary text, callbacks
├── Output: User clicks buttons
└── Logic: Only UI rendering + keyboard shortcuts

DocumentQA.tsx
├── Input: mode, callbacks
├── Output: User submits question
└── Logic: Only voice/text input, no API calls

DocumentProcessing.tsx
├── Input: fileName
├── Output: Loading animation
└── Logic: None (pure UI)
```

### 🎯 CONTAINER LAYER (components/document/)
**Responsibility:** Orchestrate child components, connect logic to UI

```
DocumentModule.tsx (SMART COMPONENT)
├── Uses: useDocumentModule() hook
├── Manages: State from hook
├── Renders: All 5 document components above
├── Passes: Handlers to children
└── Logic: Orchestration only
```

### 🧠 LOGIC LAYER (hooks/)
**Responsibility:** State management, business logic, error handling

```
useDocumentModule.ts (CUSTOM HOOK)
├── State:
│   ├── Current screen
│   ├── Uploaded file
│   ├── Document result
│   ├── Summary text
│   ├── Error messages
│   └── Q&A mode
├── Handlers:
│   ├── handleUpload()
│   ├── handleSelectArticle()
│   ├── handleStartQA()
│   └── reset()
└── Uses: documentService for API calls
```

### 🔌 SERVICE LAYER (services/)
**Responsibility:** API calls, error handling, data transformation

```
api.ts (BASE API)
├── Single API_BASE_URL
├── Reusable methods:
│   ├── request(endpoint, options)
│   ├── post(endpoint, data)
│   ├── postForm(endpoint, data)
│   └── postFormData(endpoint, formData)
└── Error handling for all calls

documentService.ts (DOMAIN-SPECIFIC)
├── uploadDocument(file)
├── summarizeArticle(docId, articleId)
└── askQuestion(docId, articleId, question)
```

## Data Flow Diagram

### User uploads a document:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. USER INTERACTION (Presentation Layer)                            │
│    DocumentUpload.tsx                                               │
│    └─ User selects file → calls onUpload(file)                    │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. ORCHESTRATION (Container Layer)                                   │
│    DocumentModule.tsx                                               │
│    └─ Receives onUpload → calls handleUpload() from hook          │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. LOGIC & STATE (Logic Layer)                                       │
│    useDocumentModule.ts                                             │
│    ├─ Update state: screen = 'processing'                          │
│    ├─ Update state: isLoading = true                               │
│    └─ Call: documentService.uploadDocument(file)                   │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. API SERVICE (Service Layer)                                       │
│    documentService.ts                                               │
│    └─ Call: api.postFormData('/process', formData)                 │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. BASE API (API Layer)                                              │
│    api.ts                                                           │
│    ├─ Build URL: http://localhost:8000 + '/process'               │
│    ├─ POST formData                                                │
│    └─ Handle errors & return response                              │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
    [BACKEND API]
    (Processes document)
    └─ Returns: { document_id, summaries, article_list }
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Back to useDocumentModule.ts                                         │
│ ├─ Update state: documentResult = response                         │
│ ├─ Update state: documentSummary = response.summaries[0]           │
│ ├─ Update state: screen = 'summary'                                │
│ └─ Update state: isLoading = false                                 │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ DocumentModule.tsx re-renders with new state                         │
│ └─ Renders DocumentSummary with summary text                       │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. UI UPDATE (Presentation Layer)                                    │
│    DocumentSummary.tsx displays summary                             │
│    ✓ User sees the results!                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Comparison: Before vs After

### Before (Monolithic)
```
App.tsx (412 lines)
├── State: documentScreen, uploadedFile, documentResult, documentSummary,
│          qaMode, documentError, selectedArticleId, isDocumentLoading
├── Handler: handleDocumentUpload() - 50 lines with API call
├── Handler: handleArticleSelect() - 40 lines with API call
├── Handler: handleAskQuestion() - simple pass-through
├── Handler: handleBackToSummary() - simple pass-through
└── JSX: 40+ lines of conditional rendering with props

DocumentQA.tsx (300+ lines)
├── API_URL constant
├── Direct fetch to /ask-question
├── Error handling inline
├── State management mixed with rendering
└── Hard to test because of API coupling
```

**Problems:**
- ❌ App.tsx is god object (does everything)
- ❌ API logic scattered in components
- ❌ Hard to reuse logic
- ❌ Hard to test
- ❌ API URL duplicated in multiple places
- ❌ No clear pattern to follow

### After (Layered Architecture)
```
App.tsx (223 lines) - ENTRY POINT
├── Navigation logic only
├── Module switching
└── Renders DocumentModule

DocumentModule.tsx - ORCHESTRATOR
├── Uses useDocumentModule hook
├── No state management
├── No API logic
├── Just renders children + passes handlers

useDocumentModule.ts - STATE MANAGER
├── All document state
├── All handlers calling documentService
├── Clear interfaces
└── Easy to test

documentService.ts - API INTERFACE
├── uploadDocument()
├── summarizeArticle()
├── askQuestion()
└── No component logic

api.ts - HTTP CLIENT
├── Centralized config
├── Reusable methods
└── Error handling

DocumentUpload.tsx - PRESENTATION
├── Pure UI component
├── Gets onUpload callback
└── No logic

DocumentSummary.tsx - PRESENTATION
├── Pure UI component
├── Gets summary + callbacks
└── No API calls

DocumentQA.tsx - PRESENTATION
├── Pure UI component
├── Uses documentService
└── Easy to test with mocked service
```

**Benefits:**
- ✅ Clear separation of concerns
- ✅ Easy to test each layer
- ✅ Easy to modify API endpoints
- ✅ Reusable components
- ✅ Consistent patterns
- ✅ Professional architecture
- ✅ Scalable to larger projects

## 🚀 How to Use This Structure

### Adding a new feature:

1. **Backend returns new data** 
   → Update `documentService.ts` return type

2. **Need to call new endpoint**
   → Add method to `documentService.ts`

3. **Need to show new UI**
   → Create new presentation component

4. **Need to manage new state**
   → Update `useDocumentModule.ts`

5. **Need to orchestrate new flow**
   → Update `DocumentModule.tsx` rendering

### No need to touch App.tsx ever!

The Document module is completely self-contained. You can develop it independently, test it independently, and maintain it independently.

## Summary

This architecture provides:
- **Clear layers:** Presentation → Container → Logic → Service → API
- **Testability:** Mock at any layer
- **Maintainability:** Find code easily, understand responsibility quickly
- **Scalability:** Add features by following patterns
- **Professional:** Enterprise-grade code organization

Perfect for team projects, large codebases, and long-term maintenance! 🎉
