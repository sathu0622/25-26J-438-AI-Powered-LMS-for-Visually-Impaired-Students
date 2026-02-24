/**
 * PROJECT ARCHITECTURE DOCUMENTATION
 * 
 * NEW CLEAN ARCHITECTURE STRUCTURE
 * =================================
 * 
 * This document outlines the refactored project structure following clean architecture principles.
 * 
 * ## Directory Structure
 * 
 * src/app/
 * ├── components/
 * │   ├── document/
 * │   │   ├── DocumentModule.tsx       (Container - Orchestrates document workflow)
 * │   │   ├── DocumentUpload.tsx       (Presentation - File upload UI)
 * │   │   ├── DocumentProcessing.tsx   (Presentation - Loading indicator UI)
 * │   │   ├── DocumentSummary.tsx      (Presentation - Summary display UI)
 * │   │   └── DocumentQA.tsx           (Presentation - Q&A interface UI)
 * │   ├── braille/
 * │   ├── quiz/
 * │   ├── history/
 * │   ├── ui/                          (Reusable UI components)
 * │   └── Navigation.tsx, HomePage.tsx, etc.
 * │
 * ├── services/
 * │   ├── api.ts                       (Base API configuration & utilities)
 * │   └── documentService.ts           (Document API calls)
 * │
 * ├── hooks/
 * │   ├── useDocumentModule.ts         (Document state management)
 * │   ├── useSpeechRecognition.ts
 * │   └── useSpeechSynthesis.ts
 * │
 * ├── data/
 * ├── utils/
 * ├── styles/
 * └── App.tsx                          (Main entry - Only imports & renders modules)
 * 
 * 
 * ## ARCHITECTURAL PRINCIPLES
 * 
 * ### 1. Separation of Concerns
 * ===============================
 * - PRESENTATION LAYER: React components handle only UI rendering (DocumentUpload, DocumentSummary, etc.)
 * - BUSINESS LOGIC LAYER: Custom hooks manage state & orchestration (useDocumentModule)
 * - CONTAINER LAYER: Module containers connect logic to UI (DocumentModule)
 * - SERVICE LAYER: API calls are isolated in services (documentService, api)
 * 
 * ### 2. API Layer (services/)
 * =============================
 * 
 * **api.ts** - Base API configuration
 * - Single source of truth for API URL
 * - Reusable fetch methods (request, post, postForm, postFormData)
 * - Centralized error handling
 * - Type-safe API utilities
 * 
 * **documentService.ts** - Domain-specific API calls
 * - uploadDocument()      → POST /process
 * - summarizeArticle()    → POST /summarize-article
 * - askQuestion()         → POST /ask-question
 * - All exported as SERVICES, not raw fetch calls
 * 
 * Benefits:
 * ✓ Easy to mock for testing
 * ✓ Single point to modify API endpoints
 * ✓ Clear error handling
 * ✓ Type-safe request/response
 * 
 * ### 3. State Management (hooks/)
 * ==================================
 * 
 * **useDocumentModule.ts** - Complete document module state
 * - Manages all document-related state
 * - Exports state and handlers
 * - Uses documentService internally
 * - Components never call APIs directly
 * 
 * State includes:
 * - Current screen (upload/processing/summary/qa)
 * - Upload file & loading status
 * - Document result & summary
 * - Q&A mode selection
 * - Error handling
 * 
 * Benefits:
 * ✓ Centralized state logic
 * ✓ Reusable across components
 * ✓ Easy to test
 * ✓ Clear data flow
 * 
 * ### 4. Component Structure
 * ============================
 * 
 * **Presentation Components** (Pure UI)
 * - DocumentUpload.tsx    → Receives onUpload callback
 * - DocumentProcessing.tsx → Receives fileName prop
 * - DocumentSummary.tsx   → Receives summary, callbacks
 * - DocumentQA.tsx        → Receives documentId, articleId, callbacks
 * 
 * RULE: No API calls, no state logic - only props and callbacks
 * 
 * **Container Component** (Orchestration)
 * - DocumentModule.tsx    → Uses useDocumentModule hook
 *                         → Renders child components
 *                         → Passes handlers to children
 *                         → NO HTML rendering except structure
 * 
 * RULE: Orchestrates component tree, connects logic to UI
 * 
 * ### 5. App.tsx (Main Entry)
 * =============================
 * 
 * BEFORE (Anti-pattern):
 * ❌ Contained all state for all modules
 * ❌ Had API calls mixed in
 * ❌ Had 20+ handlers defined
 * ❌ 400+ lines of code
 * 
 * AFTER (Clean):
 * ✓ Only imports modules & components
 * ✓ Only handles navigation between modules
 * ✓ NO API calls in App.tsx
 * ✓ NO business logic in App.tsx
 * ✓ ~220 lines of code
 * 
 * App.tsx responsibilities:
 * 1. Render VoiceCommandSystem (global)
 * 2. Handle module switching (home/document/braille/quiz/history)
 * 3. Render appropriate module component
 * 4. Render Navigation (global)
 * 
 * Each module handles its own internal state/logic.
 * 
 * 
 * ## REQUEST/RESPONSE FLOW
 * 
 * User uploads document:
 * ┌──────────────────┐
 * │ DocumentUpload   │ (Presentation)
 * │  <onUpload>      │
 * └────────┬─────────┘
 *          │ (callback)
 *          ▼
 * ┌──────────────────────┐
 * │ DocumentModule       │ (Container)
 * │ (handleUpload)       │
 * └────────┬─────────────┘
 *          │ (calls)
 *          ▼
 * ┌──────────────────────┐
 * │ useDocumentModule    │ (Hook/Logic)
 * │ (setState loading)   │
 * └────────┬─────────────┘
 *          │ (calls)
 *          ▼
 * ┌──────────────────────┐
 * │ documentService      │ (Service)
 * │ .uploadDocument()    │
 * └────────┬─────────────┘
 *          │ (calls)
 *          ▼
 * ┌──────────────────────┐
 * │ api.postFormData()   │ (API Layer)
 * │ → POST /process      │
 * └────────┬─────────────┘
 *          │
 *          ▼
 *     [Backend API]
 *
 * 
 * ## BENEFITS OF THIS STRUCTURE
 * 
 * 1. TESTABILITY
 *    - Services can be mocked easily
 *    - Components can be tested in isolation
 *    - State logic is independent
 * 
 * 2. MAINTAINABILITY
 *    - Clear responsibility of each layer
 *    - Easy to find where to make changes
 *    - API changes only affect service layer
 * 
 * 3. REUSABILITY
 *    - Services can be used by other components
 *    - Custom hooks can be used in multiple places
 *    - UI components are completely reusable
 * 
 * 4. SCALABILITY
 *    - Easy to add new features
 *    - Easy to refactor without breaking others
 *    - Clear patterns to follow
 * 
 * 5. CODE ORGANIZATION
 *    - No giant App.tsx file
 *    - Each module is self-contained
 *    - Clear separation of concerns
 * 
 * 
 * ## HOW TO ADD NEW FEATURES
 * 
 * Example: Adding a new document export feature
 * 
 * 1. CREATE SERVICE METHOD
 *    ```tsx
 *    // documentService.ts
 *    async exportDocument(documentId: string, format: 'pdf' | 'docx') {
 *      return api.post(`/export-document`, { document_id: documentId, format });
 *    }
 *    ```
 * 
 * 2. ADD HOOK HANDLER
 *    ```tsx
 *    // useDocumentModule.ts
 *    const handleExport = useCallback(async (format) => {
 *      try {
 *        const result = await documentService.exportDocument(
 *          state.documentResult.document_id,
 *          format
 *        );
 *        // Handle success
 *      } catch (err) {
 *        // Handle error
 *      }
 *    }, [state.documentResult]);
 *    ```
 * 
 * 3. CREATE UI COMPONENT
 *    ```tsx
 *    // DocumentExport.tsx
 *    export const DocumentExport = ({ onExport }) => {
 *      return (
 *        <button onClick={() => onExport('pdf')}>Export as PDF</button>
 *      );
 *    };
 *    ```
 * 
 * 4. ADD TO CONTAINER
 *    ```tsx
 *    // DocumentModule.tsx
 *    {screen === 'summary' && (
 *      <>
 *        <DocumentSummary ... />
 *        <DocumentExport onExport={handleExport} />
 *      </>
 *    )}
 *    ```
 * 
 * Done! Clean, organized, and testable.
 * 
 * 
 * ## NEXT STEPS (OPTIONAL)
 * 
 * The following modules can be refactored similarly:
 * - Braille module (BrailleUpload, BrailleEvaluation)
 * - Quiz module (QuizStart, QuizQuestion, QuizFeedback)
 * - History module (HistoryHome, LessonList, LessonPlayer)
 * 
 * Each should follow the same pattern:
 * - Create service if it needs API calls
 * - Create custom hook for state management
 * - Create container component to orchestrate
 * - Keep presentation components pure
 */
