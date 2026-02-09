import { useState } from 'react';
import { Navigation } from './components/Navigation';
import { HomePage } from './components/HomePage';
import { VoiceCommandSystem } from './components/VoiceCommandSystem';

// Document Module
import { DocumentUpload } from './components/document/DocumentUpload';
import { DocumentProcessing } from './components/document/DocumentProcessing';
import { DocumentSummary } from './components/document/DocumentSummary';
import { DocumentQA } from './components/document/DocumentQA';

// Braille Module
import { BrailleUpload } from './components/braille/BrailleUpload';
import { BrailleEvaluation } from './components/braille/BrailleEvaluation';

// Quiz Module
import { QuizStart } from './components/quiz/QuizStart';
import { QuizQuestion } from './components/quiz/QuizQuestion';
import { QuizFeedback } from './components/quiz/QuizFeedback';

// History Module
import { HistoryHome } from './components/history/HistoryHome';
import { LessonList } from './components/history/LessonList';
import { LessonPlayer } from './components/history/LessonPlayer';

// Data
import { getQuestionsByTopic } from './data/quizData';

type Module = 'home' | 'document' | 'braille' | 'quiz' | 'history';

type DocumentScreen = 'upload' | 'processing' | 'summary' | 'qa';
type BrailleScreen = 'upload' | 'evaluation';
type QuizScreen = 'start' | 'question' | 'feedback';
type HistoryScreen = 'home' | 'lessons' | 'player';

interface Question {
  id: number;
  text: string;
  topic: string;
  expectedAnswer?: string;
  feedback?: string;
}

export default function App() {
  const [currentModule, setCurrentModule] = useState<Module>('home');

  // Document module state
  const [documentScreen, setDocumentScreen] = useState<DocumentScreen>('upload');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [documentSummary, setDocumentSummary] = useState('');
  const [qaMode, setQaMode] = useState<'voice' | 'text'>('voice');
  const [documentResult, setDocumentResult] = useState<any | null>(null);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const [selectedArticleId, setSelectedArticleId] = useState<string | null>(null);
  const [isDocumentLoading, setIsDocumentLoading] = useState(false);

  // Braille module state
  const [brailleScreen, setBrailleScreen] = useState<BrailleScreen>('upload');

  // Quiz module state
  const [quizScreen, setQuizScreen] = useState<QuizScreen>('start');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [selectedTopic, setSelectedTopic] = useState<string>('');
  const [quizQuestions, setQuizQuestions] = useState<Question[]>([]);

  // History module state
  const [historyScreen, setHistoryScreen] = useState<HistoryScreen>('home');
  const [selectedGrade, setSelectedGrade] = useState<number>(10);
  const [selectedLesson, setSelectedLesson] = useState<number>(1);

  const handleNavigate = (module: string) => {
    const target = module as Module;
    setCurrentModule(target);

    // Reset module states when navigating
    if (target === 'document') {
      setDocumentScreen('upload');
    } else if (target === 'braille') {
      setBrailleScreen('upload');
    } else if (target === 'quiz') {
      setQuizScreen('start');
    } else if (target === 'history') {
      setHistoryScreen('home');
    }
  };

  // Handle voice command navigation
  const handleVoiceNavigate = (route: string) => {
    const routeMap: Record<string, Module> = {
      'home': 'home',
      'document-upload': 'document',
      'braille': 'braille',
      'quiz': 'quiz',
      'history': 'history',
    };
    
    const module = routeMap[route];
    if (module) {
      handleNavigate(module);
    }
  };

  // Get current page identifier for voice system
  const getCurrentPage = (): string => {
    if (currentModule === 'document') return 'document-upload';
    return currentModule;
  };

  // Document Module Handlers
  const handleDocumentUpload = async (file: File) => {
    setUploadedFile(file);
    setDocumentResult(null);
    setDocumentError(null);
    setSelectedArticleId(null);
    setDocumentScreen('processing');
    setIsDocumentLoading(true);

    const apiUrl =
      (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiUrl}/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = 'Processing failed. Please try again.';
        try {
          const errorData = await response.json();
          if (errorData?.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // ignore JSON parse errors
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      setDocumentResult(data);

      // Set initial summary from backend response if available
      let initialSummary = '';
      if (Array.isArray(data.summaries) && data.summaries.length > 0) {
        initialSummary = data.summaries[0]?.summary || '';
      }
      setDocumentSummary(initialSummary);

      // Default selected article (first article or full document)
      const defaultArticleId =
        data.article_list?.[0]?.article_id || 'full_document';
      setSelectedArticleId(defaultArticleId);

      setDocumentScreen('summary');
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Unable to process document. Please try again.';
      setDocumentError(message);
      setDocumentScreen('upload');
    } finally {
      setIsDocumentLoading(false);
    }
  };

  const handleArticleSelect = async (articleId: string) => {
    if (!documentResult?.document_id) return;

    setSelectedArticleId(articleId);

    const apiUrl =
      (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${apiUrl}/summarize-article`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          document_id: documentResult.document_id,
          article_id: articleId,
        }),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to summarize selected article.';
        try {
          const errorData = await response.json();
          if (errorData?.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // ignore JSON parse errors
        }
        throw new Error(errorMessage);
      }

      const summaryData = await response.json();
      setDocumentSummary(summaryData.summary || '');

      // Optionally keep last summary metadata with the result
      setDocumentResult((prev: any) =>
        prev
          ? {
              ...prev,
              summaries: [summaryData],
            }
          : prev
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Failed to summarize selected article.';
      setDocumentError(message);
    }
  };

  const handleAskQuestion = (mode: 'voice' | 'text') => {
    setQaMode(mode);
    setDocumentScreen('qa');
  };

  const handleBackToSummary = () => {
    setDocumentScreen('summary');
  };

  // Braille Module Handlers
  const handleBrailleUpload = (file: File) => {
    // File is received, now proceed to evaluation
    setBrailleScreen('evaluation');
  };

  const handleBrailleBack = () => {
    setBrailleScreen('upload');
  };

  // Quiz Module Handlers
  const handleQuizStart = (topic: string) => {
    setSelectedTopic(topic);
    const questions = getQuestionsByTopic(topic);
    setQuizQuestions(questions);
    setCurrentQuestionIndex(0);
    setQuizScreen('question');
  };

  const handleQuizSubmit = (answer: string) => {
    setCurrentAnswer(answer);
    setQuizScreen('feedback');
  };

  const handleQuizNext = () => {
    if (currentQuestionIndex < quizQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setQuizScreen('question');
    } else {
      // Quiz complete
      setQuizScreen('start');
      setCurrentQuestionIndex(0);
    }
  };

  const handleQuizSkip = () => {
    handleQuizNext();
  };

  // History Module Handlers
  const handleSelectGrade = (grade: number) => {
    setSelectedGrade(grade);
    setHistoryScreen('lessons');
  };

  const handleSelectLesson = (lessonId: number) => {
    setSelectedLesson(lessonId);
    setHistoryScreen('player');
  };

  const handleHistoryBack = () => {
    if (historyScreen === 'player') {
      setHistoryScreen('lessons');
    } else if (historyScreen === 'lessons') {
      setHistoryScreen('home');
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Voice Command System - Global Accessibility Control */}
      <VoiceCommandSystem 
        onNavigate={handleVoiceNavigate} 
        currentPage={getCurrentPage()} 
      />

      {/* Main Content */}
      <main className="min-h-screen">
        {/* Home */}
        {currentModule === 'home' && <HomePage onNavigate={handleNavigate} />}

        {/* Document Module */}
        {currentModule === 'document' && (
          <>
            {documentError && (
              <div className="mx-auto max-w-2xl p-4">
                <div
                  className="rounded-lg border border-destructive bg-destructive/10 p-4"
                  role="alert"
                  aria-live="assertive"
                >
                  <p className="font-medium">Document processing error</p>
                  <p className="text-sm">{documentError}</p>
                </div>
              </div>
            )}

            {documentScreen === 'upload' && (
              <DocumentUpload onUpload={handleDocumentUpload} />
            )}
            {documentScreen === 'processing' && uploadedFile && (
              <DocumentProcessing fileName={uploadedFile.name} />
            )}
            {documentScreen === 'summary' && (
              <DocumentSummary
                summary={documentSummary}
                onAskQuestion={handleAskQuestion}
                articles={documentResult?.article_list}
                selectedArticleId={selectedArticleId}
                onSelectArticle={handleArticleSelect}
              />
            )}
            {documentScreen === 'qa' && (
              <DocumentQA
                mode={qaMode}
                onBack={handleBackToSummary}
                documentId={documentResult?.document_id ?? ''}
                articleId={selectedArticleId ?? null}
                articleHeading={documentResult?.article_list?.find(
                  (article: any) => article.article_id === selectedArticleId
                )?.heading}
              />
            )}
          </>
        )}

        {/* Braille Module */}
        {currentModule === 'braille' && (
          <>
            {brailleScreen === 'upload' && (
              <BrailleUpload onUpload={handleBrailleUpload} />
            )}
            {brailleScreen === 'evaluation' && (
              <BrailleEvaluation onBack={handleBrailleBack} />
            )}
          </>
        )}

        {/* Quiz Module */}
        {currentModule === 'quiz' && (
          <>
            {quizScreen === 'start' && <QuizStart onStart={handleQuizStart} />}
            {quizScreen === 'question' && quizQuestions.length > 0 && (
              <QuizQuestion
                question={quizQuestions[currentQuestionIndex]}
                questionNumber={currentQuestionIndex + 1}
                totalQuestions={quizQuestions.length}
                onSubmit={handleQuizSubmit}
                onSkip={handleQuizSkip}
              />
            )}
            {quizScreen === 'feedback' && quizQuestions.length > 0 && (
              <QuizFeedback
                question={quizQuestions[currentQuestionIndex].text}
                answer={currentAnswer}
                expectedAnswer={quizQuestions[currentQuestionIndex].expectedAnswer || ''}
                feedback={quizQuestions[currentQuestionIndex].feedback || ''}
                onNext={handleQuizNext}
              />
            )}
          </>
        )}

        {/* History Module */}
        {currentModule === 'history' && (
          <>
            {historyScreen === 'home' && (
              <HistoryHome onSelectGrade={handleSelectGrade} />
            )}
            {historyScreen === 'lessons' && (
              <LessonList
                grade={selectedGrade}
                onSelectLesson={handleSelectLesson}
                onBack={handleHistoryBack}
              />
            )}
            {historyScreen === 'player' && (
              <LessonPlayer lessonId={selectedLesson} onBack={handleHistoryBack} />
            )}
          </>
        )}
      </main>

      {/* Bottom Navigation */}
      <Navigation currentModule={currentModule} onNavigate={handleNavigate} />
    </div>
  );
}