import { useState, useEffect } from 'react';
import { useTTS } from './contexts/TTSContext';
import { Navigation } from './components/Navigation';
import { HomePage } from './components/HomePage';
import { VoiceCommandSystem } from './components/VoiceCommandSystem';

// Module Components
import { DocumentModule } from './components/document/DocumentModule';
import { BrailleUpload } from './components/braille/BrailleUpload';
import { BrailleEvaluation } from './components/braille/BrailleEvaluation';
import { QuizStart } from './components/quiz/QuizStart';
import { QuizQuestion } from './components/quiz/QuizQuestion';
import { QuizFeedback } from './components/quiz/QuizFeedback';
import { QuizSummary } from './components/quiz/QuizSummary';
import { QuizDashboard } from './components/quiz/QuizDashboard';
import UserAuth from './components/UserAuth';
import { HistoryHome } from './components/history/HistoryHome';
import { LessonList } from './components/history/LessonList';
import { LessonPlayer } from './components/history/LessonPlayer';

// ✅ NEW: Import quizService instead of local data
import { quizService, QuizSetListItem, QuizSetSummary, GenerateQuestionResponse } from './services/quizService';

type Module = 'home' | 'document' | 'braille' | 'quiz' | 'history';
type BrailleScreen = 'upload' | 'evaluation';
type QuizScreen = 'start' | 'question' | 'feedback' | 'summary' | 'dashboard';
type HistoryScreen = 'home' | 'lessons' | 'player';

export function App() {
  const [currentModule, setCurrentModule] = useState<Module>('home');

  // Braille module state
  const [brailleScreen, setBrailleScreen] = useState<BrailleScreen>('upload');

  // =========================
  //  QUIZ STATE
  // =========================
  const [quizScreen, setQuizScreen] = useState<QuizScreen>('start');
  const [selectedTopic, setSelectedTopic] = useState<string>('');
  const [quizQuestions, setQuizQuestions] = useState<GenerateQuestionResponse[]>([]);
  const [quizSetId, setQuizSetId] = useState<string | null>(null);
  const [attemptId, setAttemptId] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<GenerateQuestionResponse | null>(null);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [evaluationResult, setEvaluationResult] = useState<any>(null);
  const [questionNumber, setQuestionNumber] = useState(1);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);
  const [quizSummary, setQuizSummary] = useState<QuizSetSummary | null>(null);
  const [savedQuizSets, setSavedQuizSets] = useState<QuizSetListItem[]>([]);
  // User state for Quiz
  const [quizUser, setQuizUser] = useState<string | null>(() => {
    return localStorage.getItem('quizUser');
  });

  // Persist quizUser to localStorage
  useEffect(() => {
    if (quizUser) {
      localStorage.setItem('quizUser', quizUser);
    } else {
      localStorage.removeItem('quizUser');
    }
  }, [quizUser]);

  useEffect(() => {
    const loadSets = async () => {
      if (!quizUser) {
        setSavedQuizSets([]);
        return;
      }
      try {
        const res = await quizService.getUserQuizSets(quizUser);
        setSavedQuizSets(res.quiz_sets);
      } catch (err) {
        console.error('Failed to load quiz sets', err);
      }
    };

    loadSets();
  }, [quizUser]);

  // History module state
  const [historyScreen, setHistoryScreen] = useState<HistoryScreen>('home');
  const [selectedGrade, setSelectedGrade] = useState<number>(10);
  const [selectedLesson, setSelectedLesson] = useState<number>(1);

  const handleNavigate = (module: string) => {
    const target = module as Module;
    setCurrentModule(target);

    // Reset module states when navigating
    if (target === 'braille') {
      setBrailleScreen('upload');
    } else if (target === 'quiz') {
      handleQuizHome();
      // Do NOT reset quizUser here; keep login persistent until logout
    } else if (target === 'history') {
      setHistoryScreen('home');
    }

    // Voice feedback: announce new section for visually impaired users
    const labels: Record<string, string> = {
      home: 'Home',
      document: 'Documents',
      braille: 'Braille',
      quiz: 'Quiz',
      history: 'History',
    };
    announce(labels[target] || target);
  };
  // Handler for successful login/register
  const handleQuizAuthSuccess = (username: string) => {
    setQuizUser(username);
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

  // Braille Module Handlers
  const handleBrailleUpload = (file: File) => {
    // File is received, now proceed to evaluation
    setBrailleScreen('evaluation');
  };

  const handleBrailleBack = () => {
    setBrailleScreen('upload');
  };

  // =========================
  // UPDATED QUIZ HANDLERS
  // =========================
  const handleQuizStart = async (topic: string, existingSetId?: string) => {
    if (!quizUser) return;
    try {
      const quizSet = await quizService.startQuizSet(quizUser, topic, existingSetId);
      setSelectedTopic(quizSet.chapter_name);
      setQuizSetId(quizSet.set_id);
      setAttemptId(quizSet.attempt_id);
      setQuizQuestions(quizSet.questions);
      setCurrentQuestion(quizSet.questions[0]);
      setCurrentQuestionIndex(0);
      setQuestionNumber(1);
      setCorrectCount(0);
      setCurrentAnswer('');
      setEvaluationResult(null);
      setQuizSummary(null);
      setQuizScreen('question');
    } catch (err) {
      console.error('Failed to start quiz', err);
    }
  };

  const handleQuizSubmit = async (answer: string) => {
    if (!currentQuestion || !quizSetId || !attemptId || !quizUser) return;

    setCurrentAnswer(answer);

    try {
      const result = await quizService.submitQuizSetAnswer(
        quizSetId,
        attemptId,
        quizUser,
        currentQuestionIndex,
        answer
      );

      setEvaluationResult(result);
      if (result.correct) setCorrectCount((prev) => prev + 1);
      setQuizScreen('feedback');
    } catch (err) {
      console.error('Failed to submit answer', err);
    }
  };

  const handleQuizNext = async () => {
    const nextIndex = currentQuestionIndex + 1;
    if (nextIndex >= quizQuestions.length) {
      await handleQuizComplete();
      return;
    }

    setCurrentQuestionIndex(nextIndex);
    setQuestionNumber(nextIndex + 1);
    setCurrentQuestion(quizQuestions[nextIndex]);
    setCurrentAnswer('');
    setEvaluationResult(null);
    setQuizScreen('question');
  };

  const handleQuizSkip = async () => {
    await handleQuizSubmit('Skipped');
  };

  const handleQuizComplete = async () => {
    if (!quizSetId || !attemptId || !quizUser) return;
    try {
      const completion = await quizService.completeQuizAttempt(quizSetId, attemptId, quizUser);
      setQuizSummary(completion.summary);
      setQuizScreen('summary');

      const sets = await quizService.getUserQuizSets(quizUser);
      setSavedQuizSets(sets.quiz_sets);
    } catch (err) {
      console.error('Failed to complete quiz', err);
    }
  };

  function handleQuizHome() {
    setQuizScreen('start');
    setCurrentQuestion(null);
    setCurrentAnswer('');
    setEvaluationResult(null);
    setSelectedTopic('');
    setQuestionNumber(1);
    setCurrentQuestionIndex(0);
    setCorrectCount(0);
    setQuizSummary(null);
    setQuizSetId(null);
    setAttemptId(null);
  }

  const handleShowDashboard = () => {
    setQuizScreen('dashboard');
  };

  const handleRetakeSet = (setId: string, chapterName: string) => {
    handleQuizStart(chapterName, setId);
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

  const { announce } = useTTS();

  return (
    <div className="min-h-screen bg-background">
      {/* Global Logout Button for logged-in user */}
      {quizUser && (
  <button
    style={{ position: 'fixed', top: 16, right: 16, zIndex: 1000, background: '#fff', border: '1px solid #ccc', borderRadius: 8, padding: '8px 16px', fontWeight: 600, cursor: 'pointer', boxShadow: '0 2px 8px #0001' }}
    onClick={() => {
      if (window.confirm('Are you sure you want to logout?')) setQuizUser(null);
    }}
  >
    Logout ({quizUser})
  </button>
)}
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
        {currentModule === 'document' && <DocumentModule />}
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

        {/* =========================
            UPDATED QUIZ RENDER
           ========================= */}
        {currentModule === 'quiz' && (
  <>
    {quizUser == null ? (
      <UserAuth onAuthSuccess={handleQuizAuthSuccess} />
    ) : quizScreen === 'start' && (
      <QuizStart onStart={handleQuizStart} onViewSaved={handleShowDashboard} hasSavedSets={savedQuizSets.length > 0} />
    )}

    {quizUser && quizScreen === 'dashboard' && (
      <QuizDashboard
        sets={savedQuizSets}
        onRetake={handleRetakeSet}
        onBack={handleQuizHome}
      />
    )}

    {quizUser && quizScreen === 'question' && currentQuestion && (
      <QuizQuestion
        question={currentQuestion}
        questionNumber={questionNumber}
        totalQuestions={quizQuestions.length || 10}
        onSubmit={handleQuizSubmit}
        onSkip={handleQuizSkip}
      />
    )}

    {quizUser && quizScreen === 'feedback' && evaluationResult && currentQuestion && (
      <QuizFeedback
        question={currentQuestion.question}
        answer={currentAnswer}
        result={evaluationResult}
        onNext={handleQuizNext}
        onGoHome={handleQuizHome}
        isLastQuestion={questionNumber === quizQuestions.length}
      />
    )}

    {quizUser && quizScreen === 'summary' && quizSummary && (
      <QuizSummary
        summary={quizSummary}
        correctCount={correctCount}
        totalQuestions={quizQuestions.length}
        onRetake={() => quizSetId && handleRetakeSet(quizSetId, selectedTopic)}
        onGoHome={handleQuizHome}
        onStartNew={() => setQuizScreen('start')}
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