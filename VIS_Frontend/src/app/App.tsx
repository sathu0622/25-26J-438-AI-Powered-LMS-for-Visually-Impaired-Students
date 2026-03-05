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
import { QuizModeSelect } from './components/quiz/QuizModeSelect';
import { PastPaperQuizStart } from './components/quiz/PastPaperQuizStart';
import { QuizLoading } from './components/quiz/QuizLoading';
import { AdaptiveStart } from './components/quiz/AdaptiveStart';
import { AdaptiveQuestion } from './components/quiz/AdaptiveQuestion';
import { AdaptiveSummary } from './components/quiz/AdaptiveSummary';
import { AdaptiveFeedback } from './components/quiz/AdaptiveFeedback.tsx';
import { QuizQuestion } from './components/quiz/QuizQuestion';
import { QuizFeedback } from './components/quiz/QuizFeedback';
import { QuizSummary } from './components/quiz/QuizSummary';
import { QuizDashboard } from './components/quiz/QuizDashboard';
import UserAuth from './components/UserAuth';
import { HistoryHome } from './components/history/HistoryHome';
import { LessonList } from './components/history/LessonList';
import { LessonPlayer } from './components/history/LessonPlayer';
import { UserProfilePage } from './components/UserProfilePage';

// ✅ NEW: Import quizService instead of local data
import { quizService, QuizSetListItem, QuizSetSummary, GenerateQuestionResponse } from './services/quizService';
import { pastPaperService, PastPaperQuestion } from './services/pastPaperService';
import { adaptiveService, AdaptiveItem, AdaptiveAnswerResponse } from './services/adaptiveService';

type Module = 'home' | 'document' | 'braille' | 'quiz' | 'history';
type BrailleScreen = 'upload' | 'evaluation';
type QuizScreen = 'start' | 'question' | 'feedback' | 'summary' | 'dashboard' | 'profile';
type HistoryScreen = 'home' | 'lessons' | 'player';
type QuizMode = 'none' | 'generative' | 'adaptive' | 'pastpaper';
type AdaptiveScreen = 'start' | 'question' | 'feedback' | 'summary';

export function App() {
  const [currentModule, setCurrentModule] = useState<Module>('home');

  // Braille module state
  const [brailleScreen, setBrailleScreen] = useState<BrailleScreen>('upload');

  // =========================
  //  QUIZ STATE
  // =========================
  const [quizScreen, setQuizScreen] = useState<QuizScreen>('start');
  const [quizMode, setQuizMode] = useState<QuizMode>('none');
  const [selectedTopic, setSelectedTopic] = useState<string>('');
  const [quizQuestions, setQuizQuestions] = useState<GenerateQuestionResponse[]>([]);
  const [pastPaperQuestions, setPastPaperQuestions] = useState<PastPaperQuestion[]>([]);
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
  const [quizGenerating, setQuizGenerating] = useState(false);
  const [currentGenerationTopic, setCurrentGenerationTopic] = useState('');
  const [currentGenerationMode, setCurrentGenerationMode] = useState<'generative' | 'pastpaper' | 'adaptive'>('generative');
  // Adaptive state
  const [adaptiveScreen, setAdaptiveScreen] = useState<AdaptiveScreen>('start');
  const [adaptiveSessionId, setAdaptiveSessionId] = useState<string | null>(null);
  const [adaptiveItem, setAdaptiveItem] = useState<AdaptiveItem | null>(null);
  const [adaptiveTheta, setAdaptiveTheta] = useState<number>(0);
  const [adaptiveCorrect, setAdaptiveCorrect] = useState<number>(0);
  const [adaptiveTotal, setAdaptiveTotal] = useState<number>(0);
  const [adaptiveFeedback, setAdaptiveFeedback] = useState<string>('');
  const [adaptiveLoading, setAdaptiveLoading] = useState<boolean>(false);
  const [adaptiveChapter, setAdaptiveChapter] = useState<string>('');
  const [adaptiveResult, setAdaptiveResult] = useState<AdaptiveAnswerResponse | null>(null);
  const [adaptiveLastQuestion, setAdaptiveLastQuestion] = useState<AdaptiveItem | null>(null);
  const [adaptiveLastAnswer, setAdaptiveLastAnswer] = useState<string>('');
  const [adaptiveNextItem, setAdaptiveNextItem] = useState<AdaptiveItem | null>(null);
  const [adaptiveCompleted, setAdaptiveCompleted] = useState<boolean>(false);
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
      setQuizMode('none');
      setAdaptiveScreen('start');
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
    
    // Set loading state and current generation info
    setQuizGenerating(true);
    setCurrentGenerationTopic(topic);
    setCurrentGenerationMode(quizMode === 'pastpaper' ? 'pastpaper' : 'generative');
    
    // Different logic for past paper vs generative quiz
    if (quizMode === 'pastpaper') {
      try {
        const questions = await pastPaperService.getQuestions(topic);
        setPastPaperQuestions(questions);
        setQuizQuestions([]); // Clear regular quiz questions
        setSelectedTopic(topic);
        setQuestionNumber(1);
        setCurrentQuestionIndex(0);
        setCorrectCount(0);
        setCurrentAnswer('');
        setEvaluationResult(null);
        setQuizSummary(null);
        
        if (questions.length > 0) {
          // Convert past paper question to the format expected by QuizQuestion component
          const firstQuestion = questions[0];
          setCurrentQuestion({
            question: firstQuestion.question,
            correct_answer: firstQuestion.correct_answer,
            key_phrase: firstQuestion.unique_part,
            year: firstQuestion.year
          });
          setQuizScreen('question');
        } else {
          alert('No past paper questions available for this chapter.');
        }
      } catch (error) {
        console.error('Failed to load past paper questions:', error);
        alert('Failed to load past paper questions. Please try again.');
      } finally {
        setQuizGenerating(false);
      }
    } else {
      // Original generative quiz logic
      try {
        const quizSet = await quizService.startQuizSet(quizUser, topic, existingSetId);
        setSelectedTopic(quizSet.chapter_name);
        setQuizSetId(quizSet.set_id);
        setAttemptId(quizSet.attempt_id);
        setQuizQuestions(quizSet.questions);
        setPastPaperQuestions([]); // Clear past paper questions
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
        alert('Failed to generate quiz questions. Please try again.');
      } finally {
        setQuizGenerating(false);
      }
    }
  };

  // Handle canceling quiz generation
  const handleCancelGeneration = () => {
    setQuizGenerating(false);
    setCurrentGenerationTopic('');
    
    // Return to appropriate screen based on mode
    if (quizMode === 'adaptive') {
      setAdaptiveScreen('start');
    } else {
      setQuizScreen('start');
    }
  };

  const handleQuizSubmit = async (answer: string) => {
    if (!currentQuestion || !quizUser) return;

    setCurrentAnswer(answer);

    try {
      let result;
      
      if (quizMode === 'pastpaper') {
        // Use past paper evaluation
        result = await pastPaperService.evaluateAnswer(
          answer,
          currentQuestion.correct_answer,
          currentQuestion.question,
          currentQuestion.year || ''
        );
      } else {
        // Use regular quiz evaluation
        if (!quizSetId || !attemptId) return;
        result = await quizService.submitQuizSetAnswer(
          quizSetId,
          attemptId,
          quizUser,
          currentQuestionIndex,
          answer
        );
      }

      setEvaluationResult(result);
      if (result.correct) setCorrectCount((prev) => prev + 1);
      setQuizScreen('feedback');
    } catch (err) {
      console.error('Failed to submit answer', err);
    }
  };

  const handleQuizNext = async () => {
    const nextIndex = currentQuestionIndex + 1;
    
    // Check quiz length based on quiz mode
    const totalQuestions = quizMode === 'pastpaper' ? pastPaperQuestions.length : quizQuestions.length;
    
    if (nextIndex >= totalQuestions) {
      await handleQuizComplete();
      return;
    }

    setCurrentQuestionIndex(nextIndex);
    setQuestionNumber(nextIndex + 1);
    
    // Set next question based on quiz mode
    if (quizMode === 'pastpaper') {
      const nextPastPaperQuestion = pastPaperQuestions[nextIndex];
      setCurrentQuestion({
        question: nextPastPaperQuestion.question,
        correct_answer: nextPastPaperQuestion.correct_answer,
        key_phrase: nextPastPaperQuestion.unique_part,
        year: nextPastPaperQuestion.year
      });
    } else {
      setCurrentQuestion(quizQuestions[nextIndex]);
    }
    
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
    setQuizMode('none');
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
    setAdaptiveSessionId(null);
    setAdaptiveItem(null);
    setAdaptiveFeedback('');
    setAdaptiveTheta(0);
    setAdaptiveCorrect(0);
    setAdaptiveTotal(0);
    setAdaptiveScreen('start');
    setAdaptiveResult(null);
    setAdaptiveLastQuestion(null);
    setAdaptiveLastAnswer('');
    setAdaptiveNextItem(null);
    setAdaptiveCompleted(false);
  }

  const handleShowDashboard = () => {
    setQuizScreen('dashboard');
  };

  const handleRetakeSet = (setId: string, chapterName: string) => {
    handleQuizStart(chapterName, setId);
  };

  const handleViewProfile = () => {
    setQuizScreen('profile');
  };

  const handleProfileBack = () => {
    setQuizScreen('start');
    setQuizMode('none');
  };

  // =========================
  // Adaptive Quiz Handlers
  // =========================

  const handleSelectGenerative = () => {
    setQuizMode('generative');
    setQuizScreen('start');
  };

  const handleSelectAdaptive = () => {
    setQuizMode('adaptive');
    setAdaptiveScreen('start');
  };

  const handleSelectPastPaper = () => {
    setQuizMode('pastpaper');
    setQuizScreen('start');
  };

  const handlePastPaperBack = () => {
    setQuizMode('none');
    setQuizScreen('start');
  };

  const handleAdaptiveStart = async (chapter: string) => {
    if (!quizUser) return;
    
    // Set unified loading state  
    setQuizGenerating(true);
    setCurrentGenerationTopic(chapter);
    setCurrentGenerationMode('adaptive');
    setAdaptiveLoading(true);
    
    try {
      const res = await adaptiveService.start(quizUser, chapter);
      setAdaptiveSessionId(res.session_id);
      setAdaptiveTheta(res.theta);
      setAdaptiveItem(res.item);
      setAdaptiveChapter(chapter);
      setAdaptiveCorrect(0);
      setAdaptiveTotal(0);
      setAdaptiveFeedback('');
      setAdaptiveResult(null);
      setAdaptiveLastQuestion(null);
      setAdaptiveLastAnswer('');
      setAdaptiveNextItem(null);
      setAdaptiveCompleted(false);
      setAdaptiveScreen('question');
    } catch (err) {
      console.error('Failed to start adaptive quiz', err);
      alert('Failed to start adaptive quiz. Please try again.');
    } finally {
      setAdaptiveLoading(false);
      setQuizGenerating(false);
    }
  };

  const handleAdaptiveSubmit = async (answer: string) => {
    if (!adaptiveItem || !adaptiveSessionId || !quizUser) return;
    setAdaptiveLoading(true);
    try {
      setAdaptiveLastQuestion(adaptiveItem);
      setAdaptiveLastAnswer(answer);
      const res = await adaptiveService.answer(adaptiveSessionId, adaptiveItem.item_id, answer, quizUser);
      setAdaptiveTheta(res.theta);
      setAdaptiveCorrect((c) => c + (res.correct ? 1 : 0));
      setAdaptiveTotal((t) => t + 1);
      setAdaptiveFeedback(res.correct ? 'Correct!' : `Incorrect. Correct answer: ${res.correct_answer}`);
      setAdaptiveResult(res);

      if (res.done || !res.next_item) {
        await adaptiveService.finish(adaptiveSessionId, quizUser);
        setAdaptiveCompleted(true);
        setAdaptiveNextItem(null);
      } else {
        setAdaptiveCompleted(false);
        setAdaptiveNextItem(res.next_item);
      }
      setAdaptiveScreen('feedback');
    } catch (err) {
      console.error('Failed to submit adaptive answer', err);
    } finally {
      setAdaptiveLoading(false);
    }
  };

  const handleAdaptiveNext = () => {
    if (adaptiveCompleted || !adaptiveNextItem) {
      setAdaptiveScreen('summary');
      return;
    }
    setAdaptiveItem(adaptiveNextItem);
    setAdaptiveFeedback('');
    setAdaptiveResult(null);
    setAdaptiveLastQuestion(null);
    setAdaptiveLastAnswer('');
    setAdaptiveNextItem(null);
    setAdaptiveCompleted(false);
    setAdaptiveScreen('question');
  };

  const handleAdaptiveFinish = async () => {
    if (adaptiveSessionId && quizUser) {
      try {
        await adaptiveService.finish(adaptiveSessionId, quizUser);
      } catch (err) {
        console.error('Failed to finish adaptive session', err);
      }
    }
    setAdaptiveScreen('summary');
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
    ) : quizScreen === 'profile' ? (
      <UserProfilePage username={quizUser} onBack={handleProfileBack} />
    ) : quizGenerating ? (
      <QuizLoading 
        mode={currentGenerationMode}
        topic={currentGenerationTopic}
        onCancel={handleCancelGeneration}
      />
    ) : quizMode === 'none' ? (
      <QuizModeSelect 
        onSelectGenerative={handleSelectGenerative} 
        onSelectAdaptive={handleSelectAdaptive}
        onSelectPastPaper={handleSelectPastPaper}
        onViewProfile={handleViewProfile}
        username={quizUser}
      />
    ) : quizMode === 'generative' ? (
      <>
        {quizScreen === 'start' && (
          <QuizStart
            onStart={handleQuizStart}
            onViewSaved={handleShowDashboard}
            hasSavedSets={savedQuizSets.length > 0}
          />
        )}

        {quizScreen === 'dashboard' && (
          <QuizDashboard
            sets={savedQuizSets}
            onRetake={handleRetakeSet}
            onBack={handleQuizHome}
          />
        )}

        {quizScreen === 'question' && currentQuestion && (
          <QuizQuestion
            question={currentQuestion}
            questionNumber={questionNumber}
            totalQuestions={quizQuestions.length || 10}
            onSubmit={handleQuizSubmit}
            onSkip={handleQuizSkip}
            isPastPaper={false}
          />
        )}

        {quizScreen === 'feedback' && evaluationResult && currentQuestion && (
          <QuizFeedback
            question={currentQuestion.question}
            answer={currentAnswer}
            result={evaluationResult}
            onNext={handleQuizNext}
            onGoHome={handleQuizHome}
            isLastQuestion={questionNumber === quizQuestions.length}
          />
        )}

        {quizScreen === 'summary' && quizSummary && (
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
    ) : quizMode === 'pastpaper' ? (
      <>
        {quizScreen === 'start' && (
          <PastPaperQuizStart
            onStart={handleQuizStart}
            onBack={handlePastPaperBack}
          />
        )}

        {/* Past Paper Quiz uses same QuizQuestion and QuizFeedback components */}
        {quizScreen === 'question' && currentQuestion && (
          <QuizQuestion
            question={currentQuestion}
            questionNumber={questionNumber}
            totalQuestions={pastPaperQuestions.length || 10}
            onSubmit={handleQuizSubmit}
            onSkip={handleQuizSkip}
            isPastPaper={true}
          />
        )}

        {quizScreen === 'feedback' && evaluationResult && currentQuestion && (
          <QuizFeedback
            question={currentQuestion.question}
            answer={currentAnswer}
            result={evaluationResult}
            onNext={handleQuizNext}
            onGoHome={handleQuizHome}
            isLastQuestion={questionNumber === pastPaperQuestions.length}
          />
        )}

        {quizScreen === 'summary' && quizSummary && (
          <QuizSummary
            summary={quizSummary}
            correctCount={correctCount}
            totalQuestions={pastPaperQuestions.length}
            onRetake={() => quizSetId && handleRetakeSet(quizSetId, selectedTopic)}
            onGoHome={handleQuizHome}
            onStartNew={() => setQuizScreen('start')}
          />
        )}
      </>
    ) : (
      <>
        {adaptiveScreen === 'start' && (
          <AdaptiveStart 
            onStart={handleAdaptiveStart} 
            onBack={() => {
              setQuizMode('none');
              setAdaptiveScreen('start');
            }}
          />
        )}
        {adaptiveScreen === 'question' && adaptiveItem && (
          <AdaptiveQuestion
            item={adaptiveItem}
            theta={adaptiveTheta}
            answeredCount={adaptiveTotal}
            onSubmit={handleAdaptiveSubmit}
            onFinish={handleAdaptiveFinish}
            feedback={adaptiveFeedback}
            lastResult={adaptiveResult}
            lastQuestion={adaptiveLastQuestion}
            lastAnswer={adaptiveLastAnswer}
            loading={adaptiveLoading}
          />
        )}
        {adaptiveScreen === 'feedback' && adaptiveResult && adaptiveLastQuestion && (
          <AdaptiveFeedback
            question={adaptiveLastQuestion}
            answer={adaptiveLastAnswer}
            result={adaptiveResult}
            onNext={handleAdaptiveNext}
            onFinish={handleAdaptiveFinish}
            isFinal={adaptiveCompleted || !adaptiveNextItem}
          />
        )}
        {adaptiveScreen === 'summary' && (
          <AdaptiveSummary
            correctCount={adaptiveCorrect}
            total={adaptiveTotal}
            finalTheta={adaptiveTheta}
            onRestart={() => adaptiveChapter ? handleAdaptiveStart(adaptiveChapter) : setAdaptiveScreen('start')}
            onHome={handleQuizHome}
          />
        )}
      </>
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
