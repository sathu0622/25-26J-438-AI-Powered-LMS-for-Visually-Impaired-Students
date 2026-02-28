import { useState } from 'react';
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
import { HistoryHome } from './components/history/HistoryHome';
import { LessonList } from './components/history/LessonList';
import { LessonPlayer } from './components/history/LessonPlayer';

// Data
import { getQuestionsByTopic } from './data/quizData';

type Module = 'home' | 'document' | 'braille' | 'quiz' | 'history';

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
    if (target === 'braille') {
      setBrailleScreen('upload');
    } else if (target === 'quiz') {
      setQuizScreen('start');
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

  const { announce } = useTTS();

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