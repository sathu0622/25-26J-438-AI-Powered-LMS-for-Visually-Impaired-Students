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
import { ChapterList } from './components/history/ChapterList';
import { TopicList } from './components/history/TopicList';
import { LessonPlayer } from './components/history/LessonPlayer';

// Data
import { getQuestionsByTopic } from './data/quizData';

type Module = 'home' | 'document' | 'braille' | 'quiz' | 'history';

type DocumentScreen = 'upload' | 'processing' | 'summary' | 'qa';
type BrailleScreen = 'upload' | 'evaluation';
type QuizScreen = 'start' | 'question' | 'feedback';
type HistoryScreen = 'home' | 'chapters' | 'topics' | 'player';

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
  const [selectedChapter, setSelectedChapter] = useState<number>(0);
  const [selectedChapterName, setSelectedChapterName] = useState<string>('');
  const [selectedTopicIdx, setSelectedTopicIdx] = useState<number>(0);
  const [selectedTopicName, setSelectedTopicName] = useState<string>('');
  const [selectedTopicContent, setSelectedTopicContent] = useState<string>('');

  const handleNavigate = (module: Module) => {
    setCurrentModule(module);

    // Reset module states when navigating
    if (module === 'document') {
      setDocumentScreen('upload');
    } else if (module === 'braille') {
      setBrailleScreen('upload');
    } else if (module === 'quiz') {
      setQuizScreen('start');
    } else if (module === 'history') {
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
  const handleDocumentUpload = (file: File) => {
    setUploadedFile(file);
    setDocumentScreen('processing');
  };

  const handleDocumentProcessingComplete = (summary: string) => {
    setDocumentSummary(summary);
    setDocumentScreen('summary');
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
    setHistoryScreen('chapters');
  };

  const handleSelectChapter = (chapterId: number, chapterName?: string) => {
    setSelectedChapter(chapterId);
    if (chapterName) {
      setSelectedChapterName(chapterName);
    }
    setHistoryScreen('topics');
  };

  const handleSelectTopic = (topicId: number, topicName: string, content: string) => {
    setSelectedTopicIdx(topicId);
    setSelectedTopicName(topicName);
    setSelectedTopicContent(content);
    setHistoryScreen('player');
  };

  const handleHistoryBack = () => {
    if (historyScreen === 'player') {
      setHistoryScreen('topics');
    } else if (historyScreen === 'topics') {
      setHistoryScreen('chapters');
    } else if (historyScreen === 'chapters') {
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
            {documentScreen === 'upload' && (
              <DocumentUpload onUpload={handleDocumentUpload} />
            )}
            {documentScreen === 'processing' && uploadedFile && (
              <DocumentProcessing
                fileName={uploadedFile.name}
                onComplete={handleDocumentProcessingComplete}
              />
            )}
            {documentScreen === 'summary' && (
              <DocumentSummary
                summary={documentSummary}
                onAskQuestion={handleAskQuestion}
              />
            )}
            {documentScreen === 'qa' && (
              <DocumentQA mode={qaMode} onBack={handleBackToSummary} />
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
            {historyScreen === 'chapters' && (
              <ChapterList
                grade={selectedGrade}
                onSelectChapter={handleSelectChapter}
                onBack={handleHistoryBack}
              />
            )}
            {historyScreen === 'topics' && (
              <TopicList
                grade={selectedGrade}
                chapterId={selectedChapter}
                chapterName={selectedChapterName}
                onSelectTopic={handleSelectTopic}
                onBack={handleHistoryBack}
              />
            )}
            {historyScreen === 'player' && (
              <LessonPlayer
                topicName={selectedTopicName}
                content={selectedTopicContent}
                grade={selectedGrade}
                chapterIdx={selectedChapter}
                topicIdx={selectedTopicIdx}
                autoPlay={true}
                onBack={handleHistoryBack}
              />
            )}
          </>
        )}
      </main>

      {/* Bottom Navigation */}
      <Navigation currentModule={currentModule} onNavigate={handleNavigate} />
    </div>
  );
}