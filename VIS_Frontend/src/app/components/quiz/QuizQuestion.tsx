import { useState, useEffect, useRef } from 'react';
import { Volume2, Mic, Send, SkipForward, Check } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { VoiceRecorder } from '../MockVoiceRecorder';
import { useTTS } from '../../contexts/TTSContext';

interface Question {
  id: number;
  text: string;
  topic: string;
}

interface QuizQuestionProps {
  question: {
    question: string;
    correct_answer: string;
    key_phrase: string;
    year?: string; // Optional year for past paper questions
    options?: string[]; // MCQ options (4 choices)
    correct_index?: number; // Index of correct answer in options array
  };
  questionNumber: number;
  totalQuestions: number;
  onSubmit: (answer: string) => void;
  onSkip: () => void;
  onBack?: () => void;
  isPastPaper?: boolean; // Flag to indicate if this is a past paper question
  /** Unique per question row; avoids re-running effects when parent recreates `question` every render (e.g. quiz timer ticks). */
  questionIdentityKey?: string;
  /** TTS rate (defaults follow ttsService: ~0.95). Use slightly higher for timed quizzes. */
  ttsRate?: number;
  /** Shorter prompts for timed quizzes to save listening time */
  timedQuiz?: boolean;
}

export const QuizQuestion = ({
  question,
  questionNumber,
  totalQuestions,
  onSubmit,
  onSkip,
  onBack,
  isPastPaper = false,
  questionIdentityKey,
  ttsRate,
  timedQuiz = false,
}: QuizQuestionProps) => {
  const [answer, setAnswer] = useState('');
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [hasReadQuestion, setHasReadQuestion] = useState(false);
  const [showVoiceModal, setShowVoiceModal] = useState(false);

  const { speak, cancel } = useTTS();
  const questionRef = useRef(question);
  const selectedOptionRef = useRef<number | null>(null);
  questionRef.current = question;

  // Check if this is an MCQ question
  const isMCQ = question.options && question.options.length > 0;
  const optionLabels = ['A', 'B', 'C', 'D'];

  const readAloudDepsKey =
    questionIdentityKey ??
    `${question.question}|||${question.options?.join('\u241E') ?? ''}|${question.year ?? ''}|${questionNumber}`;

  useEffect(() => {
    cancel();
    setHasReadQuestion(false);
    setSelectedOption(null);
    selectedOptionRef.current = null;
    setAnswer('');
    const q = questionRef.current;
    const mcqOpts = q.options && q.options.length > 0;
    const timer = setTimeout(() => {
      const yearAnnouncement =
        isPastPaper && q.year ? `From year ${q.year}. ` : '';

      const rate = timedQuiz ? ttsRate ?? 1.2 : ttsRate;

      if (mcqOpts && q.options) {
        const optionsText = q.options
          .map((opt, idx) => `${optionLabels[idx]}: ${opt}`)
          .join('. ');
        const intro = timedQuiz
          ? `${yearAnnouncement}Question ${questionNumber} of ${totalQuestions}. ${q.question}. Choices: ${optionsText}. Keys: A-D to choose; Enter submits; Q repeat; S skip.`
          : `Question ${questionNumber}. ${yearAnnouncement}${q.question}. This is a multiple choice question. ${optionsText}. Press A, B, C, or D to select your answer. Press Q to repeat question. Press S to skip. Press Backspace to go back.`;
        speak(intro, { interrupt: true, rate });
      } else {
        speak(
          `Question ${questionNumber}. ${yearAnnouncement}${q.question}. Press Space or Enter to record your answer, Press Q to repeat question, Press R to record, Press S to skip question, Press Backspace to go back.`,
          { interrupt: true, ...(ttsRate !== undefined ? { rate: ttsRate } : {}) }
        );
      }
      setHasReadQuestion(true);
    }, timedQuiz ? 150 : 500);
    return () => {
      clearTimeout(timer);
      cancel();
    };
  }, [readAloudDepsKey, questionNumber, totalQuestions, isPastPaper, timedQuiz, ttsRate, speak, cancel]);

  useEffect(() => {
    selectedOptionRef.current = selectedOption;
  }, [selectedOption]);

  // Keyboard shortcuts (capture phase: reliable Enter submit before nested button handlers)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === 'TEXTAREA' || tag === 'INPUT') return;

      // MCQ option selection (A, B, C, D)
      const qNow = questionRef.current;
      const opts = qNow.options;
      const isMcqHere = opts && opts.length > 0;
      if (isMcqHere && opts) {
        const key = e.key.toUpperCase();
        const optionIndex = optionLabels.indexOf(key);
        if (optionIndex !== -1 && optionIndex < opts.length) {
          e.preventDefault();
          e.stopPropagation();
          handleOptionSelect(optionIndex);
          return;
        }
      }

      // Space or Enter: submit MCQ with latest selection from ref (avoids stale closures)
      if (e.key === ' ' || e.key === 'Enter') {
        if (isMcqHere && opts) {
          const idx = selectedOptionRef.current;
          if (idx !== null && idx >= 0 && idx < opts.length) {
            e.preventDefault();
            e.stopPropagation();
            onSubmit(opts[idx]);
            setSelectedOption(null);
            selectedOptionRef.current = null;
            setAnswer('');
          }
          return;
        }

        if (answer.trim() && !showVoiceModal) {
          e.preventDefault();
          handleSubmit();
        } else if (!isMcqHere) {
          e.preventDefault();
          handleVoiceToggle();
        }
        return;
      }

      // R key to start recording (only for non-MCQ)
      if ((e.key === 'r' || e.key === 'R') && !isMcqHere) {
        e.preventDefault();
        handleVoiceToggle();
      }

      // S key to skip
      if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        e.stopPropagation();
        onSkip();
        setAnswer('');
        setSelectedOption(null);
        selectedOptionRef.current = null;
      }

      // Q key to read question
      if (e.key === 'q' || e.key === 'Q') {
        e.preventDefault();
        handleReadQuestion();
      }

      // Backspace or B key to go back
      if (e.key === 'Backspace' || e.key === 'b' || e.key === 'B') {
        e.preventDefault();
        if (onBack) onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress, true);
    return () => window.removeEventListener('keydown', handleKeyPress, true);
  }, [showVoiceModal, answer, questionIdentityKey, readAloudDepsKey, onBack, onSubmit, onSkip]);

  const mcqAnnouncementRate =
    timedQuiz ? ttsRate ?? 1.2 : ttsRate !== undefined ? ttsRate : undefined;

  const handleReadQuestion = () => {
    cancel();
    const q = questionRef.current;
    const yearAnnouncement =
      isPastPaper && q.year ? `From year ${q.year}. ` : '';

    const opts = q.options;
    if (opts && opts.length > 0) {
      const optionsText = opts
        .map((opt, idx) => `${optionLabels[idx]}: ${opt}`)
        .join('. ');
      const txt = timedQuiz
        ? `${yearAnnouncement}${q.question}. ${optionsText}`
        : `Question ${questionNumber}. ${yearAnnouncement}${q.question}. Options: ${optionsText}`;
      speak(txt, {
        interrupt: true,
        ...(mcqAnnouncementRate !== undefined ? { rate: mcqAnnouncementRate } : {}),
      });
    } else {
      speak(`Question ${questionNumber}. ${yearAnnouncement}${q.question}`, {
        interrupt: true,
        ...(mcqAnnouncementRate !== undefined ? { rate: mcqAnnouncementRate } : {}),
      });
    }
  };

  const handleOptionSelect = (index: number) => {
    if (!question.options) return;
    selectedOptionRef.current = index;
    setSelectedOption(index);
    const selectedText = question.options[index];
    const cue = timedQuiz
      ? `${optionLabels[index]}: ${selectedText}. Press Enter to submit.`
      : `Selected ${optionLabels[index]}: ${selectedText}. Press Enter to submit.`;
    speak(cue, {
      interrupt: true,
      ...(mcqAnnouncementRate !== undefined ? { rate: mcqAnnouncementRate } : {}),
    });
  };

  const handleVoiceToggle = () => {
    setInputMode('voice');
    setShowVoiceModal(true);
  };

  const handleVoiceSubmit = (text: string) => {
    setAnswer(text);
    setShowVoiceModal(false);
  };

  const handleSubmit = () => {
    if (isMCQ && selectedOption !== null && question.options) {
      // For MCQ, submit the selected option text
      onSubmit(question.options[selectedOption]);
      setSelectedOption(null);
      selectedOptionRef.current = null;
      setAnswer('');
      setHasReadQuestion(false);
    } else if (answer.trim()) {
      onSubmit(answer);
      setAnswer('');
      setHasReadQuestion(false);
    }
  };

  const handleSkip = () => {
    onSkip();
    setAnswer('');
    setSelectedOption(null);
    selectedOptionRef.current = null;
    setHasReadQuestion(false);
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            {isMCQ 
              ? (timedQuiz
                  ? 'A–D choose • Enter submits • Q repeat • S skip • B back'
                  : 'Press A/B/C/D to select • Q to repeat • S to skip • B to go back')
              : 'Press Q to repeat • Space to record • S to skip • B to go back'
            }
          </span>
          <span>
            {questionNumber} / {totalQuestions}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${(questionNumber / totalQuestions) * 100}%` }}
          />
        </div>
      </div>

     

      {/* Question Card */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 space-y-2">
              <h2 className="text-sm text-muted-foreground">
                Question {questionNumber}
              </h2>
              <p className="text-lg leading-relaxed">{question.question}</p>
            </div>
            <Button
              onClick={handleReadQuestion}
              variant="outline"
              size="icon"
              aria-label="Read question aloud"
            >
              <Volume2 className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </Card>

      {/* MCQ Options */}
      {isMCQ && question.options ? (
        <Card className="p-6">
          <div className="space-y-3">
            <label className="text-sm font-medium">Select your answer:</label>
            <div className="space-y-3">
              {question.options.map((option, index) => (
                <Button
                  key={index}
                  onClick={() => handleOptionSelect(index)}
                  variant={selectedOption === index ? 'default' : 'outline'}
                  className={`w-full justify-start min-h-[56px] text-left p-4 ${
                    selectedOption === index 
                      ? 'ring-2 ring-primary ring-offset-2' 
                      : ''
                  }`}
                  aria-label={`Option ${optionLabels[index]}: ${option}`}
                >
                  <span className="flex items-center gap-3 w-full">
                    <span className={`flex h-8 w-8 items-center justify-center rounded-full border-2 font-semibold ${
                      selectedOption === index 
                        ? 'bg-primary-foreground text-primary border-primary-foreground' 
                        : 'border-muted-foreground'
                    }`}>
                      {optionLabels[index]}
                    </span>
                    <span className="flex-1 text-base">{option}</span>
                    {selectedOption === index && (
                      <Check className="h-5 w-5 text-primary-foreground" />
                    )}
                  </span>
                </Button>
              ))}
            </div>
          </div>
        </Card>
      ) : (
        <>
          {/* Input Mode Toggle */}
          <div className="flex justify-center gap-2">
            <Button
              onClick={() => setInputMode('voice')}
              variant={inputMode === 'voice' ? 'default' : 'outline'}
              size="sm"
            >
              <Mic className="mr-2 h-4 w-4" />
              Voice
            </Button>
            <Button
              onClick={() => setInputMode('text')}
              variant={inputMode === 'text' ? 'default' : 'outline'}
              size="sm"
            >
              Text
            </Button>
          </div>

      {/* Answer Input */}
          <Card className="p-6">
            <div className="space-y-4">
              <label htmlFor="answer-input" className="text-sm">
                Your Answer
              </label>
              <Textarea
                id="answer-input"
                value={answer}
                onChange={(e) => {
                  setAnswer(e.target.value);
                  setInputMode('text');
                }}
                placeholder={
                  inputMode === 'voice'
                    ? 'Tap microphone and speak your answer...'
                    : 'Type your answer here...'
                }
                className="min-h-[120px] text-base"
              />
            </div>
          </Card>

          {/* Action Buttons for Text/Voice Mode */}
          <div className="grid gap-3 sm:grid-cols-2">
            {inputMode === 'voice' && (
              <Button
                onClick={handleVoiceToggle}
                variant="outline"
                size="lg"
                className="min-h-[56px]"
              >
                <Mic className="mr-2 h-5 w-5" />
                Record Answer
              </Button>
            )}
            <Button
              onClick={handleSubmit}
              disabled={!answer.trim()}
              size="lg"
              className="min-h-[56px]"
            >
              <Send className="mr-2 h-5 w-5" />
              Submit Answer
            </Button>
          </div>
        </>
      )}

      {/* Submit Button for MCQ */}
      {isMCQ && (
        <Button
          onClick={handleSubmit}
          disabled={selectedOption === null}
          size="lg"
          className="w-full min-h-[56px]"
        >
          <Send className="mr-2 h-5 w-5" />
          Submit Answer
        </Button>
      )}

      {/* Skip Button */}
      <Button
        onClick={handleSkip}
        variant="ghost"
        className="w-full"
        aria-label="Skip question"
      >
        <SkipForward className="mr-2 h-4 w-4" />
        Skip Question
      </Button>

      {/* Voice Recorder Modal (only for non-MCQ) */}
      {!isMCQ && (
        <VoiceRecorder
          isOpen={showVoiceModal}
          onClose={() => setShowVoiceModal(false)}
          onSubmit={handleVoiceSubmit}
          title="Record Your Answer"
          context={question.question}
        />
      )}
    </div>
  );
};