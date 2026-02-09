import { useState, useEffect } from 'react';
import { Volume2, Mic, Send, Loader2, SkipForward, AlertCircle, X } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { MockVoiceRecorder } from '../MockVoiceRecorder';
import { useMockSpeechSynthesis } from '../../hooks/useMockSpeechSynthesis';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface Question {
  id: number;
  text: string;
  topic: string;
}

interface QuizQuestionProps {
  question: Question;
  questionNumber: number;
  totalQuestions: number;
  onSubmit: (answer: string) => void;
  onSkip: () => void;
}

export const QuizQuestion = ({
  question,
  questionNumber,
  totalQuestions,
  onSubmit,
  onSkip,
}: QuizQuestionProps) => {
  const [answer, setAnswer] = useState('');
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [hasReadQuestion, setHasReadQuestion] = useState(false);
  const [showVoiceModal, setShowVoiceModal] = useState(false);

  const { speak, stop } = useMockSpeechSynthesis();

  // Auto-read question with proper speech coordination
  useEffect(() => {
    // STOP all previous speech immediately
    safeCancel();
    stop();
    setHasReadQuestion(false); // Reset for new question
    
    // Announce question after brief delay
    const timer = setTimeout(() => {
      speak(`Question ${questionNumber} of ${totalQuestions}. ${question.text}. Press Space or Enter to record your answer, Press Q to repeat question, Press R to record, Press S to skip question.`);
      setHasReadQuestion(true);
    }, 500);

    return () => {
      clearTimeout(timer);
      stop();
      safeCancel();
    };
  }, [question.id]); // Only re-run when question changes, not when hasReadQuestion changes

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Space or Enter to toggle recording/submit (when not in textarea)
      if ((e.key === ' ' || e.key === 'Enter') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        if (answer.trim() && !showVoiceModal) {
          handleSubmit();
        } else {
          handleVoiceToggle();
        }
      }

      // R key to start recording
      if ((e.key === 'r' || e.key === 'R') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        handleVoiceToggle();
      }

      // S key to skip
      if ((e.key === 's' || e.key === 'S') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        handleSkip();
      }

      // Q key to read question
      if ((e.key === 'q' || e.key === 'Q') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        handleReadQuestion();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [showVoiceModal, answer]);

  const handleReadQuestion = () => {
    stop();
    speak(`Question ${questionNumber}. ${question.text}`);
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
    if (answer.trim()) {
      onSubmit(answer);
      setAnswer('');
      setHasReadQuestion(false);
    }
  };

  const handleSkip = () => {
    onSkip();
    setAnswer('');
    setHasReadQuestion(false);
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            Press Q to repeat • Space to record • S to skip
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

      {/* Topic Badge */}
      <div className="flex justify-center">
        <span className="rounded-full bg-secondary/20 px-4 py-1 text-sm text-secondary">
          {question.topic}
        </span>
      </div>

      {/* Question Card */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 space-y-2">
              <h2 className="text-sm text-muted-foreground">
                Question {questionNumber}
              </h2>
              <p className="text-lg leading-relaxed">{question.text}</p>
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

      {/* Action Buttons */}
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

      {/* Voice Recorder Modal */}
      <MockVoiceRecorder
        isOpen={showVoiceModal}
        onClose={() => setShowVoiceModal(false)}
        onSubmit={handleVoiceSubmit}
        title="Record Your Answer"
        context={question.text}
      />
    </div>
  );
};