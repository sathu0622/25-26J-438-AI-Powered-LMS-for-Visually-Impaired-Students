import { useEffect, useState } from 'react';
import { Volume2, Mic, Send, Flag, Loader2 } from 'lucide-react';
import { FreeTextQuestion as FreeTextQuestionType } from '../../services/freeTextService';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { VoiceRecorder } from '../MockVoiceRecorder';
import { useTTS } from '../../contexts/TTSContext';

interface FreeTextQuestionProps {
  question: FreeTextQuestionType;
  questionIndex: number;
  onSubmit: (answer: string) => void;
  onFinish: () => void;
  onBack?: () => void;
  loading?: boolean;
  isRetake?: boolean;
}

export const FreeTextQuestion = ({
  question,
  questionIndex,
  onSubmit,
  onFinish,
  onBack,
  loading,
  isRetake,
}: FreeTextQuestionProps) => {
  const [answer, setAnswer] = useState('');
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const { speak, cancel } = useTTS();

  const questionNumber = questionIndex + 1;

  // Announce new question
  useEffect(() => {
    cancel();
    const timer = setTimeout(() => {
      const retakeNote = isRetake ? 'This is a retake session. ' : '';
      speak(
        `${retakeNote}Free-text question ${questionNumber}. ${question.question}. Type or record your answer. Press Space or Enter to record, Q to repeat question, F to finish session, Backspace to go back.`,
        { interrupt: true }
      );
    }, 400);

    return () => {
      clearTimeout(timer);
      cancel();
    };
  }, [question.question, questionNumber, isRetake, speak, cancel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName;
      const isTyping = targetTag === 'TEXTAREA' || targetTag === 'INPUT';

      // Space/Enter to record or submit
      if ((e.key === ' ' || e.key === 'Enter') && !isTyping) {
        e.preventDefault();
        if (answer.trim() && !showVoiceModal) {
          handleSubmit();
        } else {
          handleVoiceToggle();
        }
      }

      // R to record
      if ((e.key === 'r' || e.key === 'R') && !isTyping) {
        e.preventDefault();
        handleVoiceToggle();
      }

      // Q to repeat question
      if ((e.key === 'q' || e.key === 'Q') && !isTyping) {
        e.preventDefault();
        handleReadQuestion();
      }

      // F to finish
      if ((e.key === 'f' || e.key === 'F') && !isTyping) {
        e.preventDefault();
        onFinish();
      }

      // Backspace or B to go back
      if ((e.key === 'Backspace' || e.key === 'b' || e.key === 'B') && !isTyping) {
        e.preventDefault();
        if (onBack) onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [answer, showVoiceModal, onFinish, onBack]);

  const handleReadQuestion = () => {
    cancel();
    speak(`Question ${questionNumber}. ${question.question}`);
  };

  const handleVoiceToggle = () => {
    setInputMode('voice');
    setShowVoiceModal(true);
  };

  const handleVoiceSubmit = (text: string) => {
    setAnswer(text);
    setShowVoiceModal(false);
    setInputMode('voice');
  };

  const handleSubmit = () => {
    if (!answer.trim() || loading) return;
    onSubmit(answer.trim());
    setAnswer('');
    setShowVoiceModal(false);
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6 pb-24">
      {/* Progress bar */}
      <div className="space-y-2" aria-live="polite">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Free-Text Quiz • Question {questionNumber}</span>
          <span>Answered: {questionIndex}</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted" aria-hidden>
          <div
            className="h-full bg-green-500 transition-all"
            style={{ width: `${Math.min(100, Math.max(10, questionIndex ? (questionIndex / (questionIndex + 1)) * 100 : 20))}%` }}
          />
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1"><Volume2 className="h-4 w-4" />Press Q to repeat</span>
          <span className="inline-flex items-center gap-1"><Mic className="h-4 w-4" />Press R to record</span>
          <span className="inline-flex items-center gap-1"><Flag className="h-4 w-4" />Press F to finish</span>
          <span className="inline-flex items-center gap-1">Press B to go back</span>
        </div>
      </div>

      {/* Question card */}
      <Card className="p-6 space-y-4 border-green-200" role="group" aria-labelledby="freetext-question-heading">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 space-y-2">
            <p className="text-xs uppercase tracking-wide text-green-600 font-semibold">
              {isRetake ? 'Retake Mode' : 'AI Generated'} • Free Response
            </p>
            <h2 id="freetext-question-heading" className="text-xl font-semibold leading-relaxed">
              {question.question}
            </h2>
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
      </Card>

      {/* Input mode toggle */}
      <div className="flex justify-center gap-2">
        <Button
          onClick={() => setInputMode('voice')}
          variant={inputMode === 'voice' ? 'default' : 'outline'}
          size="sm"
          className={inputMode === 'voice' ? 'bg-green-600 hover:bg-green-700' : ''}
        >
          <Mic className="mr-2 h-4 w-4" />
          Voice
        </Button>
        <Button
          onClick={() => setInputMode('text')}
          variant={inputMode === 'text' ? 'default' : 'outline'}
          size="sm"
          className={inputMode === 'text' ? 'bg-green-600 hover:bg-green-700' : ''}
        >
          Text
        </Button>
      </div>

      {/* Answer input */}
      <Card className="p-6 space-y-4">
        <label htmlFor="freetext-answer" className="text-sm font-medium">
          Your answer (evaluated by meaning, not exact match)
        </label>
        <Textarea
          id="freetext-answer"
          value={answer}
          onChange={(e) => {
            setAnswer(e.target.value);
            setInputMode('text');
          }}
          placeholder={inputMode === 'voice' ? 'Tap microphone and speak your answer…' : 'Type your detailed answer here…'}
          className="min-h-[140px] text-base"
        />
      </Card>

      {/* Action buttons */}
      <div className="grid gap-3 sm:grid-cols-3">
        {inputMode === 'voice' && (
          <Button
            onClick={handleVoiceToggle}
            variant="outline"
            size="lg"
            className="min-h-[56px]"
            disabled={loading}
          >
            <Mic className="mr-2 h-5 w-5" />
            Record Answer
          </Button>
        )}
        <Button
          onClick={handleSubmit}
          disabled={!answer.trim() || Boolean(loading)}
          size="lg"
          className="min-h-[56px] bg-green-600 hover:bg-green-700"
        >
          {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Send className="mr-2 h-5 w-5" />}
          Submit & Next
        </Button>
        <Button
          variant="ghost"
          onClick={onFinish}
          size="lg"
          className="min-h-[56px]"
        >
          <Flag className="mr-2 h-5 w-5" />
          Finish Quiz
        </Button>
      </div>

      {/* Voice recorder modal */}
      <VoiceRecorder
        isOpen={showVoiceModal}
        onClose={() => setShowVoiceModal(false)}
        onSubmit={handleVoiceSubmit}
        title="Record your answer"
        context={question.question}
      />
    </div>
  );
};
