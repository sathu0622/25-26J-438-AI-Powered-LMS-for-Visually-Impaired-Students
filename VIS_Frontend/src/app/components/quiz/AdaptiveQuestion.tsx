import { useEffect, useMemo, useState } from 'react';
import { Volume2, Mic, Send, Flag, Play, Loader2, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import { AdaptiveItem, AdaptiveAnswerResponse } from '../../services/adaptiveService';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { VoiceRecorder } from '../MockVoiceRecorder';
import { useTTS } from '../../contexts/TTSContext';

interface AdaptiveQuestionProps {
  item: AdaptiveItem;
  onSubmit: (answer: string) => void;
  onFinish: () => void;
  feedback?: string;
  theta: number;
  loading?: boolean;
  answeredCount?: number;
  lastResult?: AdaptiveAnswerResponse | null;
  lastQuestion?: AdaptiveItem | null;
  lastAnswer?: string;
}

export const AdaptiveQuestion = ({
  item,
  onSubmit,
  onFinish,
  feedback,
  theta,
  loading,
  answeredCount = 0,
  lastResult,
  lastQuestion,
  lastAnswer,
}: AdaptiveQuestionProps) => {
  const [answer, setAnswer] = useState('');
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const { speak, cancel } = useTTS();

  const questionNumber = useMemo(() => answeredCount + 1, [answeredCount]);

  useEffect(() => {
    cancel();
    const timer = setTimeout(() => {
      speak(
        `Adaptive question ${questionNumber}. Difficulty ${item.difficulty_label}. ${item.question}. Press Space or Enter to record or submit, Q to repeat, R to record, and F to finish the session.`,
        { interrupt: true }
      );
    }, 400);

    return () => {
      clearTimeout(timer);
      cancel();
    };
  }, [item.item_id, questionNumber, item.question, item.difficulty_label, speak, cancel]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName;
      const isTyping = targetTag === 'TEXTAREA' || targetTag === 'INPUT';

      if ((e.key === ' ' || e.key === 'Enter') && !isTyping) {
        e.preventDefault();
        if (answer.trim() && !showVoiceModal) {
          handleSubmit();
        } else {
          handleVoiceToggle();
        }
      }

      if ((e.key === 'r' || e.key === 'R') && !isTyping) {
        e.preventDefault();
        handleVoiceToggle();
      }

      if ((e.key === 'q' || e.key === 'Q') && !isTyping) {
        e.preventDefault();
        handleReadQuestion();
      }

      if ((e.key === 'f' || e.key === 'F') && !isTyping) {
        e.preventDefault();
        onFinish();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [answer, showVoiceModal, onFinish]);

  useEffect(() => {
    if (lastResult && lastQuestion) {
      const correctness = lastResult.correct ? 'correct' : 'incorrect';
      const correctAnswerText = lastResult.correct_answer ? `Correct answer: ${lastResult.correct_answer}.` : '';
      const confidenceText = lastResult.probability ? `Confidence ${Math.round(lastResult.probability * 100)} percent.` : '';
      const yourAnswerText = lastAnswer ? `Your answer: ${lastAnswer}.` : 'No answer captured.';
      speak(`Previous question feedback. You were ${correctness}. ${yourAnswerText} ${correctAnswerText} ${confidenceText}`, { interrupt: true });
    } else if (feedback) {
      speak(`Feedback: ${feedback}`, { interrupt: true });
    }
    return () => cancel();
  }, [lastResult, lastQuestion, lastAnswer, feedback, speak, cancel]);

  const handleReadQuestion = () => {
    cancel();
    speak(`Question ${questionNumber}. ${item.question}`);
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
      <div className="space-y-2" aria-live="polite">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Adaptive • Question {questionNumber}</span>
          <span>Answered: {answeredCount}</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted" aria-hidden>
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${Math.min(100, Math.max(10, answeredCount ? ((answeredCount) / (answeredCount + 1)) * 100 : 20))}%` }}
          />
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1"><Volume2 className="h-4 w-4" />Press Q to repeat</span>
          <span className="inline-flex items-center gap-1"><Mic className="h-4 w-4" />Press R to record</span>
          <span className="inline-flex items-center gap-1"><Flag className="h-4 w-4" />Press F to finish</span>
        </div>
      </div>

      <Card className="p-6 space-y-4" role="group" aria-labelledby="adaptive-question-heading">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 space-y-2">
            <p className="text-xs uppercase tracking-wide text-primary font-semibold">Difficulty: {item.difficulty_label}</p>
            <h2 id="adaptive-question-heading" className="text-xl font-semibold leading-relaxed">{item.question}</h2>
            {item.context && (
              <p className="text-sm text-muted-foreground">Context: {item.context}</p>
            )}
          </div>
          <div className="flex flex-col items-end gap-2 text-sm text-muted-foreground">
            <Button
              onClick={handleReadQuestion}
              variant="outline"
              size="icon"
              aria-label="Read question aloud"
            >
              <Volume2 className="h-5 w-5" />
            </Button>
            <div className="text-right">Ability θ: {theta.toFixed(2)}</div>
          </div>
        </div>
      </Card>

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

      <Card className="p-6 space-y-4">
        <label htmlFor="adaptive-answer" className="text-sm font-medium">Your answer</label>
        <Textarea
          id="adaptive-answer"
          value={answer}
          onChange={(e) => {
            setAnswer(e.target.value);
            setInputMode('text');
          }}
          placeholder={inputMode === 'voice' ? 'Tap microphone and speak your answer…' : 'Type your answer here…'}
          className="min-h-[140px] text-base"
        />
      </Card>

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
          className="min-h-[56px]"
        >
          {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Send className="mr-2 h-5 w-5" />}
          Submit Answer
        </Button>
        <Button
          variant="ghost"
          onClick={onFinish}
          size="lg"
          className="min-h-[56px]"
        >
          <Flag className="mr-2 h-5 w-5" />
          Finish Session
        </Button>
      </div>

      {(lastResult || feedback) && lastQuestion && (
        <Card className="p-6 bg-muted/60 border border-muted" aria-live="assertive">
          <div className="flex items-start gap-4">
            <div className="pt-1">
              {lastResult ? (
                lastResult.correct ? (
                  <CheckCircle2 className="h-8 w-8 text-green-500" />
                ) : (
                  <XCircle className="h-8 w-8 text-red-500" />
                )
              ) : (
                <AlertCircle className="h-8 w-8 text-blue-500" />
              )}
            </div>
            <div className="flex-1 space-y-2">
              <p className="text-sm uppercase tracking-wide text-primary font-semibold">Previous feedback</p>
              <p className="text-lg font-semibold">
                {lastResult ? (lastResult.correct ? 'Correct' : 'Incorrect') : 'Feedback'}
              </p>
              <p className="text-sm text-muted-foreground">Previous question: {lastQuestion.question}</p>
              <p className="text-sm">Your answer: {lastAnswer || 'No answer captured.'}</p>
              <p className="text-sm">Correct answer: {lastResult?.correct_answer || 'Not provided.'}</p>
              {lastResult?.probability !== undefined && (
                <p className="text-xs text-muted-foreground">Model confidence: {Math.round(lastResult.probability * 100)}%</p>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Speak feedback"
              onClick={() => {
                const correctness = lastResult?.correct ? 'correct' : 'incorrect';
                const correctAnswerText = lastResult?.correct_answer ? `Correct answer: ${lastResult.correct_answer}.` : '';
                const confidenceText = lastResult?.probability ? `Confidence ${Math.round(lastResult.probability * 100)} percent.` : '';
                const yourAnswerText = lastAnswer ? `Your answer: ${lastAnswer}.` : 'No answer captured.';
                speak(`Previous question feedback. You were ${correctness}. ${yourAnswerText} ${correctAnswerText} ${confidenceText}`, { interrupt: true });
              }}
            >
              <Play className="h-4 w-4" />
            </Button>
          </div>
        </Card>
      )}

      <VoiceRecorder
        isOpen={showVoiceModal}
        onClose={() => setShowVoiceModal(false)}
        onSubmit={handleVoiceSubmit}
        title="Record your answer"
        context={item.question}
      />
    </div>
  );
};
