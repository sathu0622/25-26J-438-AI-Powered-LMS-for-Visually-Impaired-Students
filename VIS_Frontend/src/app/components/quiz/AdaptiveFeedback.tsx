import { useEffect, useMemo } from 'react';
import { CheckCircle2, XCircle, AlertCircle, ArrowRight, Home, Play } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { AdaptiveAnswerResponse, AdaptiveItem } from '../../services/adaptiveService';
import { useTTS } from '../../contexts/TTSContext';

interface AdaptiveFeedbackProps {
  question: AdaptiveItem;
  answer: string;
  result: AdaptiveAnswerResponse;
  onNext: () => void;
  onFinish: () => void;
  isFinal?: boolean;
}

export const AdaptiveFeedback = ({
  question,
  answer,
  result,
  onNext,
  onFinish,
  isFinal = false,
}: AdaptiveFeedbackProps) => {
  const { speak, cancel } = useTTS();

  const scorePercent = useMemo(() => Math.round((result.probability || 0) * 100), [result.probability]);
  const isCorrect = result.correct;

  useEffect(() => {
    const correctAnswerText = result.correct_answer ? `Correct answer: ${result.correct_answer}.` : 'No model answer available.';
    const confidenceText = result.probability !== undefined ? `Confidence ${scorePercent} percent.` : '';
    const yourAnswerText = answer ? `Your answer: ${answer}.` : 'No answer captured.';
    speak(`Adaptive feedback. You were ${isCorrect ? 'correct' : 'incorrect'}. ${yourAnswerText} ${correctAnswerText} ${confidenceText}`, { interrupt: true });
    return () => cancel();
  }, [answer, result.correct, result.correct_answer, result.probability, scorePercent, isCorrect, speak, cancel]);

  const getIcon = () => {
    if (result.correct) return <CheckCircle2 className="h-12 w-12 text-green-500" />;
    return <XCircle className="h-12 w-12 text-red-500" />;
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="text-center">
          <p className="text-xs uppercase tracking-wide text-primary font-semibold">Adaptive feedback</p>
          <h1 className="text-3xl font-semibold">Your result</h1>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="p-8 flex flex-col justify-between space-y-6">
            <div className="rounded-xl p-6 text-center border border-muted bg-muted/40">
              <div className="flex justify-center mb-4">{getIcon()}</div>
              <h2 className="text-2xl font-bold mb-2">{isCorrect ? 'Correct' : 'Incorrect'}</h2>
              {result.probability !== undefined && (
                <p className="text-sm text-muted-foreground">Model confidence: {scorePercent}%</p>
              )}
              <div className="flex items-center justify-center gap-2 mt-2">
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak correctness"
                  onClick={() => speak(isCorrect ? 'Your answer is correct.' : 'Your answer is incorrect.', { interrupt: true })}
                >
                  <Play className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div className="flex flex-col gap-4 mt-4">
              <Button size="lg" className="font-semibold" onClick={onNext}>
                {isFinal ? 'View Summary' : 'Next Question'}
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button variant="outline" size="lg" className="font-semibold" onClick={onFinish}>
                <Home className="mr-2 h-5 w-5" /> End Adaptive Session
              </Button>
            </div>
          </Card>

          <div className="space-y-4">
            <Card className="p-6">
              <p className="text-sm font-semibold text-muted-foreground mb-2">Question</p>
              <p className="text-base leading-relaxed">{question.question}</p>
              {question.context && (
                <p className="text-sm text-muted-foreground mt-2">Context: {question.context}</p>
              )}
            </Card>

            <Card className="p-6 flex items-start justify-between">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-muted-foreground">Your answer</p>
                <p className="text-base">{answer || 'No answer provided.'}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Speak your answer"
                onClick={() => speak(answer ? `Your answer: ${answer}` : 'No answer provided.', { interrupt: true })}
              >
                <Play className="h-4 w-4" />
              </Button>
            </Card>

            <Card className="p-6 flex items-start justify-between bg-blue-50 border border-blue-200">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-blue-700">Model answer</p>
                <p className="text-base">{result.correct_answer || 'No model answer available.'}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Speak model answer"
                onClick={() => speak(result.correct_answer ? `Model answer: ${result.correct_answer}` : 'No model answer available.', { interrupt: true })}
              >
                <Play className="h-4 w-4" />
              </Button>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};
