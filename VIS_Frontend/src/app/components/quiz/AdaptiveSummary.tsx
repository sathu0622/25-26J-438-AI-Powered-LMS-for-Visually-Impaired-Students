import { useEffect } from 'react';
import { Volume2, RefreshCw, Home } from 'lucide-react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { useTTS } from '../../contexts/TTSContext';

interface AdaptiveSummaryProps {
  correctCount: number;
  total: number;
  finalTheta: number;
  onRestart: () => void;
  onHome: () => void;
}

export const AdaptiveSummary = ({ correctCount, total, onRestart, onHome }: AdaptiveSummaryProps) => {
  const { speak, cancel } = useTTS();
  const summaryText = `You answered ${correctCount} out of ${total} questions.`;
  const accuracyPct = total > 0 ? Math.round((correctCount / total) * 100) : 0;

  useEffect(() => {
    speak(summaryText, { interrupt: true });
    return () => cancel();
  }, [summaryText, speak, cancel]);

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 p-6 pb-24" aria-live="polite">
      <Card className="p-6 sm:p-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-primary">Adaptive Summary</p>
            <h2 className="text-2xl font-semibold leading-tight">Session complete</h2>
            <p className="text-sm text-muted-foreground">
              Performance overview for this adaptive run.
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Speak summary"
            onClick={() => speak(summaryText)}
            className="self-start"
          >
            <Volume2 className="h-5 w-5" />
          </Button>
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg border border-muted bg-muted/40 p-4 text-center">
            <p className="text-xs text-muted-foreground">Questions answered</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums">{total}</p>
          </div>
          <div className="rounded-lg border border-muted bg-muted/40 p-4 text-center">
            <p className="text-xs text-muted-foreground">Correct</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums">{correctCount}</p>
            <p className="mt-1 text-xs text-muted-foreground tabular-nums">{accuracyPct}% accuracy</p>
          </div>
        </div>
      </Card>

      <div className="grid gap-3 md:grid-cols-2">
        <Button onClick={onRestart} className="min-h-[56px] w-full items-center justify-center gap-2">
          <RefreshCw className="h-4 w-4" /> Restart Adaptive Quiz
        </Button>
        <Button variant="outline" onClick={onHome} className="min-h-[56px] w-full items-center justify-center gap-2">
          <Home className="h-4 w-4" /> Back to Quiz Home
        </Button>
      </div>
    </div>
  );
};
