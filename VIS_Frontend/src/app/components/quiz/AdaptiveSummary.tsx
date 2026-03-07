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

export const AdaptiveSummary = ({ correctCount, total, finalTheta, onRestart, onHome }: AdaptiveSummaryProps) => {
  const { speak, cancel } = useTTS();
  const summaryText = `You answered ${correctCount} out of ${total} questions. Estimated ability theta is ${finalTheta.toFixed(2)}.`;

  useEffect(() => {
    speak(summaryText, { interrupt: true });
    return () => cancel();
  }, [summaryText, speak, cancel]);

  return (
    <div className="mx-auto max-w-3xl p-6 space-y-6 pb-24" aria-live="polite">
      <Card className="p-8 space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wide text-primary font-semibold">Adaptive Summary</p>
            <h2 className="text-2xl font-semibold">Session complete</h2>
            <p className="text-sm text-muted-foreground">Performance overview for this adaptive run.</p>
          </div>
          <Button variant="ghost" size="icon" aria-label="Speak summary" onClick={() => speak(summaryText)}>
            <Volume2 className="h-5 w-5" />
          </Button>
        </div>

        <div className="grid sm:grid-cols-3 gap-3">
          <Card className="p-4 text-center border-muted bg-muted/40">
            <p className="text-xs text-muted-foreground">Questions answered</p>
            <p className="text-2xl font-semibold">{total}</p>
          </Card>
          <Card className="p-4 text-center border-muted bg-muted/40">
            <p className="text-xs text-muted-foreground">Correct</p>
            <p className="text-2xl font-semibold">{correctCount}</p>
          </Card>
          <Card className="p-4 text-center border-muted bg-muted/40">
            <p className="text-xs text-muted-foreground">Ability θ</p>
            <p className="text-2xl font-semibold">{finalTheta.toFixed(2)}</p>
          </Card>
        </div>
      </Card>

      <div className="grid gap-3 md:grid-cols-2">
        <Button onClick={onRestart} className="flex items-center justify-center gap-2">
          <RefreshCw className="h-4 w-4" /> Restart Adaptive Quiz
        </Button>
        <Button variant="outline" onClick={onHome} className="flex items-center justify-center gap-2">
          <Home className="h-4 w-4" /> Back to Quiz Home
        </Button>
      </div>
    </div>
  );
};
