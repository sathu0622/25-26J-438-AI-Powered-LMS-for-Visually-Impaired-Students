import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Sparkles, Layers } from 'lucide-react';

interface QuizModeSelectProps {
  onSelectGenerative: () => void;
  onSelectAdaptive: () => void;
}

export const QuizModeSelect = ({ onSelectGenerative, onSelectAdaptive }: QuizModeSelectProps) => {
  return (
    <div className="mx-auto max-w-3xl p-4 space-y-6 pb-24">
      <h1 className="text-2xl text-center">Choose a Quiz Mode</h1>
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="p-6 space-y-4">
          <div className="flex items-center gap-3">
            <Sparkles className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Generative Quiz</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            On-the-fly LLM generated questions from the selected chapter. Same flow you already use.
          </p>
          <Button onClick={onSelectGenerative} className="w-full">Start Generative</Button>
        </Card>
        <Card className="p-6 space-y-4">
          <div className="flex items-center gap-3">
            <Layers className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Adaptive Quiz</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            Uses your curated item bank with adaptive difficulty (2PL). Questions get easier or harder as you answer.
          </p>
          <Button variant="outline" onClick={onSelectAdaptive} className="w-full">Start Adaptive</Button>
        </Card>
      </div>
    </div>
  );
};
