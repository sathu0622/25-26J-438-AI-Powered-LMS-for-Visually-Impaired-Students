import { useEffect, useState } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Play, History, ArrowLeft, RefreshCw } from 'lucide-react';
import { useTTS } from '../../contexts/TTSContext';
import { QuizSetListItem } from '../../services/quizService';

interface QuizDashboardProps {
  sets: QuizSetListItem[];
  onRetake: (setId: string, chapter: string) => void;
  onBack: () => void;
}

export const QuizDashboard = ({ sets, onRetake, onBack }: QuizDashboardProps) => {
  const { speak, cancel } = useTTS();
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    cancel();
    const intro =
      sets.length === 0
        ? 'No saved quiz sets yet. Press back to start a new one.'
        : 'Saved quiz sets loaded. Use up and down arrows to select. Press Enter to retake, Backspace to return.';
    speak(intro, { interrupt: true });

    const handleKeys = (e: KeyboardEvent) => {
      if (sets.length === 0) {
        if (e.key === 'Backspace' || e.key === 'Escape') {
          onBack();
        }
        return;
      }

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelected((prev) => (prev - 1 + sets.length) % sets.length);
        speak(sets[(selected - 1 + sets.length) % sets.length].chapter_name, { interrupt: true });
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelected((prev) => (prev + 1) % sets.length);
        speak(sets[(selected + 1) % sets.length].chapter_name, { interrupt: true });
      }

      if (e.key === 'Enter') {
        const item = sets[selected];
        if (item) onRetake(item.set_id, item.chapter_name);
      }

      if (e.key === 'Backspace' || e.key === 'Escape') {
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeys);
    return () => {
      window.removeEventListener('keydown', handleKeys);
      cancel();
    };
  }, [sets, selected, onRetake, onBack, speak, cancel]);

  return (
    <div className="mx-auto max-w-4xl space-y-4 p-6 pb-20">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Saved Quiz Sets</h1>
        <Button variant="ghost" onClick={onBack} aria-label="Go back to quiz start">
          <ArrowLeft className="mr-2 h-4 w-4" /> Back
        </Button>
      </div>
      {sets.length === 0 && (
        <Card className="p-6 text-center">
          <p className="text-muted-foreground">No saved quiz sets yet. Complete a quiz to save it.</p>
        </Card>
      )}

      <div className="grid gap-3">
        {sets.map((set, index) => {
          const isActive = index === selected;
          const summary = set.latest_attempt?.summary;
          return (
            <Card
              key={set.set_id}
              className={`p-4 flex items-center justify-between ${
                isActive ? 'border-2 border-primary bg-primary/5' : ''
              }`}
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <History className="h-4 w-4 text-primary" />
                  <span className="font-semibold">{set.chapter_name}</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  {summary
                    ? `${summary.correct_count}/${summary.total_questions} correct • Avg ${summary.average_score}%`
                    : 'No attempts yet'}
                </p>
                {set.created_at && (
                  <p className="text-xs text-muted-foreground">Created {set.created_at}</p>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  onClick={() => onRetake(set.set_id, set.chapter_name)}
                  aria-label={`Retake quiz set for ${set.chapter_name}`}
                >
                  <RefreshCw className="mr-2 h-4 w-4" /> Retake
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak stats"
                  onClick={() =>
                    speak(
                      summary
                        ? `${summary.correct_count} of ${summary.total_questions} correct. Average ${summary.average_score} percent.`
                        : 'No attempts yet'
                    )
                  }
                >
                  <Play className="h-4 w-4" />
                </Button>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
};
