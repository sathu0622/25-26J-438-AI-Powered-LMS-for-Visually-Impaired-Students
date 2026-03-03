import { useEffect } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { CheckCircle2, RefreshCw, Home, Play, BookOpen } from 'lucide-react';
import { useTTS } from '../../contexts/TTSContext';
import { QuizSetSummary } from '../../services/quizService';

interface QuizSummaryProps {
  summary: QuizSetSummary;
  correctCount: number;
  totalQuestions: number;
  onRetake: () => void;
  onGoHome: () => void;
  onStartNew: () => void;
}

export const QuizSummary = ({
  summary,
  correctCount,
  totalQuestions,
  onRetake,
  onGoHome,
  onStartNew,
}: QuizSummaryProps) => {
  const { speak, cancel } = useTTS();

  useEffect(() => {
    cancel();
    speak(
      `Quiz complete. You answered ${correctCount} out of ${totalQuestions} correctly. Average score ${summary.average_score} percent. Press Enter to retake this set, H for home, or N to start a new set.`,
      { interrupt: true }
    );

    const handleKeys = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        onRetake();
      }
      if (e.key === 'h' || e.key === 'H') {
        onGoHome();
      }
      if (e.key === 'n' || e.key === 'N') {
        onStartNew();
      }
    };

    window.addEventListener('keydown', handleKeys);
    return () => {
      window.removeEventListener('keydown', handleKeys);
      cancel();
    };
  }, [correctCount, totalQuestions, summary.average_score, onRetake, onGoHome, onStartNew, speak, cancel]);

  const accuracy = Math.round((correctCount / totalQuestions) * 100);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6 pb-24">
      <div className="text-center space-y-2">
        <div className="flex justify-center">
          <div className="rounded-full bg-green-500 p-5">
            <CheckCircle2 className="h-10 w-10 text-white" />
          </div>
        </div>
        <h1 className="text-2xl font-semibold">Quiz Summary</h1>
        <p className="text-muted-foreground">Keyboard: Enter retake • H home • N new set</p>
      </div>

      <Card className="p-6 space-y-4">
        <div className="flex justify-between text-lg font-semibold">
          <span>Correct Answers</span>
          <span>
            {correctCount} / {totalQuestions}
          </span>
        </div>
        <div className="flex justify-between text-lg font-semibold">
          <span>Accuracy</span>
          <span>{accuracy}%</span>
        </div>
        <div className="flex justify-between text-lg font-semibold">
          <span>Average Score</span>
          <span>{summary.average_score}%</span>
        </div>
      </Card>

      <div className="grid gap-3 md:grid-cols-3">
        <Button size="lg" className="min-h-[56px]" onClick={onRetake}>
          <RefreshCw className="mr-2 h-5 w-5" /> Retake This Set
        </Button>
        <Button size="lg" variant="secondary" className="min-h-[56px]" onClick={onStartNew}>
          <BookOpen className="mr-2 h-5 w-5" /> Start New Set
        </Button>
        <Button size="lg" variant="outline" className="min-h-[56px]" onClick={onGoHome}>
          <Home className="mr-2 h-5 w-5" /> Back to Dashboard
        </Button>
      </div>

      <Card className="p-6 flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-muted-foreground mb-2">Audio Recap</p>
          <p className="text-base">You can replay the summary.</p>
        </div>
        <Button
          variant="secondary"
          onClick={() => speak(`You answered ${correctCount} out of ${totalQuestions} correctly. Average score ${summary.average_score} percent.`)}
        >
          <Play className="mr-2 h-4 w-4" /> Play
        </Button>
      </Card>
    </div>
  );
};
