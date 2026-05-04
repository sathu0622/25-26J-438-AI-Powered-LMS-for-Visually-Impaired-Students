import { useCallback, useEffect, useMemo } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { CheckCircle2, RefreshCw, Home, BookOpen } from 'lucide-react';
import { useTTS } from '../../contexts/TTSContext';
import { QuizSetSummary } from '../../services/quizService';

export interface QuizSummaryReviewItem {
  question: string;
  yourAnswer: string;
  correctAnswer: string;
  correct: boolean;
  year?: string;
}

interface QuizSummaryProps {
  summary: QuizSetSummary;
  correctCount: number;
  totalQuestions: number;
  onRetake: () => void;
  onGoHome: () => void;
  onStartNew: () => void;
  /** When set (e.g. past paper), read each question aloud with verdict and answers. */
  reviewItems?: QuizSummaryReviewItem[];
}

function buildQuizSummaryPhrases(
  correctCount: number,
  totalQuestions: number,
  averageScore: number,
  accuracy: number,
  reviewItems?: QuizSummaryReviewItem[]
): string[] {
  const head = `Quiz complete. You answered ${correctCount} out of ${totalQuestions} correctly. Accuracy ${accuracy} percent. Average score ${averageScore} percent.`;
  const shortcuts = `Press Enter to retake this set, H for home, or N to start a new set.`;
  if (!reviewItems?.length) {
    return [`${head} ${shortcuts}`];
  }
  const detail = reviewItems.map((item, idx) => {
    const verdict = item.correct ? 'Correct.' : 'Incorrect.';
    const yr = item.year ? ` Year ${item.year}.` : '';
    const yours = `Your answer: ${item.yourAnswer?.trim() ? item.yourAnswer : 'none'}.`;
    const corr = item.correct
      ? ''
      : ` The correct answer was: ${item.correctAnswer}.`;
    return `Question ${idx + 1}.${yr} ${item.question}. ${verdict} ${yours}${corr}`;
  });
  return [head + ' Beginning question-by-question review.', ...detail, `End of review. ${shortcuts}`];
}

export const QuizSummary = ({
  summary,
  correctCount,
  totalQuestions,
  onRetake,
  onGoHome,
  onStartNew,
  reviewItems,
}: QuizSummaryProps) => {
  const { speak, cancel } = useTTS();

  const accuracy =
    totalQuestions > 0 ? Math.round((correctCount / totalQuestions) * 100) : 0;

  const phrases = useMemo(
    () =>
      buildQuizSummaryPhrases(
        correctCount,
        totalQuestions,
        summary.average_score,
        accuracy,
        reviewItems
      ),
    [
      correctCount,
      totalQuestions,
      summary.average_score,
      accuracy,
      reviewItems,
    ]
  );

  useEffect(() => {
    let stopped = false;
    cancel();
    let i = 0;
    const run = () => {
      if (stopped || i >= phrases.length) return;
      speak(phrases[i], {
        interrupt: i === 0,
        onEnd: () => {
          if (stopped) return;
          i += 1;
          run();
        },
      });
    };
    run();

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
      stopped = true;
      window.removeEventListener('keydown', handleKeys);
      cancel();
    };
  }, [phrases, speak, cancel, onRetake, onGoHome, onStartNew]);

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
    </div>
  );
};
