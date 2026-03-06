import { useEffect } from 'react';
import { Volume2, RefreshCw, Home, CheckCircle2, XCircle } from 'lucide-react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { useTTS } from '../../contexts/TTSContext';
import { FreeTextSummary as FreeTextSummaryType, FreeTextAnswerResponse } from '../../services/freeTextService';

interface FreeTextSummaryProps {
  summary: FreeTextSummaryType;
  answers: FreeTextAnswerResponse[];
  chapterName: string;
  onRestart: () => void;
  onRetake: () => void;
  onHome: () => void;
}

export const FreeTextSummary = ({
  summary,
  answers,
  chapterName,
  onRestart,
  onRetake,
  onHome,
}: FreeTextSummaryProps) => {
  const { speak, cancel } = useTTS();

  const summaryText = `Free-text quiz completed for ${chapterName}. You answered ${summary.correct_count} out of ${summary.total_questions} questions correctly. Your average score is ${summary.average_score} percent. Press R to retake the same questions, N for a new quiz, or H to go home.`;

  useEffect(() => {
    speak(summaryText, { interrupt: true });
    return () => cancel();
  }, [summaryText, speak, cancel]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'r') {
        e.preventDefault();
        onRetake();
      }
      if (e.key.toLowerCase() === 'n') {
        e.preventDefault();
        onRestart();
      }
      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        onHome();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onRetake, onRestart, onHome]);

  const getScoreColor = (score: number) => {
    if (score >= 85) return 'text-green-600';
    if (score >= 70) return 'text-blue-600';
    if (score >= 55) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="mx-auto max-w-3xl p-6 space-y-6 pb-24" aria-live="polite">
      {/* Summary header */}
      <Card className="p-8 space-y-4 border-green-200">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wide text-green-600 font-semibold">Free-Text Quiz Complete</p>
            <h2 className="text-2xl font-semibold">{chapterName}</h2>
            <p className="text-sm text-muted-foreground">Session completed successfully. Your answers have been saved for retake.</p>
          </div>
          <Button variant="ghost" size="icon" aria-label="Speak summary" onClick={() => speak(summaryText)}>
            <Volume2 className="h-5 w-5" />
          </Button>
        </div>

        {/* Stats cards */}
        <div className="grid sm:grid-cols-3 gap-3">
          <Card className="p-4 text-center border-muted bg-muted/40">
            <p className="text-xs text-muted-foreground">Questions</p>
            <p className="text-2xl font-semibold">{summary.total_questions}</p>
          </Card>
          <Card className="p-4 text-center border-muted bg-muted/40">
            <p className="text-xs text-muted-foreground">Correct</p>
            <p className="text-2xl font-semibold text-green-600">{summary.correct_count}</p>
          </Card>
          <Card className="p-4 text-center border-muted bg-muted/40">
            <p className="text-xs text-muted-foreground">Average Score</p>
            <p className={`text-2xl font-semibold ${getScoreColor(summary.average_score)}`}>
              {summary.average_score}%
            </p>
          </Card>
        </div>
      </Card>

      {/* Answer review */}
      {answers.length > 0 && (
        <Card className="p-6 space-y-4">
          <h3 className="font-semibold">Answer Review</h3>
          <div className="space-y-3 max-h-[300px] overflow-y-auto">
            {answers.map((ans, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg border ${ans.correct ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}
              >
                <div className="flex items-start gap-3">
                  {ans.correct ? (
                    <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-medium">Q{index + 1}: {ans.question_index !== undefined ? `Question ${ans.question_index + 1}` : ''}</p>
                    <p className="text-sm">
                      <span className="text-muted-foreground">Your answer:</span> {ans.user_answer || 'No answer'}
                    </p>
                    <p className="text-sm">
                      <span className="text-muted-foreground">Correct answer:</span> {ans.correct_answer}
                    </p>
                    <p className={`text-sm font-medium ${getScoreColor(ans.score)}`}>
                      Score: {ans.score}%
                    </p>
                    <p className="text-xs text-muted-foreground italic">{ans.feedback}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Action buttons */}
      <div className="grid gap-3 md:grid-cols-3">
        <Button
          onClick={onRetake}
          className="flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700"
        >
          <RefreshCw className="h-4 w-4" /> Retake Same Questions (R)
        </Button>
        <Button
          onClick={onRestart}
          variant="outline"
          className="flex items-center justify-center gap-2"
        >
          <RefreshCw className="h-4 w-4" /> New Quiz (N)
        </Button>
        <Button
          onClick={onHome}
          variant="outline"
          className="flex items-center justify-center gap-2"
        >
          <Home className="h-4 w-4" /> Back to Home (H)
        </Button>
      </div>

      {/* Help text */}
      <div className="text-center text-xs text-muted-foreground">
        <p>Press R to retake • N for new quiz • H to go home</p>
      </div>
    </div>
  );
};
