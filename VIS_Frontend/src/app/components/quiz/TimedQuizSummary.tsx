import { useCallback, useEffect, useMemo } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { CheckCircle2, XCircle, Home, Volume2 } from 'lucide-react';
import { useTTS } from '../../contexts/TTSContext';
import type { TimedQuizEvaluateResponse } from '../../services/timedQuizService';

interface TimedQuizSummaryProps {
  summary: TimedQuizEvaluateResponse;
  onHome: () => void;
}

function buildTimedQuizNarrationPhrases(summary: TimedQuizEvaluateResponse): string[] {
  const header = `Timed quiz complete. You answered ${summary.correct_count} out of ${summary.total_questions} correctly. Your score is ${summary.average_score} percent. Beginning question-by-question review.`;
  const qs = summary.results.map((r, idx) => {
    const verdict = r.correct ? 'Correct.' : 'Incorrect.';
    const yours = `Your answer: ${r.selected_text.trim() ? r.selected_text : 'none'}.`;
    const corr = r.correct ? '' : ` The correct answer is: ${r.correct_answer}.`;
    return `Question ${idx + 1}, ${r.chapter_name}. ${r.question}. ${verdict} ${yours}${corr}`;
  });
  const footer =
    'End of review. Use the button below when you are ready to return to quiz modes.';
  return [header, ...qs, footer];
}

export function TimedQuizSummary({ summary, onHome }: TimedQuizSummaryProps) {
  const { speak, cancel } = useTTS();

  const phrases = useMemo(() => buildTimedQuizNarrationPhrases(summary), [summary]);

  const speakRecap = useCallback(
    (parts: string[]) => {
      cancel();
      let i = 0;
      const run = () => {
        if (i >= parts.length) return;
        const first = i === 0;
        speak(parts[i], {
          interrupt: first,
          onEnd: () => {
            i += 1;
            run();
          },
        });
      };
      run();
    },
    [speak, cancel]
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
    return () => {
      stopped = true;
      cancel();
    };
  }, [phrases, speak, cancel]);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 pb-24" aria-live="polite">
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-semibold">Timed quiz results</h1>
        <p className="text-muted-foreground">
          {summary.correct_count} / {summary.total_questions} correct ({summary.average_score}%)
        </p>
      </div>

      <Card className="p-6 space-y-4">
        <h2 className="font-semibold">Question-by-question review</h2>
        <ul className="space-y-4 list-none">
          {summary.results.map((r, idx) => (
            <li key={r.item_id} className="border-b border-border pb-4 last:border-0 last:pb-0">
              <div className="flex items-start gap-2">
                {r.correct ? (
                  <CheckCircle2 className="h-5 w-5 text-green-600 shrink-0 mt-0.5" aria-label="Correct" />
                ) : (
                  <XCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" aria-label="Incorrect" />
                )}
                <div className="flex-1 space-y-1">
                  <p className="text-sm text-muted-foreground">
                    Question {idx + 1} · {r.chapter_name}
                  </p>
                  <p className="font-medium">{r.question}</p>
                  <p className="text-sm">
                    Your answer: {r.selected_text || '(no answer)'}
                  </p>
                  {!r.correct && (
                    <p className="text-sm text-green-700 dark:text-green-400">
                      Correct answer: {r.correct_answer}
                    </p>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      </Card>

      <Button
        type="button"
        variant="outline"
        className="w-full min-h-[48px] gap-2"
        aria-label="Read full results aloud again"
        onClick={() => speakRecap(phrases)}
      >
        <Volume2 className="h-4 w-4" aria-hidden />
        Play full recap
      </Button>

      <Button className="w-full min-h-[48px] gap-2" onClick={onHome}>
        <Home className="h-4 w-4" aria-hidden />
        Back to quiz modes
      </Button>
    </div>
  );
}
