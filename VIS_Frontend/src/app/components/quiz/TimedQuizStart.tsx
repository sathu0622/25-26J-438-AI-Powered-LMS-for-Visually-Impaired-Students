import { useEffect } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Clock, BookOpen, ArrowLeft } from 'lucide-react';
import { useTTS } from '../../contexts/TTSContext';

interface TimedQuizStartProps {
  username: string;
  onStart: () => void;
  onBack: () => void;
}

export function TimedQuizStart({ username, onStart, onBack }: TimedQuizStartProps) {
  const { speak, cancel } = useTTS();

  useEffect(() => {
    cancel();
    const msg = `
      Timed quiz. Welcome ${username}.
      This quiz pulls twenty multiple choice questions from the official question bank across different chapters.
      You have thirty minutes once you start. The timer begins when you press Start timed quiz.
      Each question has four choices: one correct and three incorrect answers.
      You will see your score and a review of all twenty questions when you finish or when time runs out.
      Press Start to begin, or Back to return to quiz modes.
    `;
    speak(msg, { interrupt: true });
    return () => cancel();
  }, [username, speak, cancel]);

  return (
    <main className="mx-auto max-w-2xl space-y-6 p-4 pb-24" role="main">
      <header className="space-y-2 text-center">
        <h1 className="text-2xl font-semibold">Timed quiz</h1>
        <p className="text-muted-foreground text-sm">
          20 questions from the dataset · 30 minutes · MCQ · Results at the end
        </p>
      </header>

      <Card className="p-6 space-y-4">
        <div className="flex items-start gap-3">
          <BookOpen className="h-8 w-8 text-amber-600 shrink-0" aria-hidden />
          <div>
            <h2 className="font-medium">Mixed chapters</h2>
            <p className="text-sm text-muted-foreground">
              Questions are taken from the syllabus question bank. They are not AI-generated; wrong options
              are produced as distractors for each item.
            </p>
          </div>
        </div>
        <div className="flex items-start gap-3">
          <Clock className="h-8 w-8 text-amber-600 shrink-0" aria-hidden />
          <div>
            <h2 className="font-medium">30-minute limit</h2>
            <p className="text-sm text-muted-foreground">
              The countdown starts as soon as you begin. When it reaches zero, your answers so far are submitted
              automatically and scored.
            </p>
          </div>
        </div>
      </Card>

      <div className="flex flex-col gap-3 sm:flex-row">
        <Button className="flex-1 min-h-[48px]" onClick={onStart}>
          Start timed quiz
        </Button>
        <Button variant="outline" className="flex-1 min-h-[48px] gap-2" onClick={onBack}>
          <ArrowLeft className="h-4 w-4" aria-hidden />
          Back
        </Button>
      </div>
    </main>
  );
}
