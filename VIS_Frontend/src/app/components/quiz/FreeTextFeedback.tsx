import { useEffect } from 'react';
import { CheckCircle2, XCircle, ArrowRight, Home, Play, Loader2 } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { FreeTextQuestion, FreeTextAnswerResponse } from '../../services/freeTextService';
import { useTTS } from '../../contexts/TTSContext';

interface FreeTextFeedbackProps {
  question: FreeTextQuestion;
  userAnswer: string;
  result: FreeTextAnswerResponse;
  questionNumber: number;
  onNext: () => void;
  onFinish: () => void;
  isLoadingNext?: boolean;
}

export const FreeTextFeedback = ({
  question,
  userAnswer,
  result,
  questionNumber,
  onNext,
  onFinish,
  isLoadingNext = false,
}: FreeTextFeedbackProps) => {
  const { speak, cancel } = useTTS();

  const isCorrect = result.correct;
  const scorePercent = Math.round(result.score);

  // Announce feedback on mount
  useEffect(() => {
    const correctnessText = isCorrect ? 'correct' : 'incorrect';
    const feedbackText = `
      Free-text feedback for question ${questionNumber}.
      You were ${correctnessText} with a score of ${scorePercent} percent.
      Your answer: ${userAnswer || 'No answer provided'}.
      Correct answer: ${result.correct_answer}.
      ${result.feedback}.
      Press N for next question or F to finish the quiz.
    `;
    speak(feedbackText, { interrupt: true });
    return () => cancel();
  }, [question, userAnswer, result, questionNumber, isCorrect, scorePercent, speak, cancel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      
      if (key === 'n' || key === 'enter' || key === ' ') {
        e.preventDefault();
        if (!isLoadingNext) onNext();
      }
      
      if (key === 'f') {
        e.preventDefault();
        onFinish();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNext, onFinish, isLoadingNext]);

  const getIcon = () => {
    if (isCorrect) return <CheckCircle2 className="h-12 w-12 text-green-500" />;
    return <XCircle className="h-12 w-12 text-red-500" />;
  };

  const getScoreColor = () => {
    if (scorePercent >= 80) return 'text-green-600';
    if (scorePercent >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-5xl space-y-6">
        {/* Header */}
        <div className="text-center">
          <p className="text-xs uppercase tracking-wide text-green-600 font-semibold">
            Free-Text Quiz • Question {questionNumber} Feedback
          </p>
          <h1 className="text-3xl font-semibold">Your Result</h1>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Left: Result and Actions */}
          <Card className="p-8 flex flex-col justify-between space-y-6 border-green-200">
            <div className="rounded-xl p-6 text-center border border-muted bg-muted/40">
              <div className="flex justify-center mb-4">{getIcon()}</div>
              <h2 className="text-2xl font-bold mb-2">{isCorrect ? 'Correct!' : 'Incorrect'}</h2>
              <p className={`text-3xl font-bold ${getScoreColor()}`}>
                {scorePercent}%
              </p>
              <p className="text-sm text-muted-foreground mt-1">Semantic Similarity Score</p>
              <div className="flex items-center justify-center gap-2 mt-3">
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak result"
                  onClick={() => speak(
                    `You were ${isCorrect ? 'correct' : 'incorrect'} with a score of ${scorePercent} percent.`,
                    { interrupt: true }
                  )}
                >
                  <Play className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex flex-col gap-4 mt-4">
              <Button
                size="lg"
                className="font-semibold bg-green-600 hover:bg-green-700"
                onClick={onNext}
                disabled={isLoadingNext}
              >
                {isLoadingNext ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Generating Next...
                  </>
                ) : (
                  <>
                    Next Question
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </>
                )}
              </Button>
              <Button variant="outline" size="lg" className="font-semibold" onClick={onFinish}>
                <Home className="mr-2 h-5 w-5" /> Finish Quiz
              </Button>
            </div>

            {/* Keyboard hints */}
            <div className="text-center text-xs text-muted-foreground space-y-1">
              <p>Press <kbd className="px-1 py-0.5 bg-muted rounded">N</kbd> or <kbd className="px-1 py-0.5 bg-muted rounded">Enter</kbd> for next question</p>
              <p>Press <kbd className="px-1 py-0.5 bg-muted rounded">F</kbd> to finish quiz</p>
            </div>
          </Card>

          {/* Right: Question, Answer, Correct Answer */}
          <div className="space-y-4">
            {/* Question */}
            <Card className="p-6">
              <div className="flex items-start justify-between">
                <div className="space-y-1 flex-1">
                  <p className="text-sm font-semibold text-muted-foreground">Question</p>
                  <p className="text-base leading-relaxed">{question.question}</p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak question"
                  onClick={() => speak(`Question: ${question.question}`, { interrupt: true })}
                >
                  <Play className="h-4 w-4" />
                </Button>
              </div>
            </Card>

            {/* Your answer */}
            <Card className={`p-6 ${isCorrect ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
              <div className="flex items-start justify-between">
                <div className="space-y-1 flex-1">
                  <p className={`text-sm font-semibold ${isCorrect ? 'text-green-700' : 'text-red-700'}`}>
                    Your Answer
                  </p>
                  <p className="text-base">{userAnswer || 'No answer provided.'}</p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak your answer"
                  onClick={() => speak(
                    userAnswer ? `Your answer: ${userAnswer}` : 'No answer provided.',
                    { interrupt: true }
                  )}
                >
                  <Play className="h-4 w-4" />
                </Button>
              </div>
            </Card>

            {/* Correct answer */}
            <Card className="p-6 bg-blue-50 border border-blue-200">
              <div className="flex items-start justify-between">
                <div className="space-y-1 flex-1">
                  <p className="text-sm font-semibold text-blue-700">Correct Answer</p>
                  <p className="text-base">{result.correct_answer}</p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak correct answer"
                  onClick={() => speak(`Correct answer: ${result.correct_answer}`, { interrupt: true })}
                >
                  <Play className="h-4 w-4" />
                </Button>
              </div>
            </Card>

            {/* Feedback explanation */}
            {result.feedback && (
              <Card className="p-6 bg-yellow-50 border border-yellow-200">
                <div className="flex items-start justify-between">
                  <div className="space-y-1 flex-1">
                    <p className="text-sm font-semibold text-yellow-700">Feedback</p>
                    <p className="text-base italic">{result.feedback}</p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label="Speak feedback"
                    onClick={() => speak(`Feedback: ${result.feedback}`, { interrupt: true })}
                  >
                    <Play className="h-4 w-4" />
                  </Button>
                </div>
              </Card>
            )}
          </div>
        </div>

        {/* Loading indicator for background generation */}
        {isLoadingNext && (
          <div className="text-center text-sm text-muted-foreground animate-pulse">
            <Loader2 className="h-4 w-4 inline mr-2 animate-spin" />
            Generating next question in background...
          </div>
        )}
      </div>
    </div>
  );
};
