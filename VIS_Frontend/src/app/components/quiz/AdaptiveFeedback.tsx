import { useEffect, useMemo } from 'react';
import { CheckCircle2, XCircle, AlertCircle, ArrowRight, Home, Play } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { AdaptiveAnswerResponse, AdaptiveItem } from '../../services/adaptiveService';
import { useTTS } from '../../contexts/TTSContext';

interface AdaptiveFeedbackProps {
  question: AdaptiveItem;
  answer: string;
  result: AdaptiveAnswerResponse;
  onNext: () => void;
  onFinish: () => void;
  onBack?: () => void;
  isFinal?: boolean;
}

export const AdaptiveFeedback = ({
  question,
  answer,
  result,
  onNext,
  onFinish,
  onBack,
  isFinal = false,
}: AdaptiveFeedbackProps) => {
  const { speak, cancel } = useTTS();

  const scorePercent = useMemo(() => Math.round((result.probability || 0) * 100), [result.probability]);
  const isCorrect = result.correct;

  // Determine difficulty level transition
  const currentDifficulty = question.difficulty_label?.toLowerCase() || 'medium';
  const nextDifficulty = result.next_item?.difficulty_label?.toLowerCase() || currentDifficulty;
  
  const difficultyOrder = ['easy', 'medium', 'hard'];
  const currentIndex = difficultyOrder.indexOf(currentDifficulty);
  const nextIndex = difficultyOrder.indexOf(nextDifficulty);
  
  const getDifficultyTransitionMessage = () => {
    if (isFinal || !result.next_item) return '';
    
    if (nextIndex > currentIndex) {
      // Upgrading to higher difficulty
      return `Congratulations! You are advancing to ${result.next_item.difficulty_label} difficulty level.`;
    } else if (nextIndex < currentIndex) {
      // Downgrading to lower difficulty
      return `The system is adjusting to ${result.next_item.difficulty_label} difficulty to help you learn better.`;
    } else if (!isCorrect && currentIndex > 0) {
      // Staying at same level after incorrect answer (struggling)
      return `Keep practicing! You are staying at ${currentDifficulty} difficulty.`;
    }
    return '';
  };

  // Keyboard shortcuts: N for next, Backspace/B for back, F for finish
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      
      // N or Enter or Space for next
      if (key === 'n' || key === 'enter' || key === ' ') {
        e.preventDefault();
        onNext();
      }
      
      // Backspace or B for back
      if (e.key === 'Backspace' || key === 'b') {
        e.preventDefault();
        if (onBack) {
          onBack();
        } else {
          onFinish();
        }
      }
      
      // F for finish
      if (key === 'f') {
        e.preventDefault();
        onFinish();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNext, onFinish, onBack]);

  useEffect(() => {
    const correctAnswerText = result.correct_answer ? `Correct answer: ${result.correct_answer}.` : 'No model answer available.';
    const yourAnswerText = answer ? `Your answer: ${answer}.` : 'No answer captured.';
    const difficultyMessage = getDifficultyTransitionMessage();
    const navigationHint = isFinal ? 'Press N or Enter to view summary.' : 'Press N for next question. Press F to finish.';
    
    speak(
      `You were ${isCorrect ? 'correct' : 'incorrect'}. ${yourAnswerText} ${correctAnswerText} ${difficultyMessage} ${navigationHint}`,
      { interrupt: true }
    );
    return () => cancel();
  }, [answer, result.correct, result.correct_answer, result.next_item, isFinal, isCorrect, speak, cancel]);

  const getIcon = () => {
    if (result.correct) return <CheckCircle2 className="h-12 w-12 text-green-500" />;
    return <XCircle className="h-12 w-12 text-red-500" />;
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="text-center">
          <p className="text-xs uppercase tracking-wide text-primary font-semibold">Adaptive feedback</p>
          <h1 className="text-3xl font-semibold">Your result</h1>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="p-8 flex flex-col justify-between space-y-6">
            <div className="rounded-xl p-6 text-center border border-muted bg-muted/40">
              <div className="flex justify-center mb-4">{getIcon()}</div>
              <h2 className="text-2xl font-bold mb-2">{isCorrect ? 'Correct' : 'Incorrect'}</h2>
              
              {/* Difficulty transition message */}
              {getDifficultyTransitionMessage() && (
                <div className={`mt-3 p-2 rounded-lg text-sm font-medium ${
                  nextIndex > currentIndex 
                    ? 'bg-green-100 text-green-700 border border-green-300' 
                    : nextIndex < currentIndex 
                      ? 'bg-yellow-100 text-yellow-700 border border-yellow-300'
                      : 'bg-blue-100 text-blue-700 border border-blue-300'
                }`}>
                  {getDifficultyTransitionMessage()}
                </div>
              )}
              
              <div className="flex items-center justify-center gap-2 mt-2">
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak correctness"
                  onClick={() => speak(isCorrect ? 'Your answer is correct.' : 'Your answer is incorrect.', { interrupt: true })}
                >
                  <Play className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Keyboard shortcuts hint */}
            <div className="text-xs text-muted-foreground text-center mt-2">
              Press N for next question • Press F to finish • Press B to go back
            </div>

            <div className="flex flex-col gap-4 mt-4">
              <Button size="lg" className="font-semibold" onClick={onNext}>
                {isFinal ? 'View Summary' : 'Next Question (N)'}
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button variant="outline" size="lg" className="font-semibold" onClick={onFinish}>
                <Home className="mr-2 h-5 w-5" /> End Adaptive Session (F)
              </Button>
            </div>
          </Card>

          <div className="space-y-4">
            <Card className="p-6">
              <p className="text-sm font-semibold text-muted-foreground mb-2">Question</p>
              <p className="text-base leading-relaxed">{question.question}</p>
              
            </Card>

            <Card className="p-6 flex items-start justify-between">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-muted-foreground">Your answer</p>
                <p className="text-base">{answer || 'No answer provided.'}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Speak your answer"
                onClick={() => speak(answer ? `Your answer: ${answer}` : 'No answer provided.', { interrupt: true })}
              >
                <Play className="h-4 w-4" />
              </Button>
            </Card>

            <Card className="p-6 flex items-start justify-between bg-blue-50 border border-blue-200">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-blue-700">Model answer</p>
                <p className="text-base">{result.correct_answer || 'No model answer available.'}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Speak model answer"
                onClick={() => speak(result.correct_answer ? `Model answer: ${result.correct_answer}` : 'No model answer available.', { interrupt: true })}
              >
                <Play className="h-4 w-4" />
              </Button>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};
