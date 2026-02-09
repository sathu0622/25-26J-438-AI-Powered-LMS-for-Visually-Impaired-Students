import { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, AlertCircle, ArrowRight, Loader2 } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { AudioPlayer } from '../AudioPlayer';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface QuizFeedbackProps {
  question: string;
  answer: string;
  expectedAnswer?: string;
  feedback?: string;
  onNext: () => void;
}

type FeedbackType = 'correct' | 'partial' | 'incorrect';

interface FeedbackData {
  type: FeedbackType;
  score: number;
  explanation: string;
}

export const QuizFeedback = ({ question, answer, expectedAnswer, feedback: providedFeedback, onNext }: QuizFeedbackProps) => {
  const [feedback, setFeedback] = useState<FeedbackData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasAnnounced, setHasAnnounced] = useState(false);

  useEffect(() => {
    // STOP all previous speech immediately
    safeCancel();
    
    // Simulate AI evaluation with actual quiz data if available
    setTimeout(() => {
      // Calculate similarity score (mock)
      const similarityScore = Math.floor(Math.random() * 30) + 70;
      
      const mockFeedback: FeedbackData = {
        type: similarityScore >= 85 ? 'correct' : similarityScore >= 60 ? 'partial' : 'incorrect',
        score: similarityScore,
        explanation: providedFeedback || 'Your answer demonstrates a good understanding of the concept. You correctly identified the key points about the topic.',
      };
      setFeedback(mockFeedback);
      setIsLoading(false);
    }, 2000);
    
    // Cleanup: stop speech when leaving page
    return () => {
      safeCancel();
    };
  }, [question, answer, providedFeedback]);

  // Auto-announce feedback after loading
  useEffect(() => {
    if (!isLoading && feedback && !hasAnnounced) {
      setHasAnnounced(true);
      setTimeout(() => {
        const audioText = `${getTitle()} You scored ${feedback.score} percent. ${feedback.explanation}`;
        safeSpeak(audioText, () => {
          // After feedback, announce next step
          setTimeout(() => {
            safeSpeak('Press F to replay feedback. Press Space, Enter, or N for next question.');
          }, 1000);
        });
      }, 500);
    }
  }, [isLoading, feedback, hasAnnounced]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isLoading && feedback) {
      const handleKeyPress = (e: KeyboardEvent) => {
        // F key to replay feedback
        if (e.key === 'f' || e.key === 'F') {
          e.preventDefault();
          const audioText = `${getTitle()} You scored ${feedback.score} percent. ${feedback.explanation}`;
          safeCancel();
          safeSpeak(audioText);
        }

        // N key or Space/Enter to go to next question
        if (e.key === 'n' || e.key === 'N' || e.key === ' ' || e.key === 'Enter') {
          e.preventDefault();
          onNext();
        }
      };

      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, [isLoading, feedback, onNext]);

  const getIcon = () => {
    if (!feedback) return null;
    switch (feedback.type) {
      case 'correct':
        return <CheckCircle2 className="h-16 w-16 text-success" />;
      case 'partial':
        return <AlertCircle className="h-16 w-16 text-warning" />;
      case 'incorrect':
        return <XCircle className="h-16 w-16 text-destructive" />;
    }
  };

  const getColor = () => {
    if (!feedback) return '';
    switch (feedback.type) {
      case 'correct':
        return 'border-success bg-success/10';
      case 'partial':
        return 'border-warning bg-warning/10';
      case 'incorrect':
        return 'border-destructive bg-destructive/10';
    }
  };

  const getTitle = () => {
    if (!feedback) return '';
    switch (feedback.type) {
      case 'correct':
        return 'Excellent!';
      case 'partial':
        return 'Good Effort!';
      case 'incorrect':
        return 'Keep Learning!';
    }
  };

  if (isLoading) {
    return (
      <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
        <div className="space-y-2 text-center">
          <h1 className="text-2xl">Evaluating Answer</h1>
          <p className="text-muted-foreground">AI is analyzing your response...</p>
        </div>
        <Card className="p-12">
          <div className="flex flex-col items-center gap-4 text-center">
            <Loader2 className="h-12 w-12 animate-spin text-primary" />
            <p>Please wait...</p>
          </div>
        </Card>
      </div>
    );
  }

  if (!feedback) return null;

  const audioText = `${getTitle()} You scored ${feedback.score} percent. ${feedback.explanation}`;

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-2 text-center">
        <h1 className="text-2xl">Feedback</h1>
        <p className="text-muted-foreground">
          Press F to replay feedback • Press Space/Enter/N for next question
        </p>
      </div>

      {/* Score Card */}
      <Card className={`border-2 p-8 ${getColor()}`}>
        <div className="flex flex-col items-center gap-4 text-center">
          {getIcon()}
          <div className="space-y-2">
            <h2 className="text-2xl">{getTitle()}</h2>
            <div className="space-y-1">
              <p className="text-3xl">{feedback.score}%</p>
              <p className="text-sm text-muted-foreground">
                {feedback.type === 'correct' && 'Perfect answer'}
                {feedback.type === 'partial' && 'Similarity score'}
                {feedback.type === 'incorrect' && 'Needs improvement'}
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Audio Explanation */}
      <AudioPlayer text={audioText} autoPlay={true} />

      {/* Your Answer */}
      <Card className="p-6">
        <div className="space-y-2">
          <h3 className="text-sm text-muted-foreground">Your Answer:</h3>
          <p className="leading-relaxed">{answer}</p>
        </div>
      </Card>

      {/* Expected Answer (if provided) */}
      {expectedAnswer && (
        <Card className="p-6 border-primary/20 bg-primary/5">
          <div className="space-y-2">
            <h3 className="text-sm text-muted-foreground">Model Answer:</h3>
            <p className="leading-relaxed">{expectedAnswer}</p>
          </div>
        </Card>
      )}

      {/* Explanation */}
      <Card className="p-6">
        <div className="space-y-2">
          <h3 className="text-sm text-muted-foreground">Explanation:</h3>
          <p className="leading-relaxed">{feedback.explanation}</p>
        </div>
      </Card>

      {/* Next Button */}
      <Button onClick={onNext} size="lg" className="w-full min-h-[56px]">
        Next Question
        <ArrowRight className="ml-2 h-5 w-5" />
      </Button>
    </div>
  );
};