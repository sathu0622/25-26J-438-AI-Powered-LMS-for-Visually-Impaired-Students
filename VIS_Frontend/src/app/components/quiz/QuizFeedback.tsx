import { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, AlertCircle, ArrowRight, Loader2, Home, Play } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { AudioPlayer } from '../AudioPlayer';
import { useTTS } from '../../contexts/TTSContext';

interface QuizFeedbackProps {
  question: string;
  answer: string;
  result: {
    score: number;
    feedback: string;
    correct: boolean;
    correct_answer?: string;
  };
  onNext: () => void;
  onGoHome: () => void;
  onBack?: () => void;
  isLastQuestion?: boolean;
}


export const QuizFeedback = ({
  question,
  answer,
  result,
  onNext,
  onGoHome,
  onBack,
  isLastQuestion = false,
}: QuizFeedbackProps & { onGoHome: () => void }) => {
  const { speak, cancel } = useTTS();
  const isCorrect = result.score>=60;

  // Keyboard shortcuts: N for next, Backspace/B for back
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
          onGoHome();
        }
      }
      
      // H for home
      if (key === 'h') {
        e.preventDefault();
        onGoHome();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNext, onGoHome, onBack]);

  // 🔊 Automatically speak feedback, given answer, and model answer in sequence on load
  useEffect(() => {
    cancel();
    // Chain speech segments using onEnd callback
      speak(`You scored ${result.score} percent. ${result.feedback}`, {
        onEnd: () => {
          speak(`Your answer: ${answer}`, {
            onEnd: () => {
              speak(`Model answer: ${result.correct_answer || "No model answer available."}`);
            }
          });
        }
      });
        
      
    
    return () => cancel();
  }, [result, answer, speak, cancel]);

  const getIcon = () => {
    if (result.correct && result.score > 75)
      return <CheckCircle2 className="h-12 w-12 text-green-500" />;

    if (result.correct)
      return <AlertCircle className="h-12 w-12 text-yellow-500" />;

    return <XCircle className="h-12 w-12 text-red-500" />;
  };

  

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-semibold">Feedback</h1>
        </div>
        {/* Main Split Layout */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* ================= LEFT PANE ================= */}
          <Card className="p-8 flex flex-col justify-between space-y-6">
            {/* Score Banner */}
            <div
              className={`rounded-xl p-6 text-center ${
                isCorrect
                  ? "bg-green-50 border border-green-200"
                  : "bg-red-50 border border-red-200"
              }`}
            >
             
              <h2 className="text-4xl font-bold mb-2">{result.score}%</h2>
              <div className="flex items-center justify-center gap-2 mt-2">
                {isCorrect ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : (
                  <XCircle className="h-5 w-5 text-red-500" />
                )}
                <span className={`font-medium ${isCorrect ? "text-green-700" : "text-red-700"}`}>
                  {isCorrect ? "Your Answer is Correct" : "Your Answer is Incorrect"}
                </span>
                {/* Voice button for correctness */}
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Speak correctness"
                  onClick={() => speak(isCorrect ? "Your answer is correct." : "Your answer is incorrect.")}
                  className="ml-2"
                >
                  <Play className="h-4 w-4 text-blue-600" />
                </Button>
              </div>
            </div>
            {/* Navigation Buttons */}
            <div className="flex flex-col gap-4 mt-6">
              <Button size="lg" className="font-semibold" onClick={onNext}>
                {isLastQuestion ? 'See Summary' : 'Next Question'}
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button variant="outline" size="lg" className="font-semibold" onClick={onGoHome}>
                <Home className="mr-2 h-5 w-5" /> Select Another Chapter
              </Button>
            </div>
          </Card>
          {/* ================= RIGHT PANE ================= */}
          <div className="space-y-4">
            
            {/* Your Answer */}
            <Card className="p-6 bg-white border border-gray-200 flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-muted-foreground mb-2">YOUR ANSWER</p>
                <p className="text-base">{answer}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Speak your answer"
                onClick={() => speak(`Your answer: ${answer}`)}
              >
                <Play className="h-4 w-4 text-blue-600" />
              </Button>
            </Card>
            {/* Model Answer (always show) */}
            <Card className="p-6 bg-blue-50 border border-blue-200 flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-blue-700 mb-2">MODEL ANSWER</p>
                <p className="text-base">{result.correct_answer || "No model answer available."}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Speak model answer"
                onClick={() => speak(`Model answer: ${result.correct_answer || "No model answer available."}`)}
              >
                <Play className="h-4 w-4 text-blue-600" />
              </Button>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};