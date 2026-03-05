import { useEffect } from 'react';
import { Loader2, BookOpen, Sparkles, Brain, Clock } from 'lucide-react';
import { Card } from '../ui/card';
import { useTTS } from '../../contexts/TTSContext';

interface QuizLoadingProps {
  mode: 'generative' | 'pastpaper' | 'adaptive';
  topic: string;
  onCancel?: () => void;
}

export const QuizLoading = ({ mode, topic, onCancel }: QuizLoadingProps) => {
  const { speak, cancel } = useTTS();

  // Voice announcement when loading starts
  useEffect(() => {
    const announceLoading = () => {
      cancel();
      
      setTimeout(() => {
        let message = '';
        switch (mode) {
          case 'generative':
            message = `Generating personalized quiz questions for ${topic}. Our AI is creating unique questions tailored to this chapter. This may take 10 to 20 seconds. Please wait.`;
            break;
          case 'pastpaper':
            message = `Loading past paper questions for ${topic}. Collecting examination questions from previous years. Please wait.`;
            break;
          case 'adaptive':
            message = `Preparing adaptive quiz for ${topic}. Setting up personalized difficulty adjustment. Please wait.`;
            break;
        }
        speak(message, { interrupt: true });
      }, 500);
    };

    announceLoading();
  }, [mode, topic, speak, cancel]);

  const getLoadingContent = () => {
    switch (mode) {
      case 'generative':
        return {
          icon: <Sparkles className="h-12 w-12 text-primary animate-pulse" />,
          title: 'Generating Quiz Questions',
          subtitle: 'AI is creating personalized questions for you',
          description: 'Our intelligent system is analyzing the chapter content and generating unique questions tailored to your learning needs.',
          estimatedTime: '10-20 seconds'
        };
      case 'pastpaper':
        return {
          icon: <BookOpen className="h-12 w-12 text-blue-600 animate-pulse" />,
          title: 'Loading Past Paper Questions',
          subtitle: 'Collecting examination questions from previous years',
          description: 'We are gathering authentic past paper questions from previous examinations to help you practice with real exam scenarios.',
          estimatedTime: '5-10 seconds'
        };
      case 'adaptive':
        return {
          icon: <Brain className="h-12 w-12 text-purple-600 animate-pulse" />,
          title: 'Preparing Adaptive Quiz',
          subtitle: 'Setting up personalized difficulty adjustment',
          description: 'The system is calibrating question difficulty based on your performance to provide an optimal learning experience.',
          estimatedTime: '5-15 seconds'
        };
      default:
        return {
          icon: <Loader2 className="h-12 w-12 text-primary animate-spin" />,
          title: 'Loading Quiz',
          subtitle: 'Preparing your quiz questions',
          description: 'Please wait while we prepare your quiz.',
          estimatedTime: '10-20 seconds'
        };
    }
  };

  const { icon, title, subtitle, description, estimatedTime } = getLoadingContent();

  // Keyboard navigation for cancel operation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape or B to cancel
      if ((e.key === 'Escape' || e.key.toLowerCase() === 'b') && onCancel) {
        e.preventDefault();
        cancel();
        speak('Quiz generation cancelled. Returning to previous screen.', {
          onEnd: () => onCancel()
        });
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onCancel, speak, cancel]);

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-background via-background/50 to-primary/5">
      <Card className="max-w-md w-full p-8 text-center space-y-6 shadow-lg border-2">
        {/* Loading Icon */}
        <div className="flex justify-center">
          <div className="rounded-full bg-gradient-to-r from-primary/20 to-primary/10 p-6">
            {icon}
          </div>
        </div>

        {/* Title and Subtitle */}
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-foreground">
            {title}
          </h1>
          <p className="text-lg text-muted-foreground">
            {subtitle}
          </p>
        </div>

        {/* Topic */}
        <Card className="p-4 bg-primary/5 border-primary/20">
          <div className="flex items-center gap-3 justify-center">
            <BookOpen className="h-5 w-5 text-primary" />
            <span className="font-medium text-primary">
              {topic}
            </span>
          </div>
        </Card>

        {/* Description */}
        <p className="text-sm text-muted-foreground leading-relaxed">
          {description}
        </p>

        {/* Estimated Time */}
        <div className="flex items-center gap-2 justify-center text-sm text-muted-foreground">
          <Clock className="h-4 w-4" />
          <span>Estimated time: {estimatedTime}</span>
        </div>

        {/* Animated Progress Indicator */}
        <div className="space-y-2">
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div className="bg-gradient-to-r from-primary to-primary/80 h-full rounded-full animate-pulse" 
                 style={{
                   animation: 'loading-progress 2s ease-in-out infinite',
                 }}>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Processing...
          </p>
        </div>

        {/* Cancel Button (if provided) */}
        {onCancel && (
          <div className="pt-4 border-t border-muted-foreground/20">
            <p className="text-xs text-muted-foreground mb-3">
              Press <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Escape</kbd> or{' '}
              <kbd className="px-1 py-0.5 bg-muted rounded text-xs">B</kbd> to cancel
            </p>
            <button
              onClick={() => {
                cancel();
                speak('Quiz generation cancelled. Returning to previous screen.', {
                  onEnd: () => onCancel()
                });
              }}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors underline focus:outline-none focus:ring-2 focus:ring-primary/50 rounded"
              aria-label="Cancel quiz generation and go back"
            >
              Cancel and Go Back
            </button>
          </div>
        )}
      </Card>

      <style jsx>{`
        @keyframes loading-progress {
          0% { width: 0%; }
          50% { width: 70%; }
          100% { width: 100%; }
        }
      `}</style>
    </div>
  );
};