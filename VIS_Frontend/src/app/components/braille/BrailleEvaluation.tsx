import { useState, useEffect } from 'react';
import { Loader2, CheckCircle2, XCircle, AlertCircle, Volume2 } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { AudioPlayer } from '../AudioPlayer';

interface BrailleEvaluationProps {
  onBack: () => void;
}

type EvaluationStatus = 'converting' | 'converted' | 'evaluating' | 'complete';
type ResultType = 'correct' | 'partial' | 'incorrect';

export const BrailleEvaluation = ({ onBack }: BrailleEvaluationProps) => {
  const [status, setStatus] = useState<EvaluationStatus>('converting');
  const [progress, setProgress] = useState(0);
  const [convertedText, setConvertedText] = useState('');
  const [result, setResult] = useState<ResultType | null>(null);
  const [score, setScore] = useState(0);
  const [feedback, setFeedback] = useState<string[]>([]);
  const [showDetailedReport, setShowDetailedReport] = useState(false);

  const question = "Explain the importance of accessible education for students with visual impairments.";
  const modelAnswer = "Accessible education for students with visual impairments is crucial for ensuring equal learning opportunities and fostering independence. It involves implementing adaptive technologies such as screen readers, Braille displays, and audio learning materials that enable students to access educational content effectively. Multi-sensory learning approaches help students engage with material through touch, sound, and other senses. Inclusive teaching methods ensure that educational materials are designed with accessibility in mind from the start, rather than as an afterthought. This comprehensive approach empowers visually impaired students to participate fully in academic activities, develop critical thinking skills, and achieve their educational goals without barriers.";

  // Voice announcements for page state changes
  useEffect(() => {
    // STOP all previous speech immediately when component mounts
    window.speechSynthesis.cancel();
    
    if (status === 'converted' && convertedText) {
      const announcement = new SpeechSynthesisUtterance('Answer converted from Braille. Press E to evaluate your answer, or Press A to hear your answer read aloud.');
      setTimeout(() => window.speechSynthesis.speak(announcement), 500);
    }
    
    if (status === 'complete' && !showDetailedReport) {
      setTimeout(() => {
        const announcement = new SpeechSynthesisUtterance(`Evaluation complete. Your score is ${score} percent. Press F to replay feedback, Press D for detailed report, Press B to upload another answer, or Press Escape to go back.`);
        window.speechSynthesis.speak(announcement);
      }, 10000); // After feedback finishes
    }
    
    if (showDetailedReport) {
      window.speechSynthesis.cancel(); // Stop any ongoing speech
      const announcement = new SpeechSynthesisUtterance('Detailed Report page. Press A to hear your answer, Press M to hear the model answer, Press B to go back to summary, or Press Escape.');
      setTimeout(() => window.speechSynthesis.speak(announcement), 500);
    }
    
    // Cleanup: stop speech when leaving page
    return () => {
      window.speechSynthesis.cancel();
    };
  }, [status, showDetailedReport, convertedText, score]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // A key to replay answer
      if ((e.key === 'a' || e.key === 'A') && convertedText) {
        e.preventDefault();
        const utterance = new SpeechSynthesisUtterance(convertedText);
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      }

      // M key to play model answer
      if ((e.key === 'm' || e.key === 'M') && showDetailedReport) {
        e.preventDefault();
        const utterance = new SpeechSynthesisUtterance(modelAnswer);
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      }

      // F key to replay feedback
      if ((e.key === 'f' || e.key === 'F') && status === 'complete' && feedback.length > 0) {
        e.preventDefault();
        const feedbackText = `Your score is ${score}%. ${feedback.join('. ')}`;
        const utterance = new SpeechSynthesisUtterance(feedbackText);
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      }

      // E key to evaluate (when converted)
      if ((e.key === 'e' || e.key === 'E') && status === 'converted') {
        e.preventDefault();
        handleEvaluate();
      }

      // D key to view detailed report
      if ((e.key === 'd' || e.key === 'D') && status === 'complete' && !showDetailedReport) {
        e.preventDefault();
        setShowDetailedReport(true);
      }

      // B key to go back
      if ((e.key === 'b' || e.key === 'B')) {
        e.preventDefault();
        if (showDetailedReport) {
          setShowDetailedReport(false);
        } else if (status === 'complete') {
          onBack();
        }
      }

      // Escape to go back
      if (e.key === 'Escape') {
        e.preventDefault();
        if (showDetailedReport) {
          setShowDetailedReport(false);
        } else {
          onBack();
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [status, convertedText, feedback, score, showDetailedReport, onBack]);

  useEffect(() => {
    // Simulate Braille conversion
    const timer1 = setTimeout(() => {
      setProgress(50);
      const mockConverted = "Accessible education ensures equal learning opportunities through adaptive technologies and inclusive teaching methods.";
      setConvertedText(mockConverted);
      setStatus('converted');
    }, 2000);

    return () => clearTimeout(timer1);
  }, []);

  const handleEvaluate = () => {
    setStatus('evaluating');
    setProgress(75);

    // Simulate AI evaluation
    setTimeout(() => {
      setProgress(100);
      setResult('partial');
      setScore(75);
      setFeedback([
        'Good explanation of the core concept',
        'Missing mention of multi-sensory learning approaches',
        'Could elaborate on specific adaptive technologies',
      ]);
      setStatus('complete');
    }, 2500);
  };

  const getResultIcon = () => {
    switch (result) {
      case 'correct':
        return <CheckCircle2 className="h-16 w-16 text-success" />;
      case 'partial':
        return <AlertCircle className="h-16 w-16 text-warning" />;
      case 'incorrect':
        return <XCircle className="h-16 w-16 text-destructive" />;
      default:
        return null;
    }
  };

  const getResultColor = () => {
    switch (result) {
      case 'correct':
        return 'border-success bg-success/10';
      case 'partial':
        return 'border-warning bg-warning/10';
      case 'incorrect':
        return 'border-destructive bg-destructive/10';
      default:
        return '';
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {!showDetailedReport ? (
        <>
          {/* Header */}
          <div className="space-y-2 text-center">
            <h1 className="text-2xl">Answer Evaluation</h1>
            <p className="text-muted-foreground">
              {status === 'converting' && 'Converting Braille to text...'}
              {status === 'converted' && 'Press E to evaluate • Press A to hear answer'}
              {status === 'evaluating' && 'Evaluating your answer...'}
              {status === 'complete' && 'Press F to replay feedback • Press D for details • Press B to upload another'}
            </p>
          </div>

          {/* Progress */}
          {status !== 'complete' && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <p className="text-center text-sm text-muted-foreground">
                {progress}% complete
              </p>
            </div>
          )}

          {/* Question */}
          <Card className="p-6">
            <div className="space-y-2">
              <h2 className="text-sm text-muted-foreground">Question:</h2>
              <p className="text-lg">{question}</p>
            </div>
          </Card>

          {/* Converted Text */}
          {status !== 'converting' && (
            <Card className="p-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm text-muted-foreground">
                    Your Answer (Converted from Braille):
                  </h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      const utterance = new SpeechSynthesisUtterance(convertedText);
                      window.speechSynthesis.speak(utterance);
                    }}
                    aria-label="Read answer aloud"
                  >
                    <Volume2 className="h-4 w-4" />
                  </Button>
                </div>
                <p className="leading-relaxed">{convertedText}</p>
              </div>
            </Card>
          )}

          {/* Evaluate Button */}
          {status === 'converted' && (
            <Button
              onClick={handleEvaluate}
              size="lg"
              className="w-full min-h-[56px]"
            >
              Evaluate Answer
            </Button>
          )}

          {/* Loading State */}
          {status === 'evaluating' && (
            <Card className="p-8">
              <div className="flex flex-col items-center gap-4 text-center">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <p>AI is evaluating your answer...</p>
              </div>
            </Card>
          )}

          {/* Results */}
          {status === 'complete' && result && (
            <>
              {/* Score Card */}
              <Card className={`border-2 p-8 ${getResultColor()}`}>
                <div className="flex flex-col items-center gap-4 text-center">
                  {getResultIcon()}
                  <div className="space-y-2">
                    <h2 className="text-2xl">Score: {score}%</h2>
                    <p className="text-lg">
                      {result === 'correct' && 'Excellent Work!'}
                      {result === 'partial' && 'Good Effort!'}
                      {result === 'incorrect' && 'Needs Improvement'}
                    </p>
                  </div>
                </div>
              </Card>

              {/* Feedback */}
              {feedback.length > 0 && (
                <div className="space-y-4">
                  <h2 className="text-lg">Feedback</h2>
                  <AudioPlayer
                    text={`Your score is ${score}%. ${feedback.join('. ')}`}
                    autoPlay={true}
                  />
                  <Card className="p-6">
                    <ul className="space-y-3">
                      {feedback.map((item, index) => (
                        <li key={index} className="flex gap-3">
                          <span className="text-secondary">•</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </Card>
                </div>
              )}

              {/* Actions */}
              <div className="grid gap-3 sm:grid-cols-2">
                <Button
                  onClick={onBack}
                  variant="outline"
                  size="lg"
                  className="min-h-[56px]"
                >
                  Upload Another
                </Button>
                <Button
                  size="lg"
                  className="min-h-[56px]"
                  onClick={() => setShowDetailedReport(true)}
                >
                  View Detailed Report
                </Button>
              </div>
            </>
          )}
        </>
      ) : (
        <div className="space-y-6">
          <h1 className="text-2xl">Detailed Report</h1>
          <p className="text-muted-foreground">
            Press A to hear your answer • Press M for model answer • Press B to go back
          </p>
          
          {/* Question */}
          <Card className="p-6">
            <div className="space-y-4">
              <h2 className="text-sm text-muted-foreground">Question:</h2>
              <p className="text-lg">{question}</p>
            </div>
          </Card>
          
          {/* Your Answer */}
          <Card className="p-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm text-muted-foreground">Your Answer:</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const utterance = new SpeechSynthesisUtterance(convertedText);
                    window.speechSynthesis.speak(utterance);
                  }}
                  aria-label="Read your answer aloud"
                >
                  <Volume2 className="h-4 w-4" />
                </Button>
              </div>
              <p className="leading-relaxed">{convertedText}</p>
            </div>
          </Card>
          
          {/* Score */}
          <Card className="p-6">
            <div className="space-y-4">
              <h2 className="text-sm text-muted-foreground">Score:</h2>
              <p className="text-lg">{score}%</p>
            </div>
          </Card>
          
          {/* Model Answer (100% Correct Answer) */}
          <Card className="border-2 border-success bg-success/5 p-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <h2 className="text-sm text-muted-foreground">Model Answer (100%):</h2>
                  <p className="text-xs text-muted-foreground">This is what a perfect answer looks like</p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const utterance = new SpeechSynthesisUtterance(modelAnswer);
                    window.speechSynthesis.speak(utterance);
                  }}
                  aria-label="Read model answer aloud"
                >
                  <Volume2 className="h-4 w-4" />
                </Button>
              </div>
              <p className="leading-relaxed">{modelAnswer}</p>
            </div>
          </Card>
          
          {/* Feedback */}
          <Card className="p-6">
            <div className="space-y-4">
              <h2 className="text-sm text-muted-foreground">Feedback:</h2>
              <ul className="space-y-3">
                {feedback.map((item, index) => (
                  <li key={index} className="flex gap-3">
                    <span className="text-secondary">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </Card>
          
          {/* Back Button */}
          <Button
            onClick={() => setShowDetailedReport(false)}
            size="lg"
            className="min-h-[56px]"
          >
            Back to Summary
          </Button>
        </div>
      )}
    </div>
  );
};