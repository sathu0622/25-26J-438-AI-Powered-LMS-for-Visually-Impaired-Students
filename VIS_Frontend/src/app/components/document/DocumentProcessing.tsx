import { useEffect, useState } from 'react';
import { Loader2, FileText, CheckCircle2 } from 'lucide-react';
import { Card } from '../ui/card';
import { Progress } from '../ui/progress';

interface ProcessingStep {
  label: string;
  completed: boolean;
}

interface DocumentProcessingProps {
  fileName: string;
}

export const DocumentProcessing = ({ fileName }: DocumentProcessingProps) => {
  const [steps, setSteps] = useState<ProcessingStep[]>([
    { label: 'Detecting document type', completed: false },
    { label: 'Extracting text', completed: false },
    { label: 'Generating summary', completed: false },
  ]);
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Simulate visible processing steps for accessibility feedback
    const timer1 = setTimeout(() => {
      setSteps((prev) => {
        const newSteps = [...prev];
        newSteps[0].completed = true;
        return newSteps;
      });
      setCurrentStep(1);
      setProgress(33);
    }, 1500);

    const timer2 = setTimeout(() => {
      setSteps((prev) => {
        const newSteps = [...prev];
        newSteps[1].completed = true;
        return newSteps;
      });
      setCurrentStep(2);
      setProgress(66);
    }, 3000);

    const timer3 = setTimeout(() => {
      setSteps((prev) => {
        const newSteps = [...prev];
        newSteps[2].completed = true;
        return newSteps;
      });
      setProgress(100);
    }, 4500);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, []);

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-2 text-center">
        <h1 className="text-2xl">Processing Document</h1>
        <p className="text-muted-foreground">Please wait while we analyze your file</p>
      </div>

      {/* File Info */}
      <Card className="p-6">
        <div className="flex items-center gap-4">
          <div className="rounded-lg bg-primary/10 p-3">
            <FileText className="h-8 w-8 text-primary" aria-hidden="true" />
          </div>
          <div className="flex-1">
            <p className="truncate">{fileName}</p>
            <p className="text-sm text-muted-foreground">Processing...</p>
          </div>
        </div>
      </Card>

      {/* Progress */}
      <div className="space-y-2">
        <Progress value={progress} className="h-2" aria-label={`Processing progress: ${progress}%`} />
        <p className="text-center text-sm text-muted-foreground">{progress}% complete</p>
      </div>

      {/* Steps */}
      <Card className="p-6">
        <div className="space-y-4" role="status" aria-live="polite">
          {steps.map((step, index) => (
            <div key={index} className="flex items-center gap-3">
              {step.completed ? (
                <CheckCircle2
                  className="h-6 w-6 text-success"
                  aria-hidden="true"
                />
              ) : (
                <div className="relative h-6 w-6">
                  {currentStep === index ? (
                    <Loader2
                      className="h-6 w-6 animate-spin text-primary"
                      aria-hidden="true"
                    />
                  ) : (
                    <div className="h-6 w-6 rounded-full border-2 border-muted" />
                  )}
                </div>
              )}
              <p
                className={
                  step.completed
                    ? 'text-success'
                    : currentStep === index
                    ? 'text-foreground'
                    : 'text-muted-foreground'
                }
              >
                {step.label}
              </p>
            </div>
          ))}
        </div>
      </Card>

      {/* Audio Feedback */}
      <p className="text-center text-sm text-muted-foreground" aria-live="assertive">
        Your document is being processed. Please wait.
      </p>
    </div>
  );
};
