import { useState, useEffect } from 'react';
import { Mic, MicOff, Check, X } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { safeSpeak } from '../utils/mockSpeech';

interface MockVoiceRecorderProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
  title?: string;
  context?: string; // Context for generating mock answers (e.g., question text)
}

// Mock answer generator for quiz questions
const generateMockAnswer = (context?: string): string => {
  const mockAnswers = [
    "The main event occurred during the colonial period when the Portuguese arrived in Sri Lanka in 1505. They established control over coastal areas and introduced significant changes to the administrative system.",
    "The significance lies in the fact that this period marked a major transformation in Sri Lankan society. The cultural exchanges that took place during this time influenced art, architecture, and social structures.",
    "During this era, there were important developments in trade routes and economic systems. The spice trade became particularly significant, connecting Sri Lanka to global commerce networks.",
    "The key factors include geographical advantages, strategic location, and rich natural resources. These elements combined to make the region politically and economically important.",
    "This historical development led to changes in governance structures and administrative practices. Local kingdoms had to adapt to new political realities while maintaining cultural traditions.",
  ];
  
  return mockAnswers[Math.floor(Math.random() * mockAnswers.length)];
};

export const MockVoiceRecorder = ({
  isOpen,
  onClose,
  onSubmit,
  title = 'Record Your Answer',
  context,
}: MockVoiceRecorderProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [hasRecorded, setHasRecorded] = useState(false);
  const [transcript, setTranscript] = useState('');

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setIsRecording(false);
      setRecordingTime(0);
      setHasRecorded(false);
      setTranscript('');
      
      // Announce modal opened
      setTimeout(() => {
        safeSpeak('Voice recorder ready. Press Space or R to start recording.');
      }, 300);
    }
  }, [isOpen]);

  // Recording timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  // Auto-stop recording after 10 seconds and generate mock transcript
  useEffect(() => {
    if (isRecording && recordingTime >= 10) {
      handleStopRecording();
    }
  }, [isRecording, recordingTime]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyPress = (e: KeyboardEvent) => {
      // Space or R to start/stop recording
      if (e.key === ' ' || e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        if (!isRecording && !hasRecorded) {
          handleStartRecording();
        } else if (isRecording) {
          handleStopRecording();
        }
      }

      // Enter to submit (if has recorded)
      if (e.key === 'Enter' && hasRecorded) {
        e.preventDefault();
        handleSubmit();
      }

      // Escape to cancel
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isOpen, isRecording, hasRecorded, transcript]);

  const handleStartRecording = () => {
    setIsRecording(true);
    setRecordingTime(0);
    safeSpeak('Recording started. Speak your answer now.');
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    setHasRecorded(true);
    
    // Generate mock transcript
    const mockTranscript = generateMockAnswer(context);
    setTranscript(mockTranscript);
    
    // Announce completion
    setTimeout(() => {
      safeSpeak(`Recording complete. Mock answer generated. Press Enter to submit or Space to re-record.`);
    }, 500);
  };

  const handleSubmit = () => {
    if (transcript.trim()) {
      onSubmit(transcript.trim());
      onClose();
    }
  };

  const handleReRecord = () => {
    setIsRecording(false);
    setRecordingTime(0);
    setHasRecorded(false);
    setTranscript('');
    safeSpeak('Recording cleared. Press Space to record again.');
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="voice-recorder-title"
    >
      <Card className="relative w-full max-w-lg p-6">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <h2 id="voice-recorder-title" className="text-xl">
            {title}
          </h2>
          <Button
            onClick={onClose}
            variant="ghost"
            size="icon"
            aria-label="Close recorder"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Recording Interface */}
        <div className="space-y-6">
          {/* Visual Indicator */}
          <div className="flex flex-col items-center gap-4">
            <div
              className={`flex h-32 w-32 items-center justify-center rounded-full transition-all ${
                isRecording
                  ? 'animate-pulse bg-red-500/20 ring-4 ring-red-500'
                  : hasRecorded
                  ? 'bg-green-500/20 ring-4 ring-green-500'
                  : 'bg-primary/20 ring-4 ring-primary'
              }`}
            >
              {isRecording ? (
                <Mic className="h-16 w-16 text-red-500" />
              ) : hasRecorded ? (
                <Check className="h-16 w-16 text-green-500" />
              ) : (
                <MicOff className="h-16 w-16 text-primary" />
              )}
            </div>

            {/* Status Text */}
            <div className="text-center">
              <p className="text-lg">
                {isRecording
                  ? 'Recording...'
                  : hasRecorded
                  ? 'Recording Complete!'
                  : 'Ready to Record'}
              </p>
              <p className="text-sm text-muted-foreground">
                {isRecording
                  ? formatTime(recordingTime)
                  : hasRecorded
                  ? 'Press Enter to submit'
                  : 'Press Space or R to start'}
              </p>
            </div>
          </div>

          {/* Demo Mode Notice */}
          <div className="rounded-lg bg-primary/10 p-4 text-center">
            <p className="text-sm text-primary">
              🎙️ Demo Mode: Mock Voice Recording
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {isRecording
                ? 'Simulating voice capture... Answer will be auto-generated'
                : hasRecorded
                ? 'Mock answer generated based on quiz context'
                : 'Click the button or press Space to simulate recording'}
            </p>
          </div>

          {/* Transcript Preview */}
          {hasRecorded && transcript && (
            <div className="space-y-2">
              <label className="text-sm text-muted-foreground">
                Recorded Answer:
              </label>
              <div className="rounded-lg border bg-muted/50 p-4">
                <p className="text-sm leading-relaxed">{transcript}</p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="grid gap-3">
            {!hasRecorded ? (
              <>
                <Button
                  onClick={isRecording ? handleStopRecording : handleStartRecording}
                  size="lg"
                  className="min-h-[56px]"
                  variant={isRecording ? 'destructive' : 'default'}
                >
                  {isRecording ? (
                    <>
                      <MicOff className="mr-2 h-5 w-5" />
                      Stop Recording
                    </>
                  ) : (
                    <>
                      <Mic className="mr-2 h-5 w-5" />
                      Start Recording (Space)
                    </>
                  )}
                </Button>
                <Button onClick={onClose} variant="outline" size="lg">
                  Cancel (Esc)
                </Button>
              </>
            ) : (
              <>
                <Button onClick={handleSubmit} size="lg" className="min-h-[56px]">
                  <Check className="mr-2 h-5 w-5" />
                  Submit Answer (Enter)
                </Button>
                <div className="grid grid-cols-2 gap-3">
                  <Button onClick={handleReRecord} variant="outline" size="lg">
                    Re-record (Space)
                  </Button>
                  <Button onClick={onClose} variant="outline" size="lg">
                    Cancel (Esc)
                  </Button>
                </div>
              </>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
};
