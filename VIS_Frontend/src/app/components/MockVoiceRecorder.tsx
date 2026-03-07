
import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Check, X } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { useTTS } from '../contexts/TTSContext';

interface VoiceRecorderProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
  title?: string;
  context?: string;
}

export const VoiceRecorder = ({
  isOpen,
  onClose,
  onSubmit,
  title = 'Record Your Answer',
  context,
}: VoiceRecorderProps) => {
  const { speak } = useTTS();
  const [isRecording, setIsRecording] = useState(false);
  const [hasRecorded, setHasRecorded] = useState(false);
  const [transcript, setTranscript] = useState('');
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (isOpen) {
      setIsRecording(false);
      setHasRecorded(false);
      setTranscript('');
      const t = setTimeout(
        () => speak('Voice recorder ready. Press Space or R to start recording.', { interrupt: true }),
        300
      );
      return () => clearTimeout(t);
    }
  }, [isOpen, speak]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === ' ' || e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        if (!isRecording && !hasRecorded) {
          handleStartRecording();
        } else if (isRecording) {
          handleStopRecording();
        }
      }
      if (e.key === 'Enter' && hasRecorded) {
        e.preventDefault();
        handleSubmit();
      }
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
    setHasRecorded(false);
    setTranscript('');
    // Delay starting recognition until TTS is finished
    speak('Recording started. Speak your answer now.', {
      interrupt: true,
      onEnd: () => {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
          const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
          const recognition = new SpeechRecognition();
          recognition.lang = 'en-LK';
          recognition.interimResults = false;
          recognition.maxAlternatives = 1;
          recognition.onresult = (event: any) => {
            const text = event.results[0][0].transcript;
            setTranscript(text);
            setHasRecorded(true);
            setIsRecording(false);
            speak('Recording complete. Press Enter to submit or Space to re-record.', { interrupt: true });
          };
          recognition.onerror = (event: any) => {
            setIsRecording(false);
            setHasRecorded(false);
            setTranscript('');
            speak('Sorry, could not capture your voice. Please try again.', { interrupt: true });
          };
          recognitionRef.current = recognition;
          recognition.start();
        } else {
          setIsRecording(false);
          setHasRecorded(false);
          setTranscript('');
          speak('Speech recognition is not supported in this browser.', { interrupt: true });
        }
      }
    });
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  const handleSubmit = () => {
    if (transcript.trim()) {
      onSubmit(transcript.trim());
      onClose();
    }
  };

  const handleReRecord = () => {
    setIsRecording(false);
    setHasRecorded(false);
    setTranscript('');
    speak('Recording cleared. Press Space to record again.', { interrupt: true });
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
                  ? 'Speak now...'
                  : hasRecorded
                  ? 'Press Enter to submit'
                  : 'Press Space or R to start'}
              </p>
            </div>
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
