import { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Volume2 } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { useMockSpeechSynthesis } from '../hooks/useMockSpeechSynthesis';
import { safeCancel } from '../utils/mockSpeech';

interface AudioPlayerProps {
  text: string;
  autoPlay?: boolean;
  className?: string;
}

export const AudioPlayer = ({ text, autoPlay = false, className }: AudioPlayerProps) => {
  const { speak, stop, isSpeaking, isPaused, resume, pause } = useMockSpeechSynthesis();
  const [hasPlayed, setHasPlayed] = useState(false);

  useEffect(() => {
    // Stop any other speech first
    safeCancel();
    
    if (autoPlay && text && !hasPlayed) {
      setTimeout(() => {
        speak(text);
        setHasPlayed(true);
      }, 100);
    }
    
    return () => {
      stop();
    };
  }, [autoPlay, text, hasPlayed, speak, stop]);

  const handlePlayPause = () => {
    if (isSpeaking && !isPaused) {
      pause();
    } else if (isPaused) {
      resume();
    } else {
      speak(text);
    }
  };

  const handleReplay = () => {
    stop();
    setTimeout(() => speak(text), 100);
  };

  const handleStop = () => {
    stop();
  };

  return (
    <Card className={`p-6 ${className}`}>
      <div className="flex items-center gap-4">
        <Volume2 className="h-6 w-6 text-secondary" aria-hidden="true" />
        <div className="flex-1">
          <p className="text-sm text-muted-foreground">Audio Playback</p>
          <p className="text-sm">
            {isSpeaking ? (isPaused ? 'Paused' : 'Playing...') : 'Ready to play'}
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handlePlayPause}
            size="lg"
            aria-label={isSpeaking && !isPaused ? 'Pause audio' : 'Play audio'}
            className="min-h-[56px] min-w-[56px]"
          >
            {isSpeaking && !isPaused ? (
              <Pause className="h-6 w-6" aria-hidden="true" />
            ) : (
              <Play className="h-6 w-6" aria-hidden="true" />
            )}
          </Button>
          <Button
            onClick={handleReplay}
            variant="outline"
            size="lg"
            aria-label="Replay audio"
            className="min-h-[56px] min-w-[56px]"
          >
            <RotateCcw className="h-6 w-6" aria-hidden="true" />
          </Button>
        </div>
      </div>
    </Card>
  );
};