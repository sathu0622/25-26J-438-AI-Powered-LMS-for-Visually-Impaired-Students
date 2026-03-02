import { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Volume2 } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { useTTS } from '../contexts/TTSContext';

interface AudioPlayerProps {
  text: string;
  autoPlay?: boolean;
  className?: string;
}

export const AudioPlayer = ({ text, autoPlay = false, className }: AudioPlayerProps) => {
  const { speak, cancel, isSpeaking } = useTTS();
  const [hasPlayed, setHasPlayed] = useState(false);

  useEffect(() => {
    cancel();
    if (autoPlay && text && !hasPlayed) {
      const t = setTimeout(() => {
        speak(text, { interrupt: false });
        setHasPlayed(true);
      }, 100);
      return () => clearTimeout(t);
    }
    return () => cancel();
  }, [autoPlay, text, hasPlayed, speak, cancel]);

  const handlePlayPause = () => {
    if (isSpeaking) {
      cancel();
    } else {
      speak(text, { interrupt: false });
    }
  };

  const handleReplay = () => {
    cancel();
    setTimeout(() => speak(text, { interrupt: false }), 100);
  };

  return (
    <Card className={`p-6 ${className}`}>
      <div className="flex items-center gap-4">
        <Volume2 className="h-6 w-6 text-secondary" aria-hidden="true" />
        <div className="flex-1">
          <p className="text-sm text-muted-foreground">Audio Playback</p>
          <p className="text-sm">
            {isSpeaking ? 'Playing...' : 'Ready to play'}
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handlePlayPause}
            size="lg"
            aria-label={isSpeaking ? 'Pause audio' : 'Play audio'}
            className="min-h-[56px] min-w-[56px]"
          >
            {isSpeaking ? (
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