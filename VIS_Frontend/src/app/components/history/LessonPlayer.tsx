import { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  ArrowLeft,
  Volume2,
  Loader,
  SkipBack,
  SkipForward,
  Shuffle,
  Repeat,
} from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { Slider } from '../ui/slider';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';
import { API_BASE_URL } from '../../services/api';

interface LessonPlayerProps {
  topicName: string;
  content: string;
  grade: number;
  chapterIdx: number;
  topicIdx: number;
  autoPlay?: boolean;
  onBack: () => void;
}

export const LessonPlayer = ({
  topicName,
  content,
  grade,
  chapterIdx,
  topicIdx,
  autoPlay = true,
  onBack
}: LessonPlayerProps) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasAnnounced, setHasAnnounced] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isRepeat, setIsRepeat] = useState(false);
  const [isShuffle, setIsShuffle] = useState(false);

  // Generate audio when component mounts
  useEffect(() => {
    safeCancel();
    setHasAnnounced(false);
    generateAudio();
  }, [grade, chapterIdx, topicIdx]);

  // Auto-play if enabled and audio is ready
  useEffect(() => {
    if (autoPlay && audioUrl && audioRef.current && !isLoading && !hasAnnounced) {
      const timer = setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.play().then(() => {
            setIsPlaying(true);
            safeSpeak(`Now playing: ${topicName}`);
            setHasAnnounced(true);
          }).catch(err => {
            console.error('Auto-play failed:', err);
            safeSpeak('Audio ready. Click play to start.');
            setHasAnnounced(true);
          });
        }
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [audioUrl, autoPlay, isLoading, topicName, hasAnnounced]);

  const generateAudio = async () => {
    try {
      setIsLoading(true);
      setError(null);
      safeSpeak('Generating audio using AI text-to-speech model...');

      // Call backend to generate audio using the TTS model
      const response = await fetch(
        `${API_BASE_URL}/api/audio/chapter/${grade}/${chapterIdx}/${topicIdx}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to generate audio: ${response.statusText}`);
      }

      // Create a blob URL from the audio response
      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      safeSpeak(`Audio generated successfully. Press space to play.`);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      setError(`Failed to generate audio: ${errorMsg}`);
      safeSpeak(`Error generating audio: ${errorMsg}`);
      console.error('Audio generation error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        safeSpeak('Audio paused');
      } else {
        audioRef.current.play();
        safeSpeak('Audio playing');
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleReplay = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
      setIsPlaying(true);
      safeSpeak('Replaying audio');
    }
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      safeSpeak('Audio stopped');
    }
  };

  const handleAudioEnded = () => {
    if (isRepeat) {
      if (audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play();
      }
    } else {
      setIsPlaying(false);
      safeSpeak('Audio playback completed');
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleSeek = (value: number[]) => {
    if (audioRef.current) {
      const newTime = (value[0] / 100) * duration;
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleSkipForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(audioRef.current.currentTime + 10, duration);
      safeSpeak('Skipped forward 10 seconds');
    }
  };

  const handleSkipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(audioRef.current.currentTime - 10, 0);
      safeSpeak('Skipped backward 10 seconds');
    }
  };

  const toggleRepeat = () => {
    setIsRepeat(!isRepeat);
    safeSpeak(isRepeat ? 'Repeat off' : 'Repeat on');
  };

  const toggleShuffle = () => {
    setIsShuffle(!isShuffle);
    safeSpeak(isShuffle ? 'Shuffle off' : 'Shuffle on');
  };

  const formatTime = (time: number) => {
    if (isNaN(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Space bar to play/pause
      if (e.key === ' ') {
        e.preventDefault();
        handlePlayPause();
      }

      // R key to replay
      if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleReplay();
      }

      // Arrow Left - Skip backward
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        handleSkipBackward();
      }

      // Arrow Right - Skip forward
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        handleSkipForward();
      }

      // T key to toggle repeat
      if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        toggleRepeat();
      }

      // S key to toggle shuffle
      if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        toggleShuffle();
      }

      // Escape to go back
      if (e.key === 'Escape') {
        e.preventDefault();
        safeCancel();
        handleStop();
        onBack();
      }

      // H key for help
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        safeSpeak(
          `Now playing: ${topicName}. Press Space to play or pause. Press R to replay. Press Left or Right arrows to skip. Press T to toggle repeat. Press S to toggle shuffle. Press Escape to go back.`
        );
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isPlaying, topicName, onBack, isRepeat, isShuffle]);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button 
          onClick={onBack} 
          variant="ghost" 
          size="icon" 
          aria-label="Go back"
          title="Press ESC to go back"
        >
          <ArrowLeft className="h-6 w-6" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">{topicName}</h1>
          <p className="text-sm text-muted-foreground">
            Grade {grade} • Chapter {chapterIdx + 1} • Topic {topicIdx + 1}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            SPC: Play/Pause • ←/→: Skip • R: Replay • T: Repeat • S: Shuffle • H: Help • ESC: Back
          </p>
        </div>
      </div>

      {/* Content Card */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Volume2 className="h-6 w-6 text-orange-500" aria-hidden="true" />
            <h2 className="text-lg font-semibold">Lesson Content</h2>
          </div>
          <p className="leading-relaxed text-foreground whitespace-pre-wrap">
            {content || 'No content available'}
          </p>
        </div>
      </Card>

      {/* Error Message */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Audio Player Controls */}
      <Card className="p-6 bg-gradient-to-br from-gray-900 to-gray-800 text-white">
        <div className="space-y-6">
          {/* Status */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold">
                {isLoading ? '⏳ Generating Audio...' :
                 error ? '❌ Audio Generation Failed' :
                 !audioUrl ? '⏸️ Audio Not Ready' :
                      isPlaying ? '▶️ Now Playing' : '⏸️ Paused'}
              </p>
              <p className="text-xs text-gray-400">
                {isLoading ? 'Please wait...' : 'AI Text-to-Speech'}
              </p>
            </div>
          </div>

          {/* Progress Slider */}
          <div className="space-y-2">
            <Slider
              value={[duration > 0 ? (currentTime / duration) * 100 : 0]}
              onValueChange={handleSeek}
              max={100}
              step={0.1}
              className="w-full cursor-pointer"
              disabled={!audioUrl || isLoading}
              aria-label="Audio progress"
            />
            <div className="flex justify-between text-xs text-gray-400">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center justify-center gap-3">
            <Button
              onClick={toggleShuffle}
              disabled={isLoading || !audioUrl}
              variant="ghost"
              size="icon"
              className={`h-10 w-10 ${isShuffle ? 'text-green-500' : 'text-gray-400'} hover:text-white`}
              aria-label="Toggle shuffle"
              title="Press S to toggle shuffle"
            >
              <Shuffle className="h-5 w-5" />
            </Button>

            <Button
              onClick={handleSkipBackward}
              disabled={isLoading || !audioUrl}
              variant="ghost"
              size="icon"
              className="h-12 w-12 text-white hover:text-white hover:bg-white/20"
              aria-label="Skip backward 10 seconds"
              title="Press ← to skip backward"
            >
              <SkipBack className="h-6 w-6" />
            </Button>

            <Button
              onClick={handlePlayPause}
              disabled={isLoading || !audioUrl}
              size="icon"
              className="h-16 w-16 rounded-full bg-white text-gray-900 hover:bg-gray-200 shadow-lg"
              aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
              title="Press SPACE to play/pause"
            >
              {isLoading ? (
                <Loader className="h-8 w-8 animate-spin" aria-hidden="true" />
              ) : isPlaying ? (
                <Pause className="h-8 w-8" aria-hidden="true" />
              ) : (
                <Play className="h-8 w-8" aria-hidden="true" />
              )}
            </Button>

            <Button
              onClick={handleSkipForward}
              disabled={isLoading || !audioUrl}
              variant="ghost"
              size="icon"
              className="h-12 w-12 text-white hover:text-white hover:bg-white/20"
              aria-label="Skip forward 10 seconds"
              title="Press → to skip forward"
            >
              <SkipForward className="h-6 w-6" />
            </Button>

            <Button
              onClick={toggleRepeat}
              disabled={isLoading || !audioUrl}
              variant="ghost"
              size="icon"
              className={`h-10 w-10 ${isRepeat ? 'text-green-500' : 'text-gray-400'} hover:text-white`}
              aria-label="Toggle repeat"
              title="Press T to toggle repeat"
            >
              <Repeat className="h-5 w-5" />
            </Button>
          </div>

          {/* Hidden audio element */}
          <audio
            ref={audioRef}
            src={audioUrl || undefined}
            onEnded={handleAudioEnded}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            className="hidden"
            aria-label="Lesson audio player"
          />

          {/* Demo Mode Label */}
          <div className="text-center text-xs text-gray-400 pt-2">
            🎙️ AI-Powered Voice Lesson
          </div>
        </div>
      </Card>

      {/* Regenerate Button */}
      {audioUrl && !isLoading && (
        <Button
          onClick={generateAudio}
          variant="secondary"
          className="w-full"
        >
          Regenerate Audio
        </Button>
      )}

      {/* Keyboard Shortcuts Help */}
      <Card className="p-4 bg-muted">
        <p className="text-xs font-semibold mb-2">⌨️ Keyboard Shortcuts:</p>
        <div className="text-xs space-y-1 text-muted-foreground">
          <p>• <kbd className="bg-background px-2 py-1 rounded border">SPACE</kbd> - Play/Pause Audio</p>
          <p>• <kbd className="bg-background px-2 py-1 rounded border">←</kbd> / <kbd className="bg-background px-2 py-1 rounded border">→</kbd> - Skip Backward/Forward 10s</p>
          <p>• <kbd className="bg-background px-2 py-1 rounded border">R</kbd> - Replay from Start</p>
          <p>• <kbd className="bg-background px-2 py-1 rounded border">T</kbd> - Toggle Repeat</p>
          <p>• <kbd className="bg-background px-2 py-1 rounded border">S</kbd> - Toggle Shuffle</p>
          <p>• <kbd className="bg-background px-2 py-1 rounded border">H</kbd> - Show Help</p>
          <p>• <kbd className="bg-background px-2 py-1 rounded border">ESC</kbd> - Go Back</p>
        </div>
      </Card>
    </div>
  );
};
