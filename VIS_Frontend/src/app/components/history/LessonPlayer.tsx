import { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
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
import { useTTS } from '../../contexts/TTSContext';
import { API_BASE_URL } from '../../services/historyLessonService';

interface LessonPlayerProps {
  topicName: string;
  content: string;
  grade: number;
  chapterIdx: number;
  topicIdx: number;
  autoPlay?: boolean;
  onBack: () => void;
  onGoToGradeChapters: (grade: number) => void;
}

export const LessonPlayer = ({
  topicName,
  content,
  grade,
  chapterIdx,
  topicIdx,
  autoPlay = true,
  onBack,
  onGoToGradeChapters
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
  const [playbackSpeed, setPlaybackSpeed] = useState(0.9);
  const recognitionRef = useRef<any>(null);
  const isListeningRef = useRef(false);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isComponentActiveRef = useRef(true);
  const { speak, cancel } = useTTS();

  useEffect(() => {
    isComponentActiveRef.current = true;
    return () => {
      isComponentActiveRef.current = false;
      cancel();
    };
  }, [cancel]);

  useEffect(() => {
    cancel();
    setHasAnnounced(false);
    generateAudio();
  }, [grade, chapterIdx, topicIdx, cancel]);

  useEffect(() => {
    if (autoPlay && audioUrl && audioRef.current && !isLoading && !hasAnnounced) {
      const timer = setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.playbackRate = playbackSpeed;
          audioRef.current.play().then(() => {
            setIsPlaying(true);
            setHasAnnounced(true);
          }).catch(err => {
            console.error('Auto-play failed:', err);
            setHasAnnounced(true);
          });
        }
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [audioUrl, autoPlay, isLoading, topicName, hasAnnounced, playbackSpeed]);

  const generateAudio = async () => {
    const endpoint = `${API_BASE_URL}/api/audio/chapter/${grade}/${chapterIdx}/${topicIdx}?t=${Date.now()}`;

    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(endpoint, {
        method: 'GET',
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`Failed to generate audio: ${response.statusText}`);
      }

      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      speakIfActive(`Audio generated successfully.`);
    } catch (err) {
      console.error('Audio generation error:', err);
      setAudioUrl(endpoint);
      setError(null);
      speakIfActive('Audio ready. Press space to play.');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleReplay = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
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
      audioRef.current.playbackRate = playbackSpeed;
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
    }
  };

  const handleSkipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(audioRef.current.currentTime - 10, 0);
    }
  };

  const toggleRepeat = () => setIsRepeat(!isRepeat);
  const toggleShuffle = () => setIsShuffle(!isShuffle);

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    if (audioRef.current) {
      audioRef.current.playbackRate = speed;
    }
  };

  const formatTime = (time: number) => {
    if (isNaN(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const speakIfActive = (text: string, onEnd?: () => void) => {
    if (isComponentActiveRef.current) {
      speak(text, { interrupt: true, onEnd: onEnd ?? (() => {}) });
    }
  };

  const handleVoiceCommand = (transcript: string) => {
    if (!isComponentActiveRef.current) return;
    const normalized = transcript.toLowerCase().trim();
    const audioEl = audioRef.current;
    if (!normalized) return;

    cancel();

    if (normalized.includes('hello')) {
      if (audioEl && !audioEl.paused) {
        audioEl.pause();
        setIsPlaying(false);
      }
      speakIfActive('Yes, say dear.');
      return;
    }

    if (normalized.includes('stop speech')) {
      if (audioEl && !audioEl.paused) {
        audioEl.pause();
        setIsPlaying(false);
      }
      speakIfActive("Okay, I'm silance now, say me what to do?");
      return;
    }

    const wrongContextRequest =
      normalized.includes('wrong lesson') ||
      normalized.includes('wrong page') ||
      normalized.includes('wrong chapter') ||
      normalized.includes('wrong topic') ||
      normalized.includes('wrong grade') ||
      (normalized.includes('wrong') && normalized.includes('go back')) ||
      normalized.includes('please go back');

    if (wrongContextRequest) {
      handleStop();
      speakIfActive('Ok, going back.', () => {
        setTimeout(() => onBack(), 250);
      });
      return;
    }

    if (normalized.includes('stop') || normalized.includes('pause')) {
      if (audioEl && !audioEl.paused) {
        audioEl.pause();
        setIsPlaying(false);
      }
      return;
    }

    if (normalized.includes('play') || normalized.includes('resume') || normalized.includes('start')) {
      if (audioEl && audioEl.paused) {
        audioEl.play().catch(err => console.error('Play failed:', err));
        setIsPlaying(true);
      }
      return;
    }

    const wantsChapterPageAgain =
      normalized.includes('go to the chapters') ||
      normalized.includes('go to chapters') ||
      normalized.includes('chapters again') ||
      normalized.includes('chapter page') ||
      normalized.includes('open chapters') ||
      normalized.includes('show chapters');

    if (wantsChapterPageAgain) {
      handleStop();
      speakIfActive(`Opening Grade ${grade} chapters.`);
      setTimeout(() => onGoToGradeChapters(grade), 300);
      return;
    }

    if (normalized.includes('go back') || normalized.includes('back') || normalized.includes('previous page')) {
      handleStop();
      onBack();
      return;
    }

    const wantsChapters = normalized.includes('chapter');
    const hasGrade10 = normalized.includes('grade 10') || normalized.includes('grade ten') || /\b10\b/.test(normalized);
    const hasGrade11 = normalized.includes('grade 11') || normalized.includes('grade eleven') || /\b11\b/.test(normalized);

    if (wantsChapters && hasGrade10) {
      handleStop();
      speakIfActive('Opening Grade 10 chapters.');
      setTimeout(() => onGoToGradeChapters(10), 300);
      return;
    }

    if (wantsChapters && hasGrade11) {
      handleStop();
      speakIfActive('Opening Grade 11 chapters.');
      setTimeout(() => onGoToGradeChapters(11), 300);
    }
  };

  useEffect(() => {
    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    const startListening = () => {
      if (!recognitionRef.current || isListeningRef.current) return;
      try {
        recognitionRef.current.start();
      } catch {
        if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
        restartTimeoutRef.current = setTimeout(startListening, 800);
      }
    };

    recognition.onstart = () => { isListeningRef.current = true; };
    recognition.onresult = (event: any) => {
      const latestIndex = event.results.length - 1;
      const transcript = event.results[latestIndex]?.[0]?.transcript || '';
      handleVoiceCommand(transcript);
    };
    recognition.onerror = () => {
      isListeningRef.current = false;
      if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = setTimeout(startListening, 700);
    };
    recognition.onend = () => {
      isListeningRef.current = false;
      if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = setTimeout(startListening, 700);
    };

    recognitionRef.current = recognition;
    startListening();

    return () => {
      if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
      if (recognitionRef.current) {
        try {
          recognitionRef.current.onstart = null;
          recognitionRef.current.onresult = null;
          recognitionRef.current.onerror = null;
          recognitionRef.current.onend = null;
          recognitionRef.current.stop();
        } catch { /* ignore */ }
      }
      isListeningRef.current = false;
      recognitionRef.current = null;
    };
  }, [onBack, onGoToGradeChapters]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === ' ') {
        e.preventDefault();
        handlePlayPause();
      }
      if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleReplay();
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        handleSkipBackward();
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        handleSkipForward();
      }
      if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        toggleRepeat();
      }
      if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        toggleShuffle();
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        cancel();
        handleStop();
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isPlaying, topicName, onBack, isRepeat, isShuffle]);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 pb-24">
      <div className="flex items-center gap-4">
        <Button onClick={onBack} variant="ghost" size="icon" aria-label="Go back" title="Press ESC to go back">
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

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card className="p-6 bg-gradient-to-br from-gray-900 to-gray-800 text-white">
        <div className="space-y-6">
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

          <audio
            ref={audioRef}
            src={audioUrl || undefined}
            onEnded={handleAudioEnded}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            className="hidden"
            aria-label="Lesson audio player"
          />

          <div className="text-center text-xs text-gray-400 pt-2">
            🎙️ AI-Powered Voice Lesson
          </div>
        </div>
      </Card>

      {audioUrl && !isLoading && (
        <Button onClick={generateAudio} variant="secondary" className="w-full">
          Regenerate Audio
        </Button>
      )}

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
