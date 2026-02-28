import { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ArrowLeft,
  Volume2,
  BookOpen,
} from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Slider } from '../ui/slider';
import { getLessonById } from '../../data/historyData';
import { useTTS } from '../../contexts/TTSContext';

interface LessonPlayerProps {
  lessonId: number;
  onBack: () => void;
}

export const LessonPlayer = ({ lessonId, onBack }: LessonPlayerProps) => {
  const lesson = getLessonById(lessonId);
  
  // Fallback if lesson not found
  if (!lesson) {
    return (
      <div className="mx-auto max-w-4xl p-4">
        <p className="text-center text-muted-foreground">Lesson not found</p>
        <Button onClick={onBack} className="mt-4">Go Back</Button>
      </div>
    );
  }

  const { speak, cancel, isSpeaking } = useTTS();
  const [currentTopicIndex, setCurrentTopicIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [hasAnnounced, setHasAnnounced] = useState(false);
  const hasAnnouncedRef = useRef(false);
  const lastTopicIndexRef = useRef<number>(-1);

  const currentTopic = lesson.topics[currentTopicIndex];

  useEffect(() => {
    cancel();
    setHasAnnounced(false);
    hasAnnouncedRef.current = false;

    const announcement = `${lesson.title}. Topic ${currentTopicIndex + 1} of ${lesson.topics.length}. ${currentTopic.title}. Press Space to play or pause. Press Right Arrow for next topic. Press Left Arrow for previous topic. Press number keys 1 to ${lesson.topics.length} to jump to specific topics. Press H for help.`;

    const t = setTimeout(() => {
      speak(announcement, {
        interrupt: true,
        onEnd: () => {
          hasAnnouncedRef.current = true;
          lastTopicIndexRef.current = currentTopicIndex;
          setHasAnnounced(true);
          speak(currentTopic.content, { interrupt: false });
        },
      });
    }, 500);

    return () => {
      clearTimeout(t);
      cancel();
    };
  }, [lessonId]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === ' ') {
        e.preventDefault();
        handlePlayPause();
        speak(isSpeaking ? 'Paused' : 'Playing', { interrupt: true });
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (currentTopicIndex > 0) {
          handlePrevious();
          cancel();
          speak(`Previous topic. ${lesson.topics[currentTopicIndex - 1].title}`, { interrupt: true });
        } else {
          cancel();
          speak('Already at first topic', { interrupt: true });
        }
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (currentTopicIndex < lesson.topics.length - 1) {
          handleNext();
          cancel();
          speak(`Next topic. ${lesson.topics[currentTopicIndex + 1].title}`, { interrupt: true });
        } else {
          cancel();
          speak('Already at last topic', { interrupt: true });
        }
      }
      const num = parseInt(e.key);
      if (num >= 1 && num <= lesson.topics.length) {
        e.preventDefault();
        handleSelectTopic(num - 1);
        cancel();
        speak(`Topic ${num}. ${lesson.topics[num - 1].title}`, { interrupt: true });
      }
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        cancel();
        speak(
          `Press Space to play or pause. Press Left Arrow for previous topic. Press Right Arrow for next topic. Press numbers 1 to ${lesson.topics.length} to jump to topics. Press Escape to go back.`,
          { interrupt: true }
        );
      }
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        let list = `${lesson.topics.length} topics in this lesson. `;
        lesson.topics.forEach((topic, index) => {
          list += `${index + 1}. ${topic.title}. `;
        });
        cancel();
        speak(list, { interrupt: true });
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        cancel();
        onBack();
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentTopicIndex, lesson, isSpeaking, onBack, speak, cancel]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSpeaking) {
      interval = setInterval(() => setProgress((prev) => Math.min(prev + 1, 100)), 100);
    }
    return () => clearInterval(interval);
  }, [isSpeaking]);

  useEffect(() => {
    if (!hasAnnounced) return;
    if (lastTopicIndexRef.current === currentTopicIndex) return;
    lastTopicIndexRef.current = currentTopicIndex;
    cancel();
    setProgress(0);
    const timer = setTimeout(() => speak(currentTopic.content, { interrupt: false }), 500);
    return () => clearTimeout(timer);
  }, [currentTopicIndex, currentTopic.content, speak, cancel, hasAnnounced]);

  const handlePlayPause = () => {
    cancel();
    if (isSpeaking) {
      // Pause = stop
    } else {
      speak(currentTopic.content, { interrupt: false });
    }
  };

  const handlePrevious = () => {
    if (currentTopicIndex > 0) {
      cancel();
      setCurrentTopicIndex(currentTopicIndex - 1);
      setProgress(0);
    }
  };

  const handleNext = () => {
    if (currentTopicIndex < lesson.topics.length - 1) {
      cancel();
      setCurrentTopicIndex(currentTopicIndex + 1);
      setProgress(0);
    }
  };

  const handleSelectTopic = (index: number) => {
    cancel();
    setCurrentTopicIndex(index);
    setProgress(0);
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button onClick={onBack} variant="ghost" size="icon" aria-label="Go back">
          <ArrowLeft className="h-6 w-6" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl">{lesson.title}</h1>
          <p className="text-sm text-muted-foreground">
            Space: Play/Pause • 1-{lesson.topics.length}: Jump • Arrows: Navigate • H: Help
          </p>
        </div>
      </div>

      {/* Current Topic */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Volume2 className="h-6 w-6 text-orange-500" aria-hidden="true" />
            <h2 className="text-lg">{currentTopic.title}</h2>
          </div>
          <p className="leading-relaxed text-muted-foreground">
            {currentTopic.content}
          </p>
        </div>
      </Card>

      {/* Audio Player Controls */}
      <Card className="p-6">
        <div className="space-y-6">
          {/* Progress Bar */}
          <div className="space-y-2">
            <Slider
              value={[progress]}
              onValueChange={(value) => setProgress(value[0])}
              max={100}
              step={1}
              className="w-full"
              aria-label="Audio progress"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {isSpeaking ? 'Playing' : 'Ready'}
              </span>
              <span>{progress}%</span>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center justify-center gap-4">
            <Button
              onClick={handlePrevious}
              disabled={currentTopicIndex === 0}
              variant="outline"
              size="icon"
              className="h-12 w-12"
              aria-label="Previous topic"
            >
              <SkipBack className="h-5 w-5" />
            </Button>

            <Button
              onClick={handlePlayPause}
              size="icon"
              className="h-16 w-16"
              aria-label={isSpeaking ? 'Pause' : 'Play'}
            >
              {isSpeaking ? (
                <Pause className="h-8 w-8" />
              ) : (
                <Play className="h-8 w-8" />
              )}
            </Button>

            <Button
              onClick={handleNext}
              disabled={currentTopicIndex === lesson.topics.length - 1}
              variant="outline"
              size="icon"
              className="h-12 w-12"
              aria-label="Next topic"
            >
              <SkipForward className="h-5 w-5" />
            </Button>
          </div>

          {/* Demo Mode Label */}
          <div className="text-center text-xs text-muted-foreground">
            🎙️ Demo Mode: Simulated AI Voice Lesson
          </div>
        </div>
      </Card>

      {/* Topic List */}
      <div className="space-y-3">
        <h3 className="text-sm text-muted-foreground">Topics in this lesson</h3>
        <div className="space-y-2">
          {lesson.topics.map((topic, index) => (
            <Card
              key={topic.id}
              className={`overflow-hidden transition-all ${
                index === currentTopicIndex
                  ? 'border-primary bg-primary/5'
                  : 'hover:shadow-md'
              }`}
            >
              <button
                onClick={() => handleSelectTopic(index)}
                className="w-full p-4 text-left"
                aria-label={`Play ${topic.title}`}
                aria-current={index === currentTopicIndex ? 'true' : undefined}
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm ${
                      index === currentTopicIndex
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm">{topic.title}</p>
                  </div>
                  {index === currentTopicIndex && (
                    <Volume2 className="h-4 w-4 text-primary" aria-hidden="true" />
                  )}
                </div>
              </button>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};