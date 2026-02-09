import { useState, useEffect } from 'react';
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
import { useMockSpeechSynthesis } from '../../hooks/useMockSpeechSynthesis';
import { getLessonById } from '../../data/historyData';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

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

  const [currentTopicIndex, setCurrentTopicIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [hasAnnounced, setHasAnnounced] = useState(false);
  const { speak, pause, resume, stop, isSpeaking, isPaused } = useMockSpeechSynthesis();

  const currentTopic = lesson.topics[currentTopicIndex];

  // STOP all previous speech when component mounts and announce commands
  useEffect(() => {
    safeCancel();
    stop(); // Also stop hook-based speech
    setHasAnnounced(false);
    
    let announcementTimer: NodeJS.Timeout;
    let autoPlayTimer: NodeJS.Timeout;
    
    // Announce lesson and instructions
    announcementTimer = setTimeout(() => {
      safeSpeak(
        `${lesson.title}. Topic ${currentTopicIndex + 1} of ${lesson.topics.length}. ${currentTopic.title}. Press Space to play or pause. Press Right Arrow for next topic. Press Left Arrow for previous topic. Press number keys 1 to ${lesson.topics.length} to jump to specific topics. Press H for help.`
      );
      setHasAnnounced(true);
      
      // Auto-play after announcement finishes (calculated based on announcement length)
      const announcementWords = 35; // Approximate word count
      const announcementDuration = (announcementWords / 2.5) * 1000;
      autoPlayTimer = setTimeout(() => {
        safeCancel(); // Cancel any lingering speech
        speak(currentTopic.content);
      }, announcementDuration + 500); // Add 500ms buffer
    }, 500);
    
    // Cleanup: stop speech when leaving page
    return () => {
      clearTimeout(announcementTimer);
      clearTimeout(autoPlayTimer);
      safeCancel();
      stop();
    };
  }, [lessonId]); // Only re-run when lesson changes

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Space bar to play/pause
      if (e.key === ' ') {
        e.preventDefault();
        handlePlayPause();
        const msg = isSpeaking && !isPaused ? 'Paused' : 'Playing';
        stop(); // Stop any hook speech
        setTimeout(() => safeSpeak(msg), 100);
      }

      // Arrow keys for navigation
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (currentTopicIndex > 0) {
          handlePrevious();
          stop(); // Stop hook speech before announcement
          safeCancel();
          safeSpeak(`Previous topic. ${lesson.topics[currentTopicIndex - 1].title}`);
        } else {
          stop(); // Stop hook speech before announcement
          safeCancel();
          safeSpeak('Already at first topic');
        }
      }

      if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (currentTopicIndex < lesson.topics.length - 1) {
          handleNext();
          stop(); // Stop hook speech before announcement
          safeCancel();
          safeSpeak(`Next topic. ${lesson.topics[currentTopicIndex + 1].title}`);
        } else {
          stop(); // Stop hook speech before announcement
          safeCancel();
          safeSpeak('Already at last topic');
        }
      }

      // Number keys to jump to specific topics
      const num = parseInt(e.key);
      if (num >= 1 && num <= lesson.topics.length) {
        e.preventDefault();
        handleSelectTopic(num - 1);
        stop(); // Stop hook speech before announcement
        safeCancel();
        safeSpeak(`Topic ${num}. ${lesson.topics[num - 1].title}`);
      }

      // H key for help
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        stop(); // Stop hook speech before announcement
        safeCancel();
        safeSpeak(
          `Press Space to play or pause. Press Left Arrow for previous topic. Press Right Arrow for next topic. Press numbers 1 to ${lesson.topics.length} to jump to topics. Press Escape to go back.`
        );
      }

      // L key to list all topics
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        let list = `${lesson.topics.length} topics in this lesson. `;
        lesson.topics.forEach((topic, index) => {
          list += `${index + 1}. ${topic.title}. `;
        });
        stop(); // Stop hook speech before announcement
        safeCancel();
        safeSpeak(list);
      }

      // Escape to go back
      if (e.key === 'Escape') {
        e.preventDefault();
        safeCancel();
        stop();
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentTopicIndex, lesson, isSpeaking, isPaused, onBack, stop]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSpeaking && !isPaused) {
      interval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 1, 100));
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isSpeaking, isPaused]);

  // Handle topic changes - stop current speech and play new topic
  useEffect(() => {
    // Skip initial mount (handled by first useEffect)
    if (hasAnnounced) {
      // Stop all speech
      safeCancel();
      stop();
      
      // Reset progress
      setProgress(0);
      
      // Play new topic after brief delay
      const timer = setTimeout(() => {
        speak(currentTopic.content);
      }, 500);

      return () => {
        clearTimeout(timer);
      };
    }
  }, [currentTopicIndex, currentTopic.content, stop, speak, hasAnnounced]);

  const handlePlayPause = () => {
    safeCancel(); // Stop any safeSpeak announcements
    if (isSpeaking && !isPaused) {
      pause();
    } else if (isPaused) {
      resume();
    } else {
      speak(currentTopic.content);
    }
  };

  const handlePrevious = () => {
    if (currentTopicIndex > 0) {
      stop(); // Stop current speech
      safeCancel();
      setCurrentTopicIndex(currentTopicIndex - 1);
      setProgress(0);
    }
  };

  const handleNext = () => {
    if (currentTopicIndex < lesson.topics.length - 1) {
      stop(); // Stop current speech
      safeCancel();
      setCurrentTopicIndex(currentTopicIndex + 1);
      setProgress(0);
    }
  };

  const handleSelectTopic = (index: number) => {
    stop(); // Stop current speech
    safeCancel();
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
                {isSpeaking ? (isPaused ? 'Paused' : 'Playing') : 'Ready'}
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
              aria-label={isSpeaking && !isPaused ? 'Pause' : 'Play'}
            >
              {isSpeaking && !isPaused ? (
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