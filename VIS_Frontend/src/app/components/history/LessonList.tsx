import { Play, Clock, BookOpen, ArrowLeft } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { useEffect, useState } from 'react';
import { getLessonsByGrade, Lesson } from '../../data/historyData';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface LessonListProps {
  grade: number;
  onSelectLesson: (lessonId: number) => void;
  onBack: () => void;
}

export const LessonList = ({ grade, onSelectLesson, onBack }: LessonListProps) => {
  const lessons = getLessonsByGrade(grade);
  const [hasAnnounced, setHasAnnounced] = useState(false);

  // Voice announcement on page load - reads all lesson titles
  useEffect(() => {
    // STOP all previous speech immediately
    safeCancel();
    
    if (!hasAnnounced) {
      setHasAnnounced(true);
      setTimeout(() => {
        // Build announcement with all lesson titles and numbers
        let announcement = `Grade ${grade} Sri Lankan History. ${lessons.length} lessons available. `;
        lessons.forEach((lesson, index) => {
          announcement += `Press ${index + 1} for ${lesson.title}. `;
        });
        announcement += 'Press H for help, or Escape to go back.';
        
        safeSpeak(announcement);
      }, 500);
    }
    
    // Cleanup: stop speech when leaving page
    return () => {
      safeCancel();
    };
  }, [hasAnnounced, lessons, grade]);

  // Keyboard shortcuts for number-based lesson selection
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Number keys 1-9 to select lessons
      const num = parseInt(e.key);
      if (num >= 1 && num <= lessons.length) {
        e.preventDefault();
        const selectedLesson = lessons[num - 1];
        safeSpeak(
          `${selectedLesson.title} selected. Duration ${selectedLesson.duration}. Loading lesson.`,
          () => {
            setTimeout(() => onSelectLesson(selectedLesson.id), 500);
          }
        );
      }

      // H key to repeat help
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        let help = `Press a number to select a lesson. `;
        lessons.forEach((lesson, index) => {
          help += `${index + 1} for ${lesson.title}. `;
        });
        help += 'Press Escape to go back.';
        
        safeCancel();
        safeSpeak(help);
      }

      // L key to list all lessons again
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        let list = `${lessons.length} lessons available. `;
        lessons.forEach((lesson, index) => {
          list += `Lesson ${index + 1}: ${lesson.title}. ${lesson.description}. Duration ${lesson.duration}. `;
        });
        
        safeCancel();
        safeSpeak(list);
      }

      // Escape to go back
      if (e.key === 'Escape') {
        e.preventDefault();
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [lessons, onSelectLesson, onBack]);

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button onClick={onBack} variant="ghost" size="icon" aria-label="Go back">
          <ArrowLeft className="h-6 w-6" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl">Grade {grade} Sri Lankan History</h1>
          <p className="text-sm text-muted-foreground">
            Press 1-{lessons.length} to select • Press L to list all • Press H for help
          </p>
        </div>
      </div>

      {/* Lessons */}
      <div className="space-y-4">
        {lessons.map((lesson, index) => (
          <Card
            key={lesson.id}
            className="overflow-hidden transition-all hover:shadow-lg"
          >
            <button
              onClick={() => onSelectLesson(lesson.id)}
              className="w-full p-6 text-left"
              aria-label={`Open ${lesson.title} lesson`}
            >
              <div className="flex gap-4">
                <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-orange-500">
                  <span className="text-xl text-white" aria-hidden="true">
                    {index + 1}
                  </span>
                </div>
                <div className="flex-1 space-y-2">
                  <h3 className="text-lg">{lesson.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    {lesson.description}
                  </p>
                  <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" aria-hidden="true" />
                      <span>{lesson.duration}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <BookOpen className="h-3 w-3" aria-hidden="true" />
                      <span>{lesson.topics.length} topics</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center">
                  <div className="rounded-full bg-primary/10 p-3">
                    <Play className="h-5 w-5 text-primary" aria-hidden="true" />
                  </div>
                </div>
              </div>
            </button>
          </Card>
        ))}
      </div>
    </div>
  );
};