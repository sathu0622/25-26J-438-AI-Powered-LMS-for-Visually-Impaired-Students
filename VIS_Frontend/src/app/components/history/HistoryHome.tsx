import { BookOpen, Headphones, GraduationCap } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { useEffect, useState } from 'react';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface HistoryHomeProps {
  onSelectGrade: (grade: number) => void;
}

export const HistoryHome = ({ onSelectGrade }: HistoryHomeProps) => {
  const [hasAnnounced, setHasAnnounced] = useState(false);

  // Voice announcement on page load
  useEffect(() => {
    // STOP all previous speech immediately
    safeCancel();
    
    if (!hasAnnounced) {
      setHasAnnounced(true);
      setTimeout(() => {
        safeSpeak(
          'AI History Teacher. Learn History with Smart Audio Lessons. Please select your grade. Press 1 for Grade 10: Ancient Civilizations, World History, and Cultural Studies. Press 2 for Grade 11: Modern History, World Wars, and Contemporary Issues. You can also press Escape to go back to home.'
        );
      }, 500);
    }
    
    // Cleanup: stop speech when leaving page
    return () => {
      safeCancel();
    };
  }, [hasAnnounced]);

  // Keyboard shortcuts for number-based selection
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // 1 key for Grade 10
      if (e.key === '1') {
        e.preventDefault();
        safeCancel();
        safeSpeak('Grade 10 selected. Loading lessons.', () => {
          setTimeout(() => onSelectGrade(10), 500);
        });
      }

      // 2 key for Grade 11
      if (e.key === '2') {
        e.preventDefault();
        safeCancel();
        safeSpeak('Grade 11 selected. Loading lessons.', () => {
          setTimeout(() => onSelectGrade(11), 500);
        });
      }

      // H key to repeat help
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        safeCancel();
        safeSpeak('Press 1 for Grade 10, Press 2 for Grade 11, Press Escape to go back.');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [onSelectGrade]);

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-orange-500 p-6">
            <BookOpen className="h-12 w-12 text-white" aria-hidden="true" />
          </div>
        </div>
        <h1 className="text-2xl">AI History Teacher</h1>
        <p className="text-muted-foreground">
          Press 1 for Grade 10 • Press 2 for Grade 11 • Press H for help
        </p>
      </div>

      {/* Features */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Headphones className="h-6 w-6 text-secondary" aria-hidden="true" />
            <div>
              <h3 className="text-sm">AI-Generated Audio Lessons</h3>
              <p className="text-xs text-muted-foreground">
                Listen to comprehensive history lessons
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <GraduationCap className="h-6 w-6 text-secondary" aria-hidden="true" />
            <div>
              <h3 className="text-sm">Curriculum Aligned</h3>
              <p className="text-xs text-muted-foreground">
                Follows Grade 10 & 11 syllabus
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Grade Selection */}
      <div className="space-y-3">
        <h2 className="text-center">Select Your Grade</h2>
        <div className="grid gap-4">
          <Card className="overflow-hidden transition-all hover:shadow-lg">
            <button
              onClick={() => onSelectGrade(10)}
              className="w-full p-6 text-left"
              aria-label="Select Grade 10"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-primary text-2xl text-primary-foreground">
                  10
                </div>
                <div className="flex-1 space-y-1">
                  <h3 className="text-xl">Grade 10</h3>
                  <p className="text-sm text-muted-foreground">
                    Ancient Civilizations • World History • Cultural Studies
                  </p>
                </div>
              </div>
            </button>
          </Card>

          <Card className="overflow-hidden transition-all hover:shadow-lg">
            <button
              onClick={() => onSelectGrade(11)}
              className="w-full p-6 text-left"
              aria-label="Select Grade 11"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-secondary text-2xl text-secondary-foreground">
                  11
                </div>
                <div className="flex-1 space-y-1">
                  <h3 className="text-xl">Grade 11</h3>
                  <p className="text-sm text-muted-foreground">
                    Modern History • World Wars • Contemporary Issues
                  </p>
                </div>
              </div>
            </button>
          </Card>
        </div>
      </div>

      {/* Info Card */}
      <Card className="border-secondary bg-secondary/10 p-4">
        <p className="text-center text-sm">
          Each lesson is narrated by AI and includes key topics, important dates, and
          historical context
        </p>
      </Card>
    </div>
  );
};