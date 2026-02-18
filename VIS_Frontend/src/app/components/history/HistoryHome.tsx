import { BookOpen, Headphones, GraduationCap } from 'lucide-react';
import { Card } from '../ui/card';
import { useEffect, useRef } from 'react';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface HistoryHomeProps {
  onSelectGrade: (grade: number) => void;
}

export const HistoryHome = ({ onSelectGrade }: HistoryHomeProps) => {
  // useRef instead of state to avoid re-renders causing repeat speech
  const hasAnnounced = useRef(false);

  // Voice announcement on page load
  useEffect(() => {
    safeCancel();

    if (!hasAnnounced.current) {
      hasAnnounced.current = true;

      const timer = setTimeout(() => {
        safeSpeak(
          'AI History Teacher. Learn History with Smart Audio Lessons. Please select your grade. Press 1 for Grade 10. Press 2 for Grade 11. Press H for help.'
        );
      }, 500);

      return () => clearTimeout(timer);
    }

    return () => {
      safeCancel();
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.repeat) return; // prevent spam repeat

      if (e.key === '1') {
        e.preventDefault();
        safeCancel();
        safeSpeak('Grade 10 selected.', () => {
          setTimeout(() => onSelectGrade(10), 400);
        });
      }

      if (e.key === '2') {
        e.preventDefault();
        safeCancel();
        safeSpeak('Grade 11 selected.', () => {
          setTimeout(() => onSelectGrade(11), 400);
        });
      }

      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        safeCancel();
        safeSpeak('Press 1 for Grade 10. Press 2 for Grade 11.');
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
            <BookOpen className="h-12 w-12 text-white" aria-hidden />
          </div>
        </div>

        <h1 className="text-2xl">AI History Teacher</h1>
        <p className="text-muted-foreground">
          Press 1 for Grade 10 • Press 2 for Grade 11 • Press H for help
        </p>
      </div>

      {/* Features */}
      <Card className="p-6 space-y-4">
        <div className="flex items-center gap-3">
          <Headphones className="h-6 w-6 text-secondary" />
          <div>
            <h3 className="text-sm">AI-Generated Audio Lessons</h3>
            <p className="text-xs text-muted-foreground">
              Listen to comprehensive history lessons
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <GraduationCap className="h-6 w-6 text-secondary" />
          <div>
            <h3 className="text-sm">Curriculum Aligned</h3>
            <p className="text-xs text-muted-foreground">
              Follows Grade 10 & 11 syllabus
            </p>
          </div>
        </div>
      </Card>

      {/* Grade Selection */}
      <div className="space-y-3">
        <h2 className="text-center">Select Your Grade</h2>

        {[10, 11].map((grade) => (
          <Card key={grade} className="overflow-hidden hover:shadow-lg">
            <button
              onClick={() => onSelectGrade(grade)}
              className="w-full p-6 text-left"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-primary text-2xl text-primary-foreground">
                  {grade}
                </div>

                <div className="flex-1 space-y-1">
                  <h3 className="text-xl">Grade {grade}</h3>
                  <p className="text-sm text-muted-foreground">
                    {grade === 10
                      ? 'Ancient Civilizations • World History • Cultural Studies'
                      : 'Modern History • World Wars • Contemporary Issues'}
                  </p>
                </div>
              </div>
            </button>
          </Card>
        ))}
      </div>

      {/* Info */}
      <Card className="border-secondary bg-secondary/10 p-4">
        <p className="text-center text-sm">
          Each lesson is narrated by AI and includes key topics and historical context
        </p>
      </Card>
    </div>
  );
};
