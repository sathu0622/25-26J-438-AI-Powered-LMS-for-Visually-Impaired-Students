import { useEffect, useState } from 'react';
import { ArrowLeft, BookOpen, Clock } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface Chapter {
  id: number;
  chapter_name: string;
  grade: number;
  topic_count: number;
}

interface ChapterListProps {
  grade: number;
  onSelectChapter: (chapterId: number, chapterName: string) => void;
  onBack: () => void;
}

export const ChapterList = ({ grade, onSelectChapter, onBack }: ChapterListProps) => {
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasAnnounced, setHasAnnounced] = useState(false);

  // Fetch chapters from backend
  useEffect(() => {
    const fetchChapters = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8000/api/chapters/${grade}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch chapters: ${response.statusText}`);
        }
        
        const data = await response.json();
        setChapters(data.chapters);
        setError(null);
      } catch (err) {
        console.error('Error fetching chapters:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chapters');
        safeSpeak(`Error loading chapters. Please try again.`);
      } finally {
        setLoading(false);
      }
    };

    fetchChapters();
  }, [grade]);

  // Voice announcement
  useEffect(() => {
    if (!loading && !hasAnnounced) {
      safeCancel();
      setHasAnnounced(true);

      if (chapters.length === 0) {
        safeSpeak('No chapters available for this grade.');
        return;
      }

      setTimeout(() => {
        let announcement = `Grade ${grade}. ${chapters.length} chapters available. `;
        chapters.forEach((chapter, index) => {
          announcement += `Press ${index + 1} for ${chapter.chapter_name}. ${chapter.topic_count} topics. `;
        });
        announcement += 'Press H for help, or Escape to go back.';
        safeSpeak(announcement);
      }, 500);
    }

    return () => {
      safeCancel();
    };
  }, [loading, chapters, hasAnnounced, grade]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const num = parseInt(e.key);
      if (num >= 1 && num <= chapters.length) {
        e.preventDefault();
        const selectedChapter = chapters[num - 1];
        safeSpeak(`${selectedChapter.chapter_name} selected. Loading topics.`, () => {
          setTimeout(() => onSelectChapter(selectedChapter.id, selectedChapter.chapter_name), 500);
        });
      }

      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        let help = `Press a number 1 to ${chapters.length} to select a chapter. `;
        safeCancel();
        safeSpeak(help);
      }

      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        let list = `${chapters.length} chapters available. `;
        chapters.forEach((chapter, index) => {
          list += `Chapter ${index + 1}: ${chapter.chapter_name}. ${chapter.topic_count} topics. `;
        });
        safeCancel();
        safeSpeak(list);
      }

      if (e.key === 'Escape') {
        e.preventDefault();
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [chapters, onSelectChapter, onBack]);

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button onClick={onBack} variant="ghost" size="icon" aria-label="Go back">
          <ArrowLeft className="h-6 w-6" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl">Grade {grade} - Sri Lankan History</h1>
          <p className="text-sm text-muted-foreground">
            Press 1-{chapters.length} to select chapter • Press H for help
          </p>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <Card className="p-6">
          <p className="text-center text-muted-foreground">Loading chapters...</p>
        </Card>
      )}

      {/* Error State */}
      {error && (
        <Card className="border-red-500 bg-red-50 p-6">
          <p className="text-red-800">{error}</p>
        </Card>
      )}

      {/* Chapters List */}
      {!loading && chapters.length > 0 && (
        <div className="space-y-4">
          {chapters.map((chapter, index) => (
            <Card
              key={chapter.id}
              className="overflow-hidden transition-all hover:shadow-lg"
            >
              <button
                onClick={() => onSelectChapter(chapter.id, chapter.chapter_name)}
                className="w-full p-6 text-left"
                aria-label={`Open ${chapter.chapter_name}`}
              >
                <div className="flex gap-4">
                  <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-primary">
                    <span className="text-xl font-bold text-primary-foreground">
                      {index + 1}
                    </span>
                  </div>
                  <div className="flex-1 space-y-2">
                    <h3 className="text-lg">{chapter.chapter_name}</h3>
                    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <BookOpen className="h-3 w-3" />
                        <span>{chapter.topic_count} topics</span>
                      </div>
                    </div>
                  </div>
                </div>
              </button>
            </Card>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && chapters.length === 0 && !error && (
        <Card className="p-6">
          <p className="text-center text-muted-foreground">
            No chapters available for Grade {grade}
          </p>
        </Card>
      )}
    </div>
  );
};
