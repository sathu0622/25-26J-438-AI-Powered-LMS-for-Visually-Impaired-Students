import { useEffect, useState } from 'react';
import { Volume2, Loader2, Brain } from 'lucide-react';
import { adaptiveService } from '../../services/adaptiveService';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { useTTS } from '../../contexts/TTSContext';

interface AdaptiveStartProps {
  onStart: (chapter: string) => void;
  onBack?: () => void;
}

export const AdaptiveStart = ({ onStart, onBack }: AdaptiveStartProps) => {
  const [chapters, setChapters] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const { speak, cancel } = useTTS();

  // Load chapters from backend
  useEffect(() => {
    const loadChapters = async () => {
      try {
        const res = await adaptiveService.getChapters();
        setChapters(res);

        // 🔊 Speak instructions after chapters load
        setTimeout(() => {
          speak(
            `Adaptive Quiz chapters loaded. ${res.length} chapters available. Use Up and Down arrow keys or number keys 1 to 9 to select a chapter. Press Enter to start. Press H for help. Press B to go back.`
          );
        }, 600);
      } catch (err) {
        console.error('Failed to load adaptive chapters', err);
      } finally {
        setLoading(false);
      }
    };

    loadChapters();
  }, []);

  const handleChapterSelect = (index: number) => {
    setSelectedIndex(index);
    cancel();
    speak(`${chapters[index]} selected.`);
  };

  const startQuiz = (index: number) => {
    cancel();
    onStart(chapters[index]);
  };

  // ✅ KEYBOARD NAVIGATION
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (loading || chapters.length === 0) return;

      // 🔢 Number selection (1–9)
      if (/^[1-9]$/.test(e.key)) {
        const index = parseInt(e.key) - 1;

        if (chapters[index]) {
          handleChapterSelect(index);

          // 🚀 Auto-start after short delay
          setTimeout(() => startQuiz(index), 700);
        }
      }

      // ⬆ Arrow Up
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex = prev === null ? 0 : (prev - 1 + chapters.length) % chapters.length;
          speak(chapters[newIndex]);
          return newIndex;
        });
      }

      // ⬇ Arrow Down
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex = prev === null ? 0 : (prev + 1) % chapters.length;
          speak(chapters[newIndex]);
          return newIndex;
        });
      }

      // ⏎ Enter to start
      if (e.key === 'Enter' && selectedIndex !== null) {
        startQuiz(selectedIndex);
      }

      // 🆘 Help instructions
      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        cancel();
        const currentChapter = selectedIndex !== null ? `Currently on chapter ${selectedIndex + 1}: ${chapters[selectedIndex]}. ` : '';
        speak(
          `Adaptive Quiz Help. ${currentChapter}Use Up and Down arrow keys to navigate between ${chapters.length} chapters. Press number keys 1 to 9 for quick selection. Press Enter to start adaptive quiz with the selected chapter. Press B to go back. In adaptive quiz, questions will adjust difficulty based on your performance.`,
          { interrupt: true }
        );
      }

      // 🔙 Back to quiz mode selection
      if (e.key.toLowerCase() === 'b' && onBack) {
        e.preventDefault();
        cancel();
        speak('Going back to quiz mode selection.', {
          onEnd: () => onBack()
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [chapters, loading, selectedIndex, speak, cancel, onStart, onBack]);

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center gap-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading adaptive chapters…
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-purple-500 p-6">
            <Brain className="h-12 w-12 text-white" />
          </div>
        </div>
        <h1 className="text-2xl font-bold">Adaptive Quiz</h1>
        <p className="text-sm text-muted-foreground">
          Questions adjust difficulty based on your performance
        </p>
      </div>

      {/* Navigation Help */}
      <div className="mb-4 p-3 bg-purple-50 border border-purple-200 rounded-lg" role="complementary" aria-labelledby="nav-help">
        <h2 id="nav-help" className="text-sm font-semibold mb-1">Keyboard Navigation</h2>
        <p className="text-xs text-muted-foreground">
          Up/Down arrows or 1-{chapters.length} to select chapters, Enter to start, H for help, B to go back
        </p>
        {selectedIndex !== null && (
          <p className="text-xs text-purple-600 mt-1" role="status" aria-live="polite">
            Currently selected: Chapter {selectedIndex + 1} of {chapters.length} - {chapters[selectedIndex]}
          </p>
        )}
      </div>

      <div className="space-y-3">
        <h2 className="text-center">
          Select a Chapter (Use ↑ ↓ or 1-{chapters.length})
        </h2>

        <div className="grid gap-2">
          {chapters.map((chapter, index) => (
            <Card
              key={index}
              className={`overflow-hidden transition-all cursor-pointer ${
                selectedIndex === index
                  ? 'border-purple-500 bg-purple-50 border-2'
                  : 'hover:shadow-md hover:border-purple-300'
              }`}
              role="option"
              aria-selected={selectedIndex === index}
              tabIndex={selectedIndex === index ? 0 : -1}
              onClick={() => handleChapterSelect(index)}
              onFocus={() => {
                if (selectedIndex !== index) {
                  handleChapterSelect(index);
                }
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  startQuiz(index);
                }
              }}
            >
              <button className="w-full p-4 text-left">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-100 text-purple-700 text-sm font-semibold">
                    {index + 1}
                  </div>
                  <div>
                    <span className="font-medium">{chapter}</span>
                    <p className="text-xs text-muted-foreground">Adaptive difficulty adjustment</p>
                  </div>
                  {selectedIndex === index && (
                    <span className="ml-auto text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
                      Selected
                    </span>
                  )}
                </div>
              </button>
            </Card>
          ))}
        </div>
      </div>

      <Button
        onClick={() => selectedIndex !== null && startQuiz(selectedIndex)}
        disabled={selectedIndex === null}
        size="lg"
        className="w-full min-h-[64px] bg-purple-600 hover:bg-purple-700"
      >
        <Brain className="mr-2 h-6 w-6" />
        {selectedIndex !== null
          ? 'Press Enter or Click to Start Adaptive Quiz'
          : 'Select a Chapter First'}
      </Button>

      <Button
        variant="outline"
        className="w-full"
        onClick={() => {
          cancel();
          speak(`You have ${chapters.length} chapters available. ${chapters.join(', ')}`);
        }}
      >
        <Volume2 className="mr-2 h-4 w-4" /> Read All Chapters Aloud
      </Button>

      <div className="sr-only" aria-live="polite">
        {selectedIndex !== null ? `Selected ${chapters[selectedIndex]}` : 'No chapter selected'}
      </div>
    </div>
  );
};
