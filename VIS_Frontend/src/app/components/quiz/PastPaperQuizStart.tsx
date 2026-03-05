import { useState, useEffect } from 'react';
import { Play, BookOpen, Calendar } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { useTTS } from '../../contexts/TTSContext';
import { pastPaperService } from '../../services/pastPaperService';

interface PastPaperQuizStartProps {
  onStart: (topic: string) => void;
  onBack: () => void;
}

export const PastPaperQuizStart = ({ onStart, onBack }: PastPaperQuizStartProps) => {
  const { speak, cancel } = useTTS();
  const [chapters, setChapters] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load past paper chapters from backend
  useEffect(() => {
    const loadPastPaperChapters = async () => {
      try {
        const chapters = await pastPaperService.getChapters();
        setChapters(chapters);

        // 🔊 Speak instructions after chapters load
        setTimeout(() => {
          speak(
            `Past Paper Quiz chapters loaded. ${chapters.length} chapters available. Use Up and Down arrow keys or number keys 1 to 9 to select a chapter. Press Enter to start. Press H for help. Press B to go back.`
          );
        }, 600);
      } catch (err) {
        console.error('Failed to load past paper chapters', err);
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
        setError(errorMessage);
        speak(`Error loading past paper chapters: ${errorMessage}. Press B to go back.`);
      } finally {
        setLoading(false);
      }
    };

    loadPastPaperChapters();
  }, []);

  const handleChapterSelect = (index: number) => {
    setSelectedIndex(index);
    cancel();
    speak(`${chapters[index]} selected.`);
  };

  const startPastPaperQuiz = (index: number) => {
    cancel();
    speak(`Starting Past Paper Quiz for ${chapters[index]}. Loading examination questions from previous years.`, { 
      onEnd: () => {
        onStart(chapters[index]);
      }
    });
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
          setTimeout(() => startPastPaperQuiz(index), 700);
        }
      }

      // ⬆ Arrow Up - Navigate to previous chapter
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex = prev === null ? 0 : (prev - 1 + chapters.length) % chapters.length;
          speak(chapters[newIndex]);
          return newIndex;
        });
      }

      // ⬇ Arrow Down - Navigate to next chapter
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex = prev === null ? 0 : (prev + 1) % chapters.length;
          speak(chapters[newIndex]);
          return newIndex;
        });
      }

      // ↵ Enter to start practice with selected chapter
      if (e.key === 'Enter' && selectedIndex !== null) {
        e.preventDefault();
        startPastPaperQuiz(selectedIndex);
      }

      // 🆘 Help instructions
      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        cancel();
        const currentChapter = selectedIndex !== null ? `Currently on chapter ${selectedIndex + 1}: ${chapters[selectedIndex]}. ` : '';
        speak(
          `Past Paper Quiz Help. ${currentChapter}Use Up and Down arrow keys to navigate between ${chapters.length} chapters. Press number keys 1 to 9 for quick selection. Press Enter to start practice with the selected chapter. Press B to go back to quiz mode selection.`,
          { interrupt: true }
        );
      }

      // 🔙 Back to quiz mode selection
      if (e.key.toLowerCase() === 'b') {
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

  // Focus management for keyboard navigation
  useEffect(() => {
    if (selectedIndex !== null && chapters.length > 0) {
      const selectedCard = document.querySelector(`[role="option"]:nth-child(${selectedIndex + 1})`);
      if (selectedCard instanceof HTMLElement) {
        selectedCard.focus({ preventScroll: true });
      }
    }
  }, [selectedIndex, chapters.length]);

  if (loading) {
    return (
      <main className="flex items-center justify-center min-h-[400px]" role="main">
        <div className="text-center space-y-4" role="status" aria-live="polite">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto" aria-hidden="true"></div>
          <p className="text-muted-foreground">Loading past paper chapters...</p>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="mx-auto max-w-3xl p-4 space-y-6" role="main">
        <div className="text-center space-y-4" role="alert">
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <h2 className="text-lg font-semibold text-red-800 mb-2">Error Loading Past Papers</h2>
            <p className="text-red-600">{error}</p>
            <Button 
              onClick={onBack} 
              variant="outline" 
              className="mt-3"
              aria-label="Go back to quiz mode selection"
            >
              ← Back to Quiz Modes
            </Button>
          </div>
        </div>
      </main>
    );
  }

  if (chapters.length === 0) {
    return (
      <main className="mx-auto max-w-3xl p-4 space-y-6" role="main">
        <div className="text-center space-y-4">
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg" role="alert">
            <h2 className="text-lg font-semibold text-yellow-800 mb-2">No Past Paper Chapters Available</h2>
            <p className="text-yellow-600">No past paper questions are currently available. Please check back later.</p>
            <Button 
              onClick={onBack} 
              variant="outline" 
              className="mt-3"
              aria-label="Go back to quiz mode selection"
            >
              ← Back to Quiz Modes
            </Button>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-4xl p-4 space-y-6 pb-24" role="main" aria-labelledby="page-title">
      {/* Skip to main content link for screen readers */}
      <a href="#chapter-selection" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-white p-2 rounded">
        Skip to chapter selection
      </a>
      
      {/* Page Header */}
      <header className="text-center space-y-2" role="banner">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Calendar className="h-6 w-6 text-primary" aria-hidden="true" />
          <h1 id="page-title" className="text-2xl font-bold">Past Paper Quiz</h1>
        </div>
        <p className="text-muted-foreground" aria-describedby="page-description">
          Practice with real examination questions from previous years
        </p>
        <div className="sr-only" id="page-description">
          Select a chapter to practice with past examination questions. Questions include year announcements and use advanced evaluation.
        </div>
      </header>

      {/* Navigation Help */}
      <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg" role="complementary" aria-labelledby="nav-help">
        <h2 id="nav-help" className="text-sm font-semibold mb-1">Keyboard Navigation</h2>
        <p className="text-xs text-muted-foreground">
          Up/Down arrows or 1-{chapters.length} to select chapters, Enter to start, H for detailed help, B to go back
        </p>
        {selectedIndex !== null && (
          <p className="text-xs text-blue-600 mt-1" role="status" aria-live="polite">
            Currently selected: Chapter {selectedIndex + 1} of {chapters.length} - {chapters[selectedIndex]}
          </p>
        )}
      </div>

      {/* Chapter Selection */}
      <section id="chapter-selection" role="region" aria-labelledby="chapters-heading">
        <h2 id="chapters-heading" className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BookOpen className="h-5 w-5" aria-hidden="true" />
          Select Chapter ({chapters.length} available)
        </h2>
        
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3" role="listbox" aria-label="Chapter selection list">
          {chapters.map((chapter, index) => (
            <Card
              key={chapter}
              className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                selectedIndex === index 
                  ? 'ring-2 ring-blue-500 bg-blue-50 border-blue-300' 
                  : 'hover:ring-1 hover:ring-blue-300'
              }`}
              role="option"
              aria-selected={selectedIndex === index}
              aria-label={`Chapter ${index + 1} of ${chapters.length}: ${chapter}. Real exam questions from previous years.`}
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
                  startPastPaperQuiz(index);
                }
              }}
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-blue-600 flex items-center gap-1">
                    <Calendar className="h-4 w-4" aria-hidden="true" />
                    Chapter {index + 1}
                  </span>
                  {selectedIndex === index && (
                    <span 
                      className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full" 
                      aria-label="Currently selected chapter"
                    >
                      Selected
                    </span>
                  )}
                </div>
                <h3 
                  className="font-semibold text-sm leading-tight"
                  id={`chapter-title-${index}`}
                >
                  {chapter}
                </h3>
                <p className="text-xs text-muted-foreground" aria-describedby={`chapter-title-${index}`}>
                  Contains real exam questions from previous years for practice
                </p>
              </div>
            </Card>
          ))}
        </div>
      </section>

      {/* Action Buttons */}
      <footer className="flex justify-between pt-4" role="contentinfo">
        <Button 
          variant="outline" 
          onClick={onBack}
          aria-label="Go back to quiz mode selection"
          className="flex items-center gap-2"
        >
          ← Back to Quiz Modes
        </Button>
        
        {selectedIndex !== null && (
          <Button 
            onClick={() => {
              cancel();
              speak(`Starting Past Paper Quiz for ${chapters[selectedIndex]}. Loading examination questions now.`, {
                interrupt: true,
                onEnd: () => startPastPaperQuiz(selectedIndex)
              });
            }}
            className="flex items-center gap-2"
            aria-label={`Start past paper quiz for chapter ${selectedIndex + 1}: ${chapters[selectedIndex]}`}
            onFocus={(e) => {
              // Use requestAnimationFrame to prevent render-time state updates
              requestAnimationFrame(() => {
                speak(`Start practice button focused. Press Enter to begin Past Paper Quiz for ${chapters[selectedIndex]}.`);
              });
            }}
          >
            <Play className="h-4 w-4" aria-hidden="true" />
            Start Practice
          </Button>
        )}
      </footer>
    </main>
  );
};