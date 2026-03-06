import { useEffect, useState } from 'react';
import { Volume2, Loader2, PenLine, History } from 'lucide-react';
import { freeTextService, FreeTextSessionListItem } from '../../services/freeTextService';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { useTTS } from '../../contexts/TTSContext';

interface FreeTextStartProps {
  onStart: (chapter: string, sessionId?: string) => void;
  onBack?: () => void;
  username: string;
}

export const FreeTextStart = ({ onStart, onBack, username }: FreeTextStartProps) => {
  const [chapters, setChapters] = useState<string[]>([]);
  const [savedSessions, setSavedSessions] = useState<FreeTextSessionListItem[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [showSaved, setShowSaved] = useState(false);
  const { speak, cancel } = useTTS();

  // Load chapters and saved sessions
  useEffect(() => {
    const loadData = async () => {
      try {
        const [chaptersRes, sessionsRes] = await Promise.all([
          freeTextService.getChapters(),
          freeTextService.getUserSessions(username),
        ]);
        
        setChapters(chaptersRes);
        setSavedSessions(sessionsRes.sessions);

        setTimeout(() => {
          const savedCount = sessionsRes.sessions.length;
          speak(
            `Free-Text Quiz chapters loaded. ${chaptersRes.length} chapters available. ${savedCount > 0 ? `You have ${savedCount} saved sessions for retake.` : ''} Use Up and Down arrow keys to select a chapter. Press Enter to start. Press S to view saved sessions. Press H for help. Press B to go back.`
          );
        }, 600);
      } catch (err) {
        console.error('Failed to load data', err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [username, speak]);

  const handleChapterSelect = (index: number) => {
    setSelectedIndex(index);
    cancel();
    speak(`${chapters[index]} selected.`);
  };

  const startQuiz = (index: number) => {
    cancel();
    onStart(chapters[index]);
  };

  const handleRetake = (session: FreeTextSessionListItem) => {
    cancel();
    speak(`Retaking ${session.chapter_name} quiz.`);
    onStart(session.chapter_name, session.session_id);
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (loading) return;

      const items = showSaved ? savedSessions : chapters;
      if (items.length === 0) return;

      // Number selection (1–9)
      if (/^[1-9]$/.test(e.key) && !showSaved) {
        const index = parseInt(e.key) - 1;
        if (chapters[index]) {
          handleChapterSelect(index);
          setTimeout(() => startQuiz(index), 700);
        }
      }

      // Arrow navigation
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex = prev === null ? 0 : (prev - 1 + items.length) % items.length;
          if (showSaved) {
            speak(savedSessions[newIndex].chapter_name);
          } else {
            speak(chapters[newIndex]);
          }
          return newIndex;
        });
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex = prev === null ? 0 : (prev + 1) % items.length;
          if (showSaved) {
            speak(savedSessions[newIndex].chapter_name);
          } else {
            speak(chapters[newIndex]);
          }
          return newIndex;
        });
      }

      // Enter to start/retake
      if (e.key === 'Enter' && selectedIndex !== null) {
        if (showSaved) {
          handleRetake(savedSessions[selectedIndex]);
        } else {
          startQuiz(selectedIndex);
        }
      }

      // S to toggle saved sessions
      if (e.key.toLowerCase() === 's') {
        e.preventDefault();
        setShowSaved((prev) => {
          const next = !prev;
          setSelectedIndex(null);
          if (next) {
            speak(`Showing ${savedSessions.length} saved sessions. Press S again to go back to chapters.`);
          } else {
            speak(`Showing ${chapters.length} chapters.`);
          }
          return next;
        });
      }

      // Help
      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        cancel();
        speak(
          `Free-Text Quiz Help. In this mode, you answer questions by typing or speaking. Your answers are evaluated for meaning, not exact match. Use Up and Down arrows to navigate. Press Enter to start. Press S to toggle between new quiz and saved sessions. Press B to go back.`,
          { interrupt: true }
        );
      }

      // Back
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
  }, [chapters, savedSessions, loading, selectedIndex, showSaved, speak, cancel, onStart, onBack]);

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center gap-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading free-text quiz data…
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-green-500 p-6">
            <PenLine className="h-12 w-12 text-white" />
          </div>
        </div>
        <h1 className="text-2xl font-bold">Generative Free-Text Quiz</h1>
        <p className="text-sm text-muted-foreground">
          AI-generated questions with free-form answers evaluated by semantic similarity
        </p>
      </div>

      {/* Navigation Help */}
      <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg" role="complementary" aria-labelledby="nav-help">
        <h2 id="nav-help" className="text-sm font-semibold mb-1">Keyboard Navigation</h2>
        <p className="text-xs text-muted-foreground">
          Up/Down arrows to navigate, Enter to start, S for saved sessions, H for help, B to go back
        </p>
        {selectedIndex !== null && (
          <p className="text-xs text-green-600 mt-1" role="status" aria-live="polite">
            Selected: {showSaved ? `Session ${selectedIndex + 1}` : `Chapter ${selectedIndex + 1}`} - {showSaved ? savedSessions[selectedIndex]?.chapter_name : chapters[selectedIndex]}
          </p>
        )}
      </div>

      {/* Toggle buttons */}
      <div className="flex gap-2">
        <Button
          variant={!showSaved ? 'default' : 'outline'}
          className={`flex-1 ${!showSaved ? 'bg-green-600 hover:bg-green-700' : ''}`}
          onClick={() => {
            setShowSaved(false);
            setSelectedIndex(null);
          }}
        >
          <PenLine className="mr-2 h-4 w-4" /> New Quiz ({chapters.length})
        </Button>
        <Button
          variant={showSaved ? 'default' : 'outline'}
          className={`flex-1 ${showSaved ? 'bg-green-600 hover:bg-green-700' : ''}`}
          onClick={() => {
            setShowSaved(true);
            setSelectedIndex(null);
          }}
          disabled={savedSessions.length === 0}
        >
          <History className="mr-2 h-4 w-4" /> Saved ({savedSessions.length})
        </Button>
      </div>

      {/* Chapter or Saved Session List */}
      {!showSaved ? (
        <div className="space-y-3">
          <h2 className="text-center">Select a Chapter</h2>
          <div className="grid gap-2 max-h-[400px] overflow-y-auto">
            {chapters.map((chapter, index) => (
              <Card
                key={index}
                className={`overflow-hidden transition-all cursor-pointer ${
                  selectedIndex === index
                    ? 'border-green-500 bg-green-50 border-2'
                    : 'hover:shadow-md hover:border-green-300'
                }`}
                role="option"
                aria-selected={selectedIndex === index}
                tabIndex={selectedIndex === index ? 0 : -1}
                onClick={() => handleChapterSelect(index)}
                onFocus={() => {
                  if (selectedIndex !== index) handleChapterSelect(index);
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
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-100 text-green-700 text-sm font-semibold">
                      {index + 1}
                    </div>
                    <div>
                      <span className="font-medium">{chapter}</span>
                      <p className="text-xs text-muted-foreground">Free-text answers</p>
                    </div>
                    {selectedIndex === index && (
                      <span className="ml-auto text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
                        Selected
                      </span>
                    )}
                  </div>
                </button>
              </Card>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <h2 className="text-center">Saved Sessions (Retake)</h2>
          <div className="grid gap-2 max-h-[400px] overflow-y-auto">
            {savedSessions.map((session, index) => (
              <Card
                key={session.session_id}
                className={`overflow-hidden transition-all cursor-pointer ${
                  selectedIndex === index
                    ? 'border-green-500 bg-green-50 border-2'
                    : 'hover:shadow-md hover:border-green-300'
                }`}
                role="option"
                aria-selected={selectedIndex === index}
                onClick={() => {
                  setSelectedIndex(index);
                  speak(`${session.chapter_name}, ${session.questions_count} questions.`);
                }}
              >
                <button className="w-full p-4 text-left">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-medium">{session.chapter_name}</span>
                      <p className="text-xs text-muted-foreground">
                        {session.questions_count} questions • {session.attempts_count} attempts
                      </p>
                      {session.latest_attempt?.summary && (
                        <p className="text-xs text-green-600">
                          Last: {session.latest_attempt.summary.correct_count}/{session.latest_attempt.summary.total_questions} correct ({session.latest_attempt.summary.average_score}%)
                        </p>
                      )}
                    </div>
                    {selectedIndex === index && (
                      <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
                        Selected
                      </span>
                    )}
                  </div>
                </button>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Start Button */}
      <Button
        onClick={() => {
          if (selectedIndex !== null) {
            if (showSaved) {
              handleRetake(savedSessions[selectedIndex]);
            } else {
              startQuiz(selectedIndex);
            }
          }
        }}
        disabled={selectedIndex === null}
        size="lg"
        className="w-full min-h-[64px] bg-green-600 hover:bg-green-700"
      >
        <PenLine className="mr-2 h-6 w-6" />
        {selectedIndex !== null
          ? showSaved ? 'Retake Selected Session' : 'Start Free-Text Quiz'
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
    </div>
  );
};
