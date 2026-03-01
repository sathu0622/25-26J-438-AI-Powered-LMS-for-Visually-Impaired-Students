import { useState, useEffect } from 'react';
import { Play, BookOpen } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';
import { quizService } from '../../services/quizService';

interface QuizStartProps {
  onStart: (topic: string) => void;
}

export const QuizStart = ({ onStart }: QuizStartProps) => {
  const [topics, setTopics] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  // Load chapters from backend
  useEffect(() => {
    const loadChapters = async () => {
      try {
        const chapters = await quizService.getChapters();
        setTopics(chapters);

        // 🔊 Speak instructions after chapters load
        setTimeout(() => {
          safeSpeak(
            `Chapters loaded. Use number keys or arrow keys to select a chapter. Press Enter to start.`
          );
        }, 600);
      } catch (err) {
        console.error('Failed to load chapters', err);
      } finally {
        setLoading(false);
      }
    };

    loadChapters();
  }, []);

  const handleTopicSelect = (index: number) => {
    setSelectedIndex(index);
    safeCancel();
    safeSpeak(`${topics[index]} selected.`);
  };

  const startQuiz = (index: number) => {
    safeCancel();
    onStart(topics[index]);
  };

  // ✅ KEYBOARD NAVIGATION
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (loading || topics.length === 0) return;

      // 🔢 Number selection (1–9)
      if (/^[1-9]$/.test(e.key)) {
        const index = parseInt(e.key) - 1;

        if (topics[index]) {
          handleTopicSelect(index);

          // 🚀 Auto-start after short delay
          setTimeout(() => startQuiz(index), 700);
        }
      }

      // ⬆ Arrow Up
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex =
            prev === null
              ? 0
              : (prev - 1 + topics.length) % topics.length;

          safeSpeak(topics[newIndex]);
          return newIndex;
        });
      }

      // ⬇ Arrow Down
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => {
          const newIndex =
            prev === null
              ? 0
              : (prev + 1) % topics.length;

          safeSpeak(topics[newIndex]);
          return newIndex;
        });
      }

      // ⏎ Enter to start
      if (e.key === 'Enter' && selectedIndex !== null) {
        startQuiz(selectedIndex);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [topics, selectedIndex, loading]);

  if (loading) {
    return (
      <div className="p-6 text-center">
        <p>Loading Chapters...</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-green-500 p-6">
            <BookOpen className="h-12 w-12 text-white" />
          </div>
        </div>
        <h1 className="text-2xl">Voice-Enabled History Quiz</h1>
      </div>

      <div className="space-y-3">
        <h2 className="text-center">
          Select a Chapter (Use ↑ ↓ or 1-{topics.length})
        </h2>

        <div className="grid gap-2">
          {topics.map((topic, index) => (
            <Card
              key={index}
              className={`overflow-hidden transition-all ${
                selectedIndex === index
                  ? 'border-primary bg-primary/5 border-2'
                  : 'hover:shadow-md'
              }`}
            >
              <button
                onClick={() => handleTopicSelect(index)}
                className="w-full p-4 text-left"
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-sm">
                    {index + 1}
                  </div>
                  <span>{topic}</span>
                </div>
              </button>
            </Card>
          ))}
        </div>
      </div>

      <Button
        onClick={() =>
          selectedIndex !== null && startQuiz(selectedIndex)
        }
        disabled={selectedIndex === null}
        size="lg"
        className="w-full min-h-[64px]"
      >
        <Play className="mr-2 h-6 w-6" />
        {selectedIndex !== null
          ? 'Press Enter or Click to Start'
          : 'Select a Chapter First'}
      </Button>
    </div>
  );
};