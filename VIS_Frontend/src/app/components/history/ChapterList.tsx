import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, BookOpen, Clock } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';
import { API_BASE_URL } from '../../services/api';

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

const cleanChapterNameForSpeech = (name: string) => {
  return name.replace(/^\s*\d+\s*[.):-]?\s*/, '').trim();
};

const speakSlow = (text: string, onEnd?: () => void) => {
  safeCancel();

  if (
    typeof window !== 'undefined' &&
    'speechSynthesis' in window &&
    typeof window.SpeechSynthesisUtterance === 'function'
  ) {
    try {
      const utterance = new window.SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      if (onEnd) {
        utterance.onend = onEnd;
      }
      window.speechSynthesis.speak(utterance);
      return;
    } catch {
      // Fall back to shared helper if browser API fails.
    }
  }

  safeSpeak(text, onEnd);
};

const SPOKEN_NUMBERS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
};

export const ChapterList = ({ grade, onSelectChapter, onBack }: ChapterListProps) => {
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasAnnounced, setHasAnnounced] = useState(false);
  const recognitionRef = useRef<any>(null);
  const isListeningRef = useRef(false);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const speakChapterList = useCallback(() => {
    if (chapters.length === 0) {
      speakSlow('No chapters available for this grade.');
      return;
    }

    let list = `Grade ${grade}. ${chapters.length} chapters available. Press number or say number you want. `;
    chapters.forEach((chapter, index) => {
      list += `${index + 1}. ${cleanChapterNameForSpeech(chapter.chapter_name)}. `;
    });
    speakSlow(list);
  }, [chapters, grade]);

  const selectChapterByNumber = useCallback((chapterNumber: number) => {
    if (chapterNumber < 1 || chapterNumber > chapters.length) {
      speakSlow(`Invalid chapter number. Please say a number from 1 to ${chapters.length}.`);
      return;
    }

    const selectedChapter = chapters[chapterNumber - 1];
    speakSlow(`${cleanChapterNameForSpeech(selectedChapter.chapter_name)} selected. Loading topics.`, () => {
      setTimeout(() => onSelectChapter(selectedChapter.id, selectedChapter.chapter_name), 500);
    });
  }, [chapters, onSelectChapter]);

  const getSpokenNumber = useCallback((transcript: string): number | null => {
    const normalized = transcript.toLowerCase();
    const digitMatch = normalized.match(/\b(10|[1-9])\b/);
    if (digitMatch) {
      return parseInt(digitMatch[1], 10);
    }

    for (const [word, number] of Object.entries(SPOKEN_NUMBERS)) {
      if (normalized.includes(word)) {
        return number;
      }
    }

    return null;
  }, []);

  const handleVoiceCommand = useCallback((transcript: string) => {
    const normalized = transcript.toLowerCase().trim();
    if (!normalized) {
      return;
    }

    if (normalized.includes('stop') || normalized.includes('pause') || normalized.includes('silent')) {
      safeCancel();
      return;
    }

    // Voice commands take priority over any ongoing speech.
    safeCancel();

    if (normalized.includes('back') || normalized.includes('go back') || normalized.includes('escape')) {
      speakSlow('Going back.', () => {
        setTimeout(() => onBack(), 250);
      });
      return;
    }

    if (normalized.includes('help')) {
      speakSlow(`Say or press a number from 1 to ${chapters.length} to select a chapter. Say list to hear chapters again. Say back to go back.`);
      return;
    }

    if (normalized.includes('list') || normalized.includes('repeat') || normalized.includes('explain') || normalized.includes('again')) {
      speakChapterList();
      return;
    }

    const spokenNumber = getSpokenNumber(normalized);
    if (spokenNumber !== null) {
      selectChapterByNumber(spokenNumber);
      return;
    }

    speakSlow('Command not recognized. Say a chapter number, explain, stop, help, or back.');
  }, [chapters.length, getSpokenNumber, onBack, selectChapterByNumber, speakChapterList]);

  const startListening = useCallback(() => {
    if (!recognitionRef.current || isListeningRef.current) {
      return;
    }

    try {
      recognitionRef.current.start();
    } catch {
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }
      restartTimeoutRef.current = setTimeout(() => {
        startListening();
      }, 800);
    }
  }, []);

  // Fetch chapters from backend
  useEffect(() => {
    const fetchChapters = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/api/chapters/${grade}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch chapters: ${response.statusText}`);
        }
        
        const data = await response.json();
        setChapters(data.chapters);
        setError(null);
      } catch (err) {
        console.error('Error fetching chapters:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chapters');
        speakSlow(`Error loading chapters. Please try again.`);
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
        speakSlow('No chapters available for this grade.');
        return;
      }

      setTimeout(() => {
        let announcement = `Grade ${grade}. ${chapters.length} chapters available. Press number or say number you want. `;
        chapters.forEach((chapter, index) => {
          announcement += `${index + 1}. ${cleanChapterNameForSpeech(chapter.chapter_name)}. `;
        });
        speakSlow(announcement);
      }, 500);
    }

    return () => {
      safeCancel();
    };
  }, [loading, chapters, hasAnnounced, grade]);

  // Always-on voice commands on chapter screen
  useEffect(() => {
    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    if (!SpeechRecognition) {
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      isListeningRef.current = true;
    };

    recognition.onresult = (event: any) => {
      const latestIndex = event.results.length - 1;
      const transcript = event.results[latestIndex]?.[0]?.transcript || '';
      handleVoiceCommand(transcript);
    };

    recognition.onerror = () => {
      isListeningRef.current = false;
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }
      restartTimeoutRef.current = setTimeout(() => {
        startListening();
      }, 700);
    };

    recognition.onend = () => {
      isListeningRef.current = false;
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }
      restartTimeoutRef.current = setTimeout(() => {
        startListening();
      }, 700);
    };

    recognitionRef.current = recognition;
    startListening();

    return () => {
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }

      if (recognitionRef.current) {
        try {
          recognitionRef.current.onstart = null;
          recognitionRef.current.onresult = null;
          recognitionRef.current.onerror = null;
          recognitionRef.current.onend = null;
          recognitionRef.current.stop();
        } catch {
          // ignore cleanup errors
        }
      }

      isListeningRef.current = false;
      recognitionRef.current = null;
    };
  }, [handleVoiceCommand, startListening]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const num = parseInt(e.key);
      if (num >= 1 && num <= chapters.length) {
        e.preventDefault();
        safeCancel();
        selectChapterByNumber(num);
      }

      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        let help = `Press a number 1 to ${chapters.length} to select a chapter. `;
        safeCancel();
        speakSlow(help);
      }

      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        safeCancel();
        speakChapterList();
      }

      if (e.key === 'Escape') {
        e.preventDefault();
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [chapters.length, onBack, selectChapterByNumber, speakChapterList]);

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
