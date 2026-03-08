import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, BookOpen, Play } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';
import { API_BASE_URL } from '../../services/api';

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

const normalizeSpeechText = (text: string) => {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
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

interface Topic {
  id: number;
  topic_name: string;
  chapter: string;
  grade: number;
  original_text?: string;
  simplified_text?: string;
  narrative_text?: string;
  emotion?: string;
  sound_effects?: string;
}

interface TopicListProps {
  grade: number;
  chapterId: number;
  chapterName: string;
  onSelectTopic: (topicId: number, topicName: string, content: string) => void;
  onBack: () => void;
}

export const TopicList = ({ grade, chapterId, chapterName, onSelectTopic, onBack }: TopicListProps) => {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasAnnounced, setHasAnnounced] = useState(false);
  const recognitionRef = useRef<any>(null);
  const isListeningRef = useRef(false);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const speakTopicOverview = useCallback(() => {
    if (topics.length === 0) {
      speakSlow('No topics available for this chapter.');
      return;
    }

    let announcement = `Chapter: ${chapterName}. ${topics.length} topics available. Press number or say number you want. `;
    topics.forEach((topic, index) => {
      announcement += `${index + 1}. ${topic.topic_name}. `;
    });
    speakSlow(announcement);
  }, [chapterName, topics]);

  const getSpokenNumber = useCallback((transcript: string): number | null => {
    const normalized = transcript.toLowerCase();
    const digitMatch = normalized.match(/\b(\d+)\b/);
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

  const selectTopic = useCallback((topic: Topic) => {
    const content = topic.simplified_text || topic.original_text || topic.narrative_text || '';

    speakSlow(`ok,${topic.topic_name} selected. Loading lesson.`, () => {
      setTimeout(() => onSelectTopic(topic.id, topic.topic_name, content), 500);
    });
  }, [onSelectTopic]);

  const getTopicBySpokenName = useCallback((transcript: string): Topic | null => {
    const normalizedTranscript = normalizeSpeechText(transcript);
    if (!normalizedTranscript) {
      return null;
    }

    // Strong match: full topic name appears in transcript.
    const fullMatch = topics.find((topic) => normalizedTranscript.includes(normalizeSpeechText(topic.topic_name)));
    if (fullMatch) {
      return fullMatch;
    }

    // Fallback match: transcript appears within topic name.
    const partialMatch = topics.find((topic) => normalizeSpeechText(topic.topic_name).includes(normalizedTranscript));
    return partialMatch || null;
  }, [topics]);

  const selectTopicByNumber = useCallback((topicNumber: number) => {
    if (topicNumber < 1 || topicNumber > topics.length) {
      speakSlow(`Invalid topic number. Please say a number from 1 to ${topics.length}.`);
      return;
    }

    const selectedTopic = topics[topicNumber - 1];
    selectTopic(selectedTopic);
  }, [selectTopic, topics]);

  const handleVoiceCommand = useCallback((transcript: string) => {
    const normalized = transcript.toLowerCase().trim();
    if (!normalized) {
      return;
    }

    if (normalized.includes('hello')) {
      safeCancel();
      speakSlow('Yes, say dear.');
      return;
    }

    if (normalized.includes('stop speech')) {
      safeCancel();
      speakSlow("Okay, I'm silance now, say me what to do?");
      return;
    }

    if (normalized.includes('stop') || normalized.includes('pause') || normalized.includes('silent')) {
      safeCancel();
      return;
    }

    safeCancel();

    if (normalized.includes('back') || normalized.includes('go back') || normalized.includes('escape')) {
      speakSlow('Going back.', () => {
        setTimeout(() => onBack(), 250);
      });
      return;
    }

    if (normalized.includes('explain') || normalized.includes('again') || normalized.includes('repeat') || normalized.includes('list')) {
      speakTopicOverview();
      return;
    }

    if (normalized.includes('help')) {
      speakSlow(`Say a topic number from 1 to ${topics.length}. Say explain to hear the topic list again. Say stop to stop speech. Say back to go back.`);
      return;
    }

    const spokenNumber = getSpokenNumber(normalized);
    if (spokenNumber !== null) {
      selectTopicByNumber(spokenNumber);
      return;
    }

    const spokenTopic = getTopicBySpokenName(normalized);
    if (spokenTopic) {
      selectTopic(spokenTopic);
      return;
    }

    speakSlow('Command not recognized. Say a topic number or topic name, explain, stop, help, or back.');
  }, [getSpokenNumber, getTopicBySpokenName, onBack, selectTopic, selectTopicByNumber, speakTopicOverview, topics.length]);

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

  // Fetch topics from backend
  useEffect(() => {
    const fetchTopics = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `${API_BASE_URL}/api/chapters/${grade}/${chapterId}/topics`
        );
        
        if (!response.ok) {
          throw new Error(`Failed to fetch topics: ${response.statusText}`);
        }
        
        const data = await response.json();
        setTopics(data.topics);
        setError(null);
      } catch (err) {
        console.error('Error fetching topics:', err);
        setError(err instanceof Error ? err.message : 'Failed to load topics');
        speakSlow(`Error loading topics. Please try again.`);
      } finally {
        setLoading(false);
      }
    };

    fetchTopics();
  }, [grade, chapterId]);

  // Voice announcement
  useEffect(() => {
    if (!loading && !hasAnnounced) {
      safeCancel();
      setHasAnnounced(true);

      setTimeout(() => {
        speakTopicOverview();
      }, 500);
    }

    return () => {
      safeCancel();
    };
  }, [loading, hasAnnounced, speakTopicOverview]);

  // Always-on voice commands on topic screen
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
          // Ignore cleanup errors.
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
      if (num >= 1 && num <= topics.length) {
        e.preventDefault();
        safeCancel();
        selectTopicByNumber(num);
      }

      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        let help = `Press a number 1 to ${topics.length} to select a topic. `;
        safeCancel();
        speakSlow(help);
      }

      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        safeCancel();
        speakTopicOverview();
      }

      if (e.key === 'Escape') {
        e.preventDefault();
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [onBack, selectTopicByNumber, speakTopicOverview, topics.length]);

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button onClick={onBack} variant="ghost" size="icon" aria-label="Go back">
          <ArrowLeft className="h-6 w-6" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl">{chapterName}</h1>
          <p className="text-sm text-muted-foreground">
            Grade {grade} • Press 1-{topics.length} to select topic
          </p>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <Card className="p-6">
          <p className="text-center text-muted-foreground">Loading topics...</p>
        </Card>
      )}

      {/* Error State */}
      {error && (
        <Card className="border-red-500 bg-red-50 p-6">
          <p className="text-red-800">{error}</p>
        </Card>
      )}

      {/* Topics List */}
      {!loading && topics.length > 0 && (
        <div className="space-y-4">
          {topics.map((topic, index) => (
            <Card
              key={topic.id}
              className="overflow-hidden transition-all hover:shadow-lg"
            >
              <button
                onClick={() => {
                  const content = topic.simplified_text || topic.original_text || topic.narrative_text || '';
                  onSelectTopic(topic.id, topic.topic_name, content);
                }}
                className="w-full p-6 text-left"
                aria-label={`Open ${topic.topic_name}`}
              >
                <div className="flex gap-4">
                  <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-secondary">
                    <span className="text-xl font-bold text-secondary-foreground">
                      {index + 1}
                    </span>
                  </div>
                  <div className="flex-1 space-y-2">
                    <h3 className="text-lg">{topic.topic_name}</h3>
                    {topic.emotion && (
                      <p className="text-xs text-muted-foreground">
                        Tone: {topic.emotion}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center">
                    <div className="rounded-full bg-secondary/10 p-3">
                      <Play className="h-5 w-5 text-secondary" />
                    </div>
                  </div>
                </div>
              </button>
            </Card>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && topics.length === 0 && !error && (
        <Card className="p-6">
          <p className="text-center text-muted-foreground">
            No topics available for this chapter
          </p>
        </Card>
      )}
    </div>
  );
};
