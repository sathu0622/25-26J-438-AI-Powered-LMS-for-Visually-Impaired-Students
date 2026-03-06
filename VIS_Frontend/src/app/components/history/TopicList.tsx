import { useEffect, useState } from 'react';
import { ArrowLeft, BookOpen, Play } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech'; import { API_BASE_URL } from '../../services/api';
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
        safeSpeak(`Error loading topics. Please try again.`);
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

      if (topics.length === 0) {
        safeSpeak('No topics available for this chapter.');
        return;
      }

      setTimeout(() => {
        let announcement = `Chapter: ${chapterName}. ${topics.length} topics available. `;
        topics.forEach((topic, index) => {
          announcement += `Press ${index + 1} for ${topic.topic_name}. `;
        });
        announcement += 'Press H for help, or Escape to go back.';
        safeSpeak(announcement);
      }, 500);
    }

    return () => {
      safeCancel();
    };
  }, [loading, topics, hasAnnounced, chapterName]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const num = parseInt(e.key);
      if (num >= 1 && num <= topics.length) {
        e.preventDefault();
        const selectedTopic = topics[num - 1];
        const content = selectedTopic.simplified_text || selectedTopic.original_text || selectedTopic.narrative_text || '';
        
        safeSpeak(`${selectedTopic.topic_name} selected. Loading content.`, () => {
          setTimeout(() => onSelectTopic(selectedTopic.id, selectedTopic.topic_name, content), 500);
        });
      }

      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        let help = `Press a number 1 to ${topics.length} to select a topic. `;
        safeCancel();
        safeSpeak(help);
      }

      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        let list = `${topics.length} topics available. `;
        topics.forEach((topic, index) => {
          list += `Topic ${index + 1}: ${topic.topic_name}. `;
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
  }, [topics, onSelectTopic, onBack]);

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
