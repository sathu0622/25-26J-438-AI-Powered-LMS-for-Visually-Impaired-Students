import { useState, useEffect } from 'react';
import { Play, BookOpen, ArrowLeft } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';
import { getAllTopics } from '../../data/quizData';

interface QuizStartProps {
  onStart: (topic: string) => void;
}

export const QuizStart = ({ onStart }: QuizStartProps) => {
  const topics = getAllTopics();
  const [hasAnnounced, setHasAnnounced] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);

  // Voice announcement on page load
  useEffect(() => {
    safeCancel();
    
    if (!hasAnnounced) {
      setHasAnnounced(true);
      setTimeout(() => {
        let announcement = 'Voice-Enabled Quiz. Select a topic to begin. ';
        topics.forEach((topic, index) => {
          announcement += `Press ${index + 1} for ${topic}. `;
        });
        announcement += 'Press H for help, or Escape to go back.';
        
        safeSpeak(announcement);
      }, 500);
    }
    
    return () => {
      safeCancel();
    };
  }, [hasAnnounced, topics]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Number keys to select topic
      const num = parseInt(e.key);
      if (num >= 1 && num <= topics.length) {
        e.preventDefault();
        const topic = topics[num - 1];
        setSelectedTopic(topic);
        safeCancel();
        safeSpeak(`${topic} selected. Press Enter to start quiz.`);
      }

      // Enter to start quiz with selected topic
      if (e.key === 'Enter' && selectedTopic) {
        e.preventDefault();
        safeCancel();
        onStart(selectedTopic);
      }

      // H key for help
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        let help = 'Available topics: ';
        topics.forEach((topic, index) => {
          help += `${index + 1}. ${topic}. `;
        });
        help += 'Press a number to select, then Enter to start.';
        safeCancel();
        safeSpeak(help);
      }

      // L key to list all topics
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        let list = `${topics.length} topics available. `;
        topics.forEach((topic, index) => {
          list += `${index + 1}. ${topic}. `;
        });
        safeCancel();
        safeSpeak(list);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedTopic, onStart, topics]);

  const handleTopicSelect = (topic: string) => {
    setSelectedTopic(topic);
    safeCancel();
    safeSpeak(`${topic} selected. Press Start Quiz or Enter to begin.`);
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-green-500 p-6">
            <BookOpen className="h-12 w-12 text-white" aria-hidden="true" />
          </div>
        </div>
        <h1 className="text-2xl">Voice-Enabled Quiz</h1>
        <p className="text-muted-foreground">
          1-{topics.length}: Select Topic • Enter: Start • H: Help • L: List All
        </p>
      </div>

      {/* Topics */}
      <div className="space-y-3">
        <h2 className="text-center">Select a Topic</h2>
        <div className="grid gap-2">
          {topics.map((topic, index) => (
            <Card 
              key={index} 
              className={`overflow-hidden transition-all ${
                selectedTopic === topic
                  ? 'border-primary bg-primary/5 border-2'
                  : 'hover:shadow-md'
              }`}
            >
              <button
                onClick={() => handleTopicSelect(topic)}
                className="w-full p-4 text-left"
                aria-label={`Select ${topic}`}
                aria-pressed={selectedTopic === topic}
              >
                <div className="flex items-center gap-3">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm ${
                    selectedTopic === topic
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-muted-foreground'
                  }`}>
                    {index + 1}
                  </div>
                  <span>{topic}</span>
                </div>
              </button>
            </Card>
          ))}
        </div>
      </div>

      {/* Start Button */}
      <Button 
        onClick={() => selectedTopic && onStart(selectedTopic)} 
        disabled={!selectedTopic}
        size="lg" 
        className="w-full min-h-[64px]"
      >
        <Play className="mr-2 h-6 w-6" aria-hidden="true" />
        {selectedTopic ? `Start ${selectedTopic} Quiz` : 'Select a Topic First'}
      </Button>
    </div>
  );
};