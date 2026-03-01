import { useState, useEffect } from 'react';
import { Volume2, Mic, Send, SkipForward } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { MockVoiceRecorder } from '../MockVoiceRecorder';
import { useTTS } from '../../contexts/TTSContext';

interface Question {
  id: number;
  text: string;
  topic: string;
}

interface QuizQuestionProps {
  question: {
    question: string;
    correct_answer: string;
    key_phrase: string;
  };
  questionNumber: number;
  totalQuestions: number;
  onSubmit: (answer: string) => void;
  onSkip: () => void;
}

export const QuizQuestion = ({
  question,
  questionNumber,
  totalQuestions,
  onSubmit,
  onSkip,
}: QuizQuestionProps) => {
  const [answer, setAnswer] = useState('');
  const [showVoiceModal, setShowVoiceModal] = useState(false);

  const { speak, cancel } = useTTS();

  useEffect(() => {
    cancel();
    setHasReadQuestion(false);
    const timer = setTimeout(() => {
      speak(
        `Question ${questionNumber} of ${totalQuestions}. ${question.text}. Press Space or Enter to record your answer, Press Q to repeat question, Press R to record, Press S to skip question.`,
        { interrupt: true }
      );
      setHasReadQuestion(true);
    }, 500);
    return () => {
      clearTimeout(timer);
      cancel();
    };
  }, [question.id, questionNumber, totalQuestions, question.text, speak, cancel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Space or Enter to toggle recording/submit (when not in textarea)
      if ((e.key === ' ' || e.key === 'Enter') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        if (answer.trim() && !showVoiceModal) {
          handleSubmit();
        } else {
          handleVoiceToggle();
        }
      }

      // R key to start recording
      if ((e.key === 'r' || e.key === 'R') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        handleVoiceToggle();
      }

      // S key to skip
      if ((e.key === 's' || e.key === 'S') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        handleSkip();
      }

      // Q key to read question
      if ((e.key === 'q' || e.key === 'Q') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        handleReadQuestion();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [showVoiceModal, answer]);

  const handleReadQuestion = () => {
    cancel();
    speak(`Question ${questionNumber}. ${question.text}`, { interrupt: true });
  };

  const handleVoiceToggle = () => {
    setInputMode('voice');
    setShowVoiceModal(true);
  };

  const handleVoiceSubmit = (text: string) => {
    setAnswer(text);
    setShowVoiceModal(false);
  };

  const handleSubmit = () => {
    if (answer.trim()) {
      onSubmit(answer);
      setAnswer('');
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      <Card className="p-6">
        <h2 className="text-lg">{question.question}</h2>
        <Button
          variant="outline"
          size="icon"
          onClick={() => speak(question.question)}
        >
          <Volume2 className="h-5 w-5" />
        </Button>
      </Card>

      <Card className="p-6">
        <Textarea
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          placeholder="Type your answer..."
          className="min-h-[120px]"
        />
      </Card>

      <div className="grid gap-3 sm:grid-cols-2">
        <Button
          variant="outline"
          onClick={() => setShowVoiceModal(true)}
        >
          <Mic className="mr-2 h-5 w-5" />
          Record Answer
        </Button>

        <Button
          onClick={handleSubmit}
          disabled={!answer.trim()}
        >
          <Send className="mr-2 h-5 w-5" />
          Submit
        </Button>
      </div>

      <Button variant="ghost" onClick={onSkip}>
        <SkipForward className="mr-2 h-4 w-4" />
        Skip Question
      </Button>

      <MockVoiceRecorder
        isOpen={showVoiceModal}
        onClose={() => setShowVoiceModal(false)}
        onSubmit={(text) => {
          setAnswer(text);
          setShowVoiceModal(false);
        }}
        title="Record Your Answer"
        context={question.question}
      />
    </div>
  );
};"mr-2 h-5 w-5" />
          Submit
        </Button>
      </div>

      <Button variant="ghost" onClick={onSkip}>
        <SkipForward className="mr-2 h-4 w-4" />
        Skip Question
      </Button>

      <MockVoiceRecorder
        isOpen={showVoiceModal}
        onClose={() => setShowVoiceModal(false)}
        onSubmit={(text) => {
          setAnswer(text);
          setShowVoiceModal(false);
        }}
        title="Record Your Answer"
        context={question.question}
      />
    </div>
  );
};