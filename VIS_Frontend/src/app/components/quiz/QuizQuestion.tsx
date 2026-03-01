import { useState, useEffect } from 'react';
import { Volume2, Mic, Send, SkipForward } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { MockVoiceRecorder } from '../MockVoiceRecorder';
import { useSpeechSynthesis } from '../../hooks/useSpeechSynthesis';

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

  const { speak, stop } = useSpeechSynthesis();

  useEffect(() => {
    stop();
    const timer = setTimeout(() => {
      speak(
        `Question ${questionNumber}. ${question.question}. Press space to answer or S to skip.`
      );
    }, 500);

    return () => {
      clearTimeout(timer);
      stop();
    };
  }, [question]);

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
};