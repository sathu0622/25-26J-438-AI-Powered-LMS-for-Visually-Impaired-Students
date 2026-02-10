import { useState, useEffect } from 'react';
import { Send, Loader2, ArrowLeft, AlertCircle, X } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { VoiceButton } from '../VoiceButton';
import { AudioPlayer } from '../AudioPlayer';
import { useSpeechRecognition } from '../../hooks/useSpeechRecognition';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface QAItem {
  question: string;
  answer: string;
}

interface DocumentQAProps {
  mode: 'voice' | 'text';
  onBack: () => void;
}

export const DocumentQA = ({ mode, onBack }: DocumentQAProps) => {
  const [question, setQuestion] = useState('');
  const [qaHistory, setQaHistory] = useState<QAItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAnswer, setCurrentAnswer] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTimer, setRecordingTimer] = useState<NodeJS.Timeout | null>(null);
  const [hasAutoStarted, setHasAutoStarted] = useState(false);

  const {
    isListening,
    transcript,
    interimTranscript,
    startListening,
    stopListening,
    resetTranscript,
    error: speechError,
    permissionDenied,
    clearError,
  } = useSpeechRecognition();

  // Auto-start voice recording when entering voice mode
  useEffect(() => {
    // STOP all previous speech immediately
    safeCancel();
    
    if (mode === 'voice' && !hasAutoStarted) {
      setHasAutoStarted(true);
      // Announce and auto-start
      safeSpeak('Ask a Question page. Press Space or Enter to record or submit your question. Press R to re-record. Press A to replay answer after receiving it. Press Escape to go back. Recording will start automatically in 2 seconds.');
      
      // Auto-start after announcement
      setTimeout(() => {
        handleVoiceToggle();
      }, 2000);
    }
    
    // Cleanup: stop speech when leaving page
    return () => {
      safeCancel();
    };
  }, [mode]);

  // Keyboard shortcuts for voice recording
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Space or Enter to toggle recording (when not in textarea)
      if ((e.key === ' ' || e.key === 'Enter') && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        if (!isLoading) {
          if (question.trim() && !isRecording) {
            // Submit question if we have text
            handleAsk();
          } else {
            // Toggle recording
            handleVoiceToggle();
          }
        }
      }

      // R key to start recording
      if (e.key === 'r' || e.key === 'R') {
        if (!isLoading && !isRecording && e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
          e.preventDefault();
          handleVoiceToggle();
        }
      }

      // A key to replay answer audio
      if ((e.key === 'a' || e.key === 'A') && currentAnswer) {
        if (e.target && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
          e.preventDefault();
          // Trigger audio replay by speaking the answer again
          safeCancel(); // Stop any current speech
          safeSpeak(currentAnswer);
        }
      }

      // Escape to go back
      if (e.key === 'Escape') {
        e.preventDefault();
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isRecording, isLoading, question, currentAnswer, onBack]);

  useEffect(() => {
    if (transcript && !isListening) {
      setQuestion(transcript);
    }
  }, [transcript, isListening]);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setIsLoading(true);
    setCurrentAnswer(null);

    // Simulate AI response
    setTimeout(() => {
      const mockAnswer = `Based on the document, ${question.toLowerCase().includes('what') ? 'the answer involves' : 'this relates to'} the educational methodologies discussed. The document emphasizes accessible learning materials and adaptive technologies that support students with visual impairments through multi-sensory experiences and personalized teaching approaches.`;

      setQaHistory((prev) => [...prev, { question, answer: mockAnswer }]);
      setCurrentAnswer(mockAnswer);
      setQuestion('')
      resetTranscript();
      setIsLoading(false);
      
      // Announce that answer is ready and how to replay
      setTimeout(() => {
        safeSpeak('Answer received. Press A to replay the answer anytime, or press Space to ask another question.');
      }, 8000); // Wait for answer to finish reading
    }, 2000);
  };

  const handleVoiceToggle = () => {
    // Use mock voice recording for demo
    if (isRecording) {
      // Clear timer if manually stopped
      if (recordingTimer) {
        clearTimeout(recordingTimer);
        setRecordingTimer(null);
      }
      setIsRecording(false);
      // Simulate question after recording
      const mockQuestions = [
        'What are the main benefits of accessible education?',
        'How do adaptive technologies help visually impaired students?',
        'What is multi-sensory learning and why is it important?',
        'Can you explain the role of inclusive teaching methods?'
      ];
      const randomQuestion = mockQuestions[Math.floor(Math.random() * mockQuestions.length)];
      setQuestion(randomQuestion);
    } else {
      setIsRecording(true);
      setQuestion('');
      // Auto-stop after 3 seconds for demo
      const timer = setTimeout(() => {
        const mockQuestions = [
          'What are the main benefits of accessible education?',
          'How do adaptive technologies help visually impaired students?',
          'What is multi-sensory learning and why is it important?',
          'Can you explain the role of inclusive teaching methods?'
        ];
        const randomQuestion = mockQuestions[Math.floor(Math.random() * mockQuestions.length)];
        setQuestion(randomQuestion);
        setIsRecording(false);
        setRecordingTimer(null);
      }, 3000);
      setRecordingTimer(timer);
    }
  };

  const displayQuestion = isRecording
    ? 'Listening...'
    : question;

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button
          onClick={onBack}
          variant="ghost"
          size="icon"
          aria-label="Go back - Press Escape"
        >
          <ArrowLeft className="h-6 w-6" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl">Ask a Question</h1>
          <p className="text-sm text-muted-foreground">
            {mode === 'voice' 
              ? currentAnswer 
                ? 'Space to ask new • A to replay answer • Esc to go back'
                : 'Space/Enter to record • R to record again • Esc to go back'
              : 'Type your question'}
          </p>
        </div>
      </div>

      {/* Error Alert */}
      {speechError && mode === 'voice' && (
        <Card className="border-destructive bg-destructive/10 p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 flex-shrink-0 text-destructive" />
            <div className="flex-1 space-y-2">
              <p className="text-sm leading-relaxed">{speechError}</p>
              {permissionDenied && (
                <div className="space-y-2 text-sm">
                  <p className="font-medium">How to enable microphone:</p>
                  <ul className="list-disc space-y-1 pl-5">
                    <li>Click the lock/info icon in the address bar</li>
                    <li>Find "Microphone" and set to "Allow"</li>
                    <li>Reload the page</li>
                  </ul>
                  <p className="text-muted-foreground">
                    You can still type your question in text mode.
                  </p>
                </div>
              )}
            </div>
            <Button
              onClick={clearError}
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              aria-label="Dismiss error"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </Card>
      )}

      {/* Input Area */}
      <Card className="p-6">
        <div className="space-y-4">
          <label htmlFor="question-input" className="text-sm">
            Your Question
          </label>
          <Textarea
            id="question-input"
            value={displayQuestion}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              mode === 'voice'
                ? 'Tap microphone and speak...'
                : 'Type your question here...'
            }
            className="min-h-[120px] text-base"
            disabled={isRecording || isLoading}
            aria-label="Question input"
          />
          {isRecording && (
            <p className="text-sm text-secondary animate-pulse" aria-live="polite">
              🎤 Recording... Speak now
            </p>
          )}
        </div>
      </Card>

      {/* Action Buttons */}
      <div className="flex gap-3">
        {mode === 'voice' && (
          <VoiceButton
            isListening={isRecording}
            onClick={handleVoiceToggle}
            disabled={isLoading}
            className="flex-1"
          />
        )}
        <Button
          onClick={handleAsk}
          disabled={!question.trim() || isLoading || isRecording}
          size="lg"
          className="flex-1"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" aria-hidden="true" />
              Processing...
            </>
          ) : (
            <>
              <Send className="mr-2 h-5 w-5" aria-hidden="true" />
              Ask Question
            </>
          )}
        </Button>
      </div>

      {/* Current Answer */}
      {currentAnswer && (
        <div className="space-y-4">
          <h2 className="text-lg">Answer</h2>
          <AudioPlayer text={currentAnswer} autoPlay={true} />
          <Card className="p-6">
            <p className="leading-relaxed">{currentAnswer}</p>
          </Card>
        </div>
      )}

      {/* Q&A History */}
      {qaHistory.length > 0 && !currentAnswer && (
        <div className="space-y-4">
          <h2 className="text-lg">Previous Questions</h2>
          <div className="space-y-4">
            {qaHistory.slice().reverse().map((qa, index) => (
              <Card key={index} className="p-4">
                <div className="space-y-3">
                  <div>
                    <p className="text-sm text-muted-foreground">Question:</p>
                    <p>{qa.question}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Answer:</p>
                    <p className="text-sm">{qa.answer}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};