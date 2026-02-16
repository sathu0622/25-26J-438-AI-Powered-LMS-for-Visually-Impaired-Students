import { useState, useEffect } from 'react';
import { Send, Loader2, ArrowLeft, AlertCircle, X } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { VoiceButton } from '../VoiceButton';
import { AudioPlayer } from '../AudioPlayer';
import { useSpeechRecognition } from '../../hooks/useSpeechRecognition';
import { documentService } from '../../services/documentService';

interface QAItem {
  question: string;
  answer: string;
  confidence?: number;
  articleHeading?: string;
  timestamp?: string;
  context?: string;
}

interface DocumentQAProps {
  mode: 'voice' | 'text';
  onBack: () => void;
  documentId: string;
  articleId: string | null;
  articleHeading?: string;
}

export const DocumentQA = ({
  mode,
  onBack,
  documentId,
  articleId,
  articleHeading,
}: DocumentQAProps) => {
  const [question, setQuestion] = useState('');
  const [qaHistory, setQaHistory] = useState<QAItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAnswer, setCurrentAnswer] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTimer, setRecordingTimer] = useState<NodeJS.Timeout | null>(null);
  const [hasAutoStarted, setHasAutoStarted] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);

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
    window.speechSynthesis.cancel();
    
    if (mode === 'voice' && !hasAutoStarted) {
      setHasAutoStarted(true);
      // Announce and auto-start
      const utterance = new SpeechSynthesisUtterance('Ask a Question page. Press Space or Enter to record or submit your question. Press R to re-record. Press A to replay answer after receiving it. Press Escape to go back. Recording will start automatically in 2 seconds.');
      window.speechSynthesis.speak(utterance);
      
      // Auto-start after announcement
      setTimeout(() => {
        handleVoiceToggle();
      }, 2000);
    }
    
    // Cleanup: stop speech when leaving page
    return () => {
      window.speechSynthesis.cancel();
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
          window.speechSynthesis.cancel(); // Stop any current speech
          const utterance = new SpeechSynthesisUtterance(currentAnswer);
          window.speechSynthesis.speak(utterance);
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
    if (!question.trim() || !documentId || !articleId) return;

    setIsLoading(true);
    setCurrentAnswer(null);
    setQaError(null);

    try {
      const qaData = await documentService.askQuestion(
        documentId,
        articleId,
        question,
        64,
        0.15
      );

      const timestamp = new Date().toLocaleTimeString();
      const newItem: QAItem = {
        question,
        answer: qaData.answer,
        confidence: qaData.confidence,
        articleHeading: articleHeading || qaData.article_heading,
        timestamp,
        context: qaData.context_preview,
      };

      setQaHistory((prev) => [...prev, newItem]);
      setCurrentAnswer(qaData.answer);
      setQuestion('');
      resetTranscript();

      // Announce that answer is ready and how to replay
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(
          'Answer received. Press A to replay the answer anytime, or press Space to ask another question.'
        );
        window.speechSynthesis.speak(utterance);
      }, 8000);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Failed to get answer. Please try again.';
      setQaError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceToggle = () => {
    if (isRecording) {
      if (recordingTimer) {
        clearTimeout(recordingTimer);
        setRecordingTimer(null);
      }
      setIsRecording(false);
      stopListening();
    } else {
      setIsRecording(true);
      setQuestion('');
      resetTranscript();
      startListening();
      // Auto-stop after 10 seconds to avoid very long recordings
      const timer = setTimeout(() => {
        stopListening();
        setIsRecording(false);
        setRecordingTimer(null);
      }, 10000);
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

      {/* Speech Recognition Error */}
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

      {/* Q&A API Error */}
      {qaError && (
        <Card className="border-destructive bg-destructive/10 p-4">
          <div className="space-y-1 text-sm">
            <p className="font-medium">Question answering error</p>
            <p>{qaError}</p>
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