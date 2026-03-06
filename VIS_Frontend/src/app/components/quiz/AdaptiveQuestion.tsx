import { useEffect, useMemo, useState } from 'react';
import { Volume2, Mic, Send, Flag, Play, Loader2, CheckCircle2, XCircle, AlertCircle, Check } from 'lucide-react';
import { AdaptiveItem, AdaptiveAnswerResponse } from '../../services/adaptiveService';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { VoiceRecorder } from '../MockVoiceRecorder';
import { useTTS } from '../../contexts/TTSContext';

interface AdaptiveQuestionProps {
  item: AdaptiveItem;
  onSubmit: (answer: string) => void;
  onFinish: () => void;
  onBack?: () => void;
  feedback?: string;
  theta: number;
  loading?: boolean;
  answeredCount?: number;
  lastResult?: AdaptiveAnswerResponse | null;
  lastQuestion?: AdaptiveItem | null;
  lastAnswer?: string;
}

export const AdaptiveQuestion = ({
  item,
  onSubmit,
  onFinish,
  onBack,
  feedback,
  theta,
  loading,
  answeredCount = 0,
  lastResult,
  lastQuestion,
  lastAnswer,
}: AdaptiveQuestionProps) => {
  const [answer, setAnswer] = useState('');
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const { speak, cancel } = useTTS();

  const questionNumber = useMemo(() => answeredCount + 1, [answeredCount]);
  
  // Check if this is an MCQ question
  const isMCQ = item.options && item.options.length > 0;
  const optionLabels = ['A', 'B', 'C', 'D'];

  useEffect(() => {
    cancel();
    setSelectedOption(null);
    setAnswer('');
    const timer = setTimeout(() => {
      if (isMCQ && item.options) {
        // Read MCQ options
        const optionsText = item.options
          .map((opt, idx) => `${optionLabels[idx]}: ${opt}`)
          .join('. ');
        speak(
          `Adaptive question ${questionNumber}. Difficulty ${item.difficulty_label}. ${item.question}. This is a multiple choice question. ${optionsText}. Press A, B, C, or D to select. Press Q to repeat. Press F to finish. Press Backspace to go back.`,
          { interrupt: true }
        );
      } else {
        speak(
          `Adaptive question ${questionNumber}. Difficulty ${item.difficulty_label}. ${item.question}. Press Space or Enter to record or submit, Q to repeat, R to record, F to finish, and Backspace to go back.`,
          { interrupt: true }
        );
      }
    }, 400);

    return () => {
      clearTimeout(timer);
      cancel();
    };
  }, [item.item_id, questionNumber, item.question, item.difficulty_label, speak, cancel]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName;
      const isTyping = targetTag === 'TEXTAREA' || targetTag === 'INPUT';

      // MCQ option selection (A, B, C, D)
      if (isMCQ && item.options && !isTyping) {
        const key = e.key.toUpperCase();
        const optionIndex = optionLabels.indexOf(key);
        if (optionIndex !== -1 && optionIndex < item.options.length) {
          e.preventDefault();
          handleOptionSelect(optionIndex);
          return;
        }
      }

      if ((e.key === ' ' || e.key === 'Enter') && !isTyping) {
        e.preventDefault();
        if (isMCQ && selectedOption !== null) {
          handleSubmit();
        } else if (answer.trim() && !showVoiceModal) {
          handleSubmit();
        } else if (!isMCQ) {
          handleVoiceToggle();
        }
      }

      if ((e.key === 'r' || e.key === 'R') && !isTyping && !isMCQ) {
        e.preventDefault();
        handleVoiceToggle();
      }

      if ((e.key === 'q' || e.key === 'Q') && !isTyping) {
        e.preventDefault();
        handleReadQuestion();
      }

      if ((e.key === 'f' || e.key === 'F') && !isTyping) {
        e.preventDefault();
        onFinish();
      }

      // Backspace or B key to go back
      if ((e.key === 'Backspace' || e.key === 'b' || e.key === 'B') && !isTyping) {
        e.preventDefault();
        if (onBack) onBack();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [answer, showVoiceModal, onFinish, onBack, selectedOption, isMCQ]);

  useEffect(() => {
    if (lastResult && lastQuestion) {
      const correctness = lastResult.correct ? 'correct' : 'incorrect';
      const correctAnswerText = lastResult.correct_answer ? `Correct answer: ${lastResult.correct_answer}.` : '';
      const yourAnswerText = lastAnswer ? `Your answer: ${lastAnswer}.` : 'No answer captured.';
      
      // Check for difficulty level transition
      const difficultyOrder = ['easy', 'medium', 'hard'];
      const prevDifficulty = lastQuestion.difficulty_label?.toLowerCase() || 'medium';
      const currDifficulty = item.difficulty_label?.toLowerCase() || 'medium';
      const prevIndex = difficultyOrder.indexOf(prevDifficulty);
      const currIndex = difficultyOrder.indexOf(currDifficulty);
      
      let difficultyMessage = '';
      if (currIndex > prevIndex) {
        difficultyMessage = `Congratulations! You advanced to ${item.difficulty_label} difficulty level.`;
      } else if (currIndex < prevIndex) {
        difficultyMessage = `The system adjusted to ${item.difficulty_label} difficulty to help you learn better.`;
      }
      
      speak(`Previous question feedback. You were ${correctness}. ${yourAnswerText} ${correctAnswerText} ${difficultyMessage}`, { interrupt: true });
    } else if (feedback) {
      speak(`Feedback: ${feedback}`, { interrupt: true });
    }
    return () => cancel();
  }, [lastResult, lastQuestion, lastAnswer, feedback, speak, cancel]);

  const handleReadQuestion = () => {
    cancel();
    if (isMCQ && item.options) {
      const optionsText = item.options
        .map((opt, idx) => `${optionLabels[idx]}: ${opt}`)
        .join('. ');
      speak(`Question ${questionNumber}. ${item.question}. Options: ${optionsText}`, { interrupt: true });
    } else {
      speak(`Question ${questionNumber}. ${item.question}`);
    }
  };

  const handleOptionSelect = (index: number) => {
    if (!item.options) return;
    setSelectedOption(index);
    const selectedText = item.options[index];
    speak(`Selected ${optionLabels[index]}: ${selectedText}. Press Enter to submit.`, { interrupt: true });
  };

  const handleVoiceToggle = () => {
    setInputMode('voice');
    setShowVoiceModal(true);
  };

  const handleVoiceSubmit = (text: string) => {
    setAnswer(text);
    setShowVoiceModal(false);
    setInputMode('voice');
  };

  const handleSubmit = () => {
    if (loading) return;
    
    if (isMCQ && selectedOption !== null && item.options) {
      // For MCQ, submit the selected option text
      onSubmit(item.options[selectedOption]);
      setSelectedOption(null);
      setAnswer('');
      setShowVoiceModal(false);
    } else if (answer.trim()) {
      onSubmit(answer.trim());
      setAnswer('');
      setShowVoiceModal(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6 pb-24">
      <div className="space-y-2" aria-live="polite">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Adaptive • Question {questionNumber}</span>
          <span>Answered: {answeredCount}</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted" aria-hidden>
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${Math.min(100, Math.max(10, answeredCount ? ((answeredCount) / (answeredCount + 1)) * 100 : 20))}%` }}
          />
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1"><Volume2 className="h-4 w-4" />Press Q to repeat</span>
          {isMCQ ? (
            <span className="inline-flex items-center gap-1">Press A/B/C/D to select</span>
          ) : (
            <span className="inline-flex items-center gap-1"><Mic className="h-4 w-4" />Press R to record</span>
          )}
          <span className="inline-flex items-center gap-1"><Flag className="h-4 w-4" />Press F to finish</span>
          <span className="inline-flex items-center gap-1">Press B to go back</span>
        </div>
      </div>

      <Card className="p-6 space-y-4" role="group" aria-labelledby="adaptive-question-heading">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 space-y-2">
            <p className="text-xs uppercase tracking-wide text-primary font-semibold">Difficulty: {item.difficulty_label}</p>
            <h2 id="adaptive-question-heading" className="text-xl font-semibold leading-relaxed">{item.question}</h2>
          </div>
          <div className="flex flex-col items-end gap-2 text-sm text-muted-foreground">
            <Button
              onClick={handleReadQuestion}
              variant="outline"
              size="icon"
              aria-label="Read question aloud"
            >
              <Volume2 className="h-5 w-5" />
            </Button>
            
          </div>
        </div>
      </Card>

      {/* MCQ Options */}
      {isMCQ && item.options ? (
        <Card className="p-6">
          <div className="space-y-3">
            <label className="text-sm font-medium">Select your answer:</label>
            <div className="space-y-3">
              {item.options.map((option, index) => (
                <Button
                  key={index}
                  onClick={() => handleOptionSelect(index)}
                  variant={selectedOption === index ? 'default' : 'outline'}
                  className={`w-full justify-start min-h-[56px] text-left p-4 ${
                    selectedOption === index 
                      ? 'ring-2 ring-primary ring-offset-2' 
                      : ''
                  }`}
                  aria-label={`Option ${optionLabels[index]}: ${option}`}
                >
                  <span className="flex items-center gap-3 w-full">
                    <span className={`flex h-8 w-8 items-center justify-center rounded-full border-2 font-semibold ${
                      selectedOption === index 
                        ? 'bg-primary-foreground text-primary border-primary-foreground' 
                        : 'border-muted-foreground'
                    }`}>
                      {optionLabels[index]}
                    </span>
                    <span className="flex-1 text-base">{option}</span>
                    {selectedOption === index && (
                      <Check className="h-5 w-5 text-primary-foreground" />
                    )}
                  </span>
                </Button>
              ))}
            </div>
          </div>
        </Card>
      ) : (
        <>
          {/* Input Mode Toggle for non-MCQ */}
          <div className="flex justify-center gap-2">
            <Button
              onClick={() => setInputMode('voice')}
              variant={inputMode === 'voice' ? 'default' : 'outline'}
              size="sm"
            >
              <Mic className="mr-2 h-4 w-4" />
              Voice
            </Button>
            <Button
              onClick={() => setInputMode('text')}
              variant={inputMode === 'text' ? 'default' : 'outline'}
              size="sm"
            >
              Text
            </Button>
          </div>

          <Card className="p-6 space-y-4">
            <label htmlFor="adaptive-answer" className="text-sm font-medium">Your answer</label>
            <Textarea
              id="adaptive-answer"
              value={answer}
              onChange={(e) => {
                setAnswer(e.target.value);
                setInputMode('text');
              }}
              placeholder={inputMode === 'voice' ? 'Tap microphone and speak your answer…' : 'Type your answer here…'}
              className="min-h-[140px] text-base"
            />
          </Card>
        </>
      )}

      {/* Action Buttons */}
      <div className="grid gap-3 sm:grid-cols-3">
        {!isMCQ && inputMode === 'voice' && (
          <Button
            onClick={handleVoiceToggle}
            variant="outline"
            size="lg"
            className="min-h-[56px]"
            disabled={loading}
          >
            <Mic className="mr-2 h-5 w-5" />
            Record Answer
          </Button>
        )}
        <Button
          onClick={handleSubmit}
          disabled={isMCQ ? selectedOption === null || Boolean(loading) : !answer.trim() || Boolean(loading)}
          size="lg"
          className="min-h-[56px]"
        >
          {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Send className="mr-2 h-5 w-5" />}
          Submit Answer
        </Button>
        <Button
          variant="ghost"
          onClick={onFinish}
          size="lg"
          className="min-h-[56px]"
        >
          <Flag className="mr-2 h-5 w-5" />
          Finish Session
        </Button>
      </div>

      {(lastResult || feedback) && lastQuestion && (
        <Card className="p-6 bg-muted/60 border border-muted" aria-live="assertive">
          <div className="flex items-start gap-4">
            <div className="pt-1">
              {lastResult ? (
                lastResult.correct ? (
                  <CheckCircle2 className="h-8 w-8 text-green-500" />
                ) : (
                  <XCircle className="h-8 w-8 text-red-500" />
                )
              ) : (
                <AlertCircle className="h-8 w-8 text-blue-500" />
              )}
            </div>
            <div className="flex-1 space-y-2">
              <p className="text-sm uppercase tracking-wide text-primary font-semibold">Previous feedback</p>
              <p className="text-lg font-semibold">
                {lastResult ? (lastResult.correct ? 'Correct' : 'Incorrect') : 'Feedback'}
              </p>
              <p className="text-sm text-muted-foreground">Previous question: {lastQuestion.question}</p>
              <p className="text-sm">Your answer: {lastAnswer || 'No answer captured.'}</p>
              <p className="text-sm">Correct answer: {lastResult?.correct_answer || 'Not provided.'}</p>
              
              {/* Difficulty level transition */}
              {(() => {
                const difficultyOrder = ['easy', 'medium', 'hard'];
                const prevDifficulty = lastQuestion.difficulty_label?.toLowerCase() || 'medium';
                const currDifficulty = item.difficulty_label?.toLowerCase() || 'medium';
                const prevIndex = difficultyOrder.indexOf(prevDifficulty);
                const currIndex = difficultyOrder.indexOf(currDifficulty);
                
                if (currIndex > prevIndex) {
                  return (
                    <div className="mt-2 p-2 rounded-lg text-sm font-medium bg-green-100 text-green-700 border border-green-300">
                      🎉 Advancing to {item.difficulty_label} difficulty!
                    </div>
                  );
                } else if (currIndex < prevIndex) {
                  return (
                    <div className="mt-2 p-2 rounded-lg text-sm font-medium bg-yellow-100 text-yellow-700 border border-yellow-300">
                      Adjusted to {item.difficulty_label} difficulty to help you learn better.
                    </div>
                  );
                }
                return null;
              })()}
            </div>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Speak feedback"
              onClick={() => {
                const correctness = lastResult?.correct ? 'correct' : 'incorrect';
                const correctAnswerText = lastResult?.correct_answer ? `Correct answer: ${lastResult.correct_answer}.` : '';
                const yourAnswerText = lastAnswer ? `Your answer: ${lastAnswer}.` : 'No answer captured.';
                
                // Check for difficulty level transition
                const difficultyOrder = ['easy', 'medium', 'hard'];
                const prevDifficulty = lastQuestion.difficulty_label?.toLowerCase() || 'medium';
                const currDifficulty = item.difficulty_label?.toLowerCase() || 'medium';
                const prevIndex = difficultyOrder.indexOf(prevDifficulty);
                const currIndex = difficultyOrder.indexOf(currDifficulty);
                
                let difficultyMessage = '';
                if (currIndex > prevIndex) {
                  difficultyMessage = `You advanced to ${item.difficulty_label} difficulty level!`;
                } else if (currIndex < prevIndex) {
                  difficultyMessage = `Adjusted to ${item.difficulty_label} difficulty to help you learn.`;
                }
                
                speak(`Previous question feedback. You were ${correctness}. ${yourAnswerText} ${correctAnswerText} ${difficultyMessage}`, { interrupt: true });
              }}
            >
              <Play className="h-4 w-4" />
            </Button>
          </div>
        </Card>
      )}

      {/* Voice Recorder Modal (only for non-MCQ) */}
      {!isMCQ && (
        <VoiceRecorder
          isOpen={showVoiceModal}
          onClose={() => setShowVoiceModal(false)}
          onSubmit={handleVoiceSubmit}
          title="Record your answer"
          context={item.question}
        />
      )}
    </div>
  );
};
