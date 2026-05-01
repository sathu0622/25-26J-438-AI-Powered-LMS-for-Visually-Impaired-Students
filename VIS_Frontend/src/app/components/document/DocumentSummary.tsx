import { useState, useEffect } from 'react';
import { ZoomIn, ZoomOut, MessageSquare, Keyboard } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { AudioPlayer } from '../AudioPlayer';

interface DocumentSummaryProps {
  summary: string;
  onAskQuestion: (mode: 'voice' | 'text') => void;
}

export const DocumentSummary = ({ summary, onAskQuestion }: DocumentSummaryProps) => {
  const [textSize, setTextSize] = useState(100);
  const [hasAutoPlayed, setHasAutoPlayed] = useState(false);

  // Auto-play summary when page loads
  useEffect(() => {
    // STOP all previous speech immediately
    window.speechSynthesis.cancel();
    
    if (!hasAutoPlayed) {
      setHasAutoPlayed(true);
      // Announce page and auto-play
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance('Document Summary page. Press A to replay summary, Press V for voice question, Press T for text question, Press Plus to increase text size, Press Minus to decrease text size. Playing summary now.');
        window.speechSynthesis.speak(utterance);
        
        // Auto-play summary after announcement
        setTimeout(() => {
          const summaryUtterance = new SpeechSynthesisUtterance(summary);
          window.speechSynthesis.speak(summaryUtterance);
        }, 6000);
      }, 500);
    }
    
    // Cleanup: stop speech when leaving page
    return () => {
      window.speechSynthesis.cancel();
    };
  }, [summary, hasAutoPlayed]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // A key to replay summary
      if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        const utterance = new SpeechSynthesisUtterance(summary);
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      }

      // V key for voice question
      if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        onAskQuestion('voice');
      }

      // T key for text question
      if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        onAskQuestion('text');
      }

      // Plus/Equals key to increase text size
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        increaseTextSize();
      }

      // Minus key to decrease text size
      if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        decreaseTextSize();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [summary, onAskQuestion]);

  const increaseTextSize = () => {
    setTextSize((prev) => Math.min(prev + 10, 150));
  };

  const decreaseTextSize = () => {
    setTextSize((prev) => Math.max(prev - 10, 80));
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-2xl">Document Summary</h1>
        <p className="text-muted-foreground">
          A to replay • V for voice question • T for text question • +/- to resize text
        </p>
      </div>

      {/* Audio Player */}
      <AudioPlayer text={summary} autoPlay={false} />

      {/* Text Controls */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">Text Size</p>
        <div className="flex gap-2">
          <Button
            onClick={decreaseTextSize}
            variant="outline"
            size="sm"
            aria-label="Decrease text size"
          >
            <ZoomOut className="h-4 w-4" aria-hidden="true" />
          </Button>
          <span className="flex items-center px-2 text-sm" aria-live="polite">
            {textSize}%
          </span>
          <Button
            onClick={increaseTextSize}
            variant="outline"
            size="sm"
            aria-label="Increase text size"
          >
            <ZoomIn className="h-4 w-4" aria-hidden="true" />
          </Button>
        </div>
      </div>

      {/* Summary Text */}
      <Card className="p-6">
        <div
          className="space-y-4"
          style={{ fontSize: `${textSize}%` }}
          role="article"
          aria-label="Document summary"
        >
          <h2 className="text-lg">Summary</h2>
          <p className="leading-relaxed">{summary}</p>
        </div>
      </Card>

      {/* Question Options */}
      <div className="space-y-3">
        <h2 className="text-center">Ask a Question About This Document</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <Button
            onClick={() => onAskQuestion('voice')}
            size="lg"
            className="min-h-[72px] flex-col gap-2"
          >
            <MessageSquare className="h-6 w-6" aria-hidden="true" />
            <span>Voice Question</span>
          </Button>
          <Button
            onClick={() => onAskQuestion('text')}
            size="lg"
            variant="outline"
            className="min-h-[72px] flex-col gap-2"
          >
            <Keyboard className="h-6 w-6" aria-hidden="true" />
            <span>Text Question</span>
          </Button>
        </div>
      </div>
    </div>
  );
};