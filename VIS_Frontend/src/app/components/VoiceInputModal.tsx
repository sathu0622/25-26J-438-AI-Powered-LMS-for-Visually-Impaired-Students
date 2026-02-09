import { useState, useEffect } from 'react';
import { Mic, X } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';

interface VoiceInputModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
  title?: string;
  placeholder?: string;
}

export const VoiceInputModal = ({
  isOpen,
  onClose,
  onSubmit,
  title = 'Voice Input',
  placeholder = 'Type what you would say...',
}: VoiceInputModalProps) => {
  const [inputText, setInputText] = useState('');

  // Reset input when modal opens
  useEffect(() => {
    if (isOpen) {
      setInputText('');
    }
  }, [isOpen]);

  // Focus textarea when modal opens
  useEffect(() => {
    if (isOpen) {
      const textarea = document.getElementById('voice-input-textarea');
      textarea?.focus();
    }
  }, [isOpen]);

  const handleSubmit = () => {
    if (inputText.trim()) {
      onSubmit(inputText.trim());
      setInputText('');
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="voice-input-title"
    >
      <div
        className="relative w-full max-w-md rounded-lg bg-background p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mic className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 id="voice-input-title" className="text-lg">
              {title}
            </h2>
          </div>
          <Button
            onClick={onClose}
            variant="ghost"
            size="icon"
            aria-label="Close dialog"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Info */}
        <p className="mb-4 text-sm text-muted-foreground">
          This is a demo mode. Type your response below to simulate voice input.
          In production, this would use your microphone.
        </p>

        {/* Input */}
        <Textarea
          id="voice-input-textarea"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="mb-4 min-h-[120px] resize-none text-base"
          aria-label={placeholder}
        />

        {/* Actions */}
        <div className="flex gap-2">
          <Button onClick={onClose} variant="outline" className="flex-1">
            Cancel (Esc)
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!inputText.trim()}
            className="flex-1"
          >
            Submit (Ctrl+Enter)
          </Button>
        </div>

        {/* Keyboard shortcuts help */}
        <p className="mt-3 text-center text-xs text-muted-foreground">
          Ctrl+Enter to submit • Esc to cancel
        </p>
      </div>
    </div>
  );
};