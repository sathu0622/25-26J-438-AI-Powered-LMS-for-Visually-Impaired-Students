import { Mic, MicOff } from 'lucide-react';
import { Button } from './ui/button';
import { cn } from './ui/utils';

interface VoiceButtonProps {
  isListening: boolean;
  onClick: () => void;
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
  disabled?: boolean;
}

export const VoiceButton = ({
  isListening,
  onClick,
  size = 'lg',
  className,
  disabled = false,
}: VoiceButtonProps) => {
  return (
    <Button
      onClick={onClick}
      size={size}
      disabled={disabled}
      className={cn(
        'min-h-[56px] min-w-[56px] transition-all',
        isListening && 'bg-destructive hover:bg-destructive/90 animate-pulse',
        className
      )}
      aria-label={isListening ? 'Stop listening' : 'Start listening'}
      aria-pressed={isListening}
    >
      {isListening ? (
        <MicOff className="h-6 w-6" aria-hidden="true" />
      ) : (
        <Mic className="h-6 w-6" aria-hidden="true" />
      )}
    </Button>
  );
};
