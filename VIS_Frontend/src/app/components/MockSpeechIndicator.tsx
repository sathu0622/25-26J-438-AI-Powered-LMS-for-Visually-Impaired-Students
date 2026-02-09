import { useEffect, useState } from 'react';
import { Volume2, VolumeX } from 'lucide-react';
import { Card } from './ui/card';

interface MockSpeechIndicatorProps {
  className?: string;
}

/**
 * Visual indicator showing when mock speech is active
 * Helps users understand that voice output is working in demo mode
 */
export const MockSpeechIndicator = ({ className = '' }: MockSpeechIndicatorProps) => {
  const [currentText, setCurrentText] = useState<string>('');
  const [isActive, setIsActive] = useState(false);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    // Intercept console.log to catch mock speech announcements
    const originalLog = console.log;
    
    console.log = (...args: any[]) => {
      const message = args.join(' ');
      
      // Check if this is a mock TTS message
      if (message.includes('[Mock TTS] Speaking')) {
        const match = message.match(/Speaking \((\d+)s\): (.+)/);
        if (match) {
          const seconds = parseInt(match[1]);
          const text = match[2];
          setCurrentText(text);
          setIsActive(true);
          setDuration(seconds);
          
          // Auto-hide after duration
          setTimeout(() => {
            setIsActive(false);
            setTimeout(() => setCurrentText(''), 500);
          }, seconds * 1000);
        }
      } else if (message.includes('[Mock TTS] Stopped') || message.includes('[Mock TTS] Cancelled')) {
        setIsActive(false);
        setTimeout(() => setCurrentText(''), 500);
      }
      
      // Call original log
      originalLog.apply(console, args);
    };

    return () => {
      console.log = originalLog;
    };
  }, []);

  if (!currentText) return null;

  return (
    <Card 
      className={`fixed bottom-20 left-4 right-4 z-50 border-2 shadow-lg transition-all ${
        isActive 
          ? 'border-primary bg-primary/10 opacity-100' 
          : 'border-muted bg-muted/50 opacity-70'
      } ${className}`}
      style={{ maxWidth: '600px', margin: '0 auto' }}
    >
      <div className="p-4">
        <div className="flex items-start gap-3">
          <div className="shrink-0 pt-1">
            {isActive ? (
              <Volume2 className="h-6 w-6 animate-pulse text-primary" />
            ) : (
              <VolumeX className="h-6 w-6 text-muted-foreground" />
            )}
          </div>
          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold">
                {isActive ? '🎙️ Voice Output (Speaking...)' : '🎙️ Voice Output (Finished)'}
              </p>
              {isActive && duration > 0 && (
                <span className="rounded-full bg-primary/20 px-2 py-0.5 text-xs text-primary">
                  ~{duration}s
                </span>
              )}
            </div>
            <p className="text-sm leading-relaxed">{currentText}</p>
          </div>
        </div>
      </div>
    </Card>
  );
};