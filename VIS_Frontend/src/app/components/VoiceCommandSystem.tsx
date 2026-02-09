import { useEffect, useState, useCallback } from 'react';
import { Mic, MicOff, HelpCircle, X } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface VoiceCommandSystemProps {
  onNavigate: (route: string) => void;
  currentPage: string;
}

export const VoiceCommandSystem = ({ onNavigate, currentPage }: VoiceCommandSystemProps) => {
  const [isListening, setIsListening] = useState(false);
  const [lastCommand, setLastCommand] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const [showCommandPicker, setShowCommandPicker] = useState(false);

  const commands = {
    // Navigation commands
    'go to document upload': 'document-upload',
    'document upload': 'document-upload',
    'upload document': 'document-upload',
    'go to braille': 'braille',
    'braille evaluation': 'braille',
    'braille sheet': 'braille',
    'go to quiz': 'quiz',
    'take quiz': 'quiz',
    'quiz system': 'quiz',
    'go to history': 'history',
    'history lessons': 'history',
    'audio lessons': 'history',
    'go home': 'home',
    'home page': 'home',
    'main menu': 'home',
    // Help commands
    'help': 'help',
    'show help': 'help',
    'what can i say': 'help',
    'voice commands': 'help',
  };

  const quickCommands = [
    { text: 'Go to Document Upload', command: 'go to document upload' },
    { text: 'Go to Braille Evaluation', command: 'go to braille' },
    { text: 'Take Quiz', command: 'take quiz' },
    { text: 'Go to History Lessons', command: 'go to history' },
    { text: 'Go Home', command: 'go home' },
    { text: 'Show Help', command: 'help' },
  ];

  const speakText = useCallback((text: string) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);
  }, []);

  const processCommand = useCallback((transcript: string) => {
    const normalizedCommand = transcript.toLowerCase().trim();
    setLastCommand(normalizedCommand);

    // Check for exact or partial matches
    for (const [command, route] of Object.entries(commands)) {
      if (normalizedCommand.includes(command)) {
        if (route === 'help') {
          setShowHelp(true);
          speakText('Showing voice commands help. You can say commands like: go to document upload, go to braille evaluation, take quiz, or go to history lessons.');
        } else {
          speakText(`Navigating to ${command}`);
          setTimeout(() => onNavigate(route), 500);
        }
        return;
      }
    }

    // If no command matched
    speakText('Command not recognized. Say "help" to hear available commands.');
  }, [onNavigate, speakText, commands]);

  const startVoiceCommand = useCallback(() => {
    // Try to use real voice recognition first
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      // Fallback to command picker if not supported
      speakText('Voice recognition not supported in this browser. Showing command menu instead.');
      setShowCommandPicker(true);
      return;
    }

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      speakText('Listening for command. Please speak now.');
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      console.log('Voice command heard:', transcript);
      processCommand(transcript);
    };

    recognition.onerror = (event: any) => {
      console.error('Voice recognition error:', event.error);
      setIsListening(false);
      
      if (event.error === 'not-allowed' || event.error === 'permission-denied') {
        speakText('Microphone permission denied. Please allow microphone access in your browser settings, or use keyboard shortcuts instead. Showing command menu.');
        setTimeout(() => setShowCommandPicker(true), 1500);
      } else if (event.error === 'no-speech') {
        speakText('No speech detected. Please try again.');
      } else if (event.error === 'aborted') {
        // Silent abort, user stopped it
      } else {
        speakText('Voice recognition error. Showing command menu instead.');
        setTimeout(() => setShowCommandPicker(true), 1000);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    try {
      recognition.start();
    } catch (error) {
      console.error('Error starting voice recognition:', error);
      speakText('Could not start voice commands. Showing command menu instead.');
      setIsListening(false);
      setShowCommandPicker(true);
    }
  }, [processCommand, speakText]);

  // Announce current page on mount or change
  useEffect(() => {
    const pageNames: Record<string, string> = {
      'home': 'Home page - Main menu',
      'document-upload': 'Document Upload and Q&A module',
      'braille': 'Braille Answer Sheet Evaluation module',
      'quiz': 'Voice-Enabled Quiz System module',
      'history': 'Audio History Learning module',
    };

    const announcement = pageNames[currentPage] || 'Page loaded';
    setTimeout(() => {
      speakText(`${announcement}. Press F1 for voice commands, or H for help.`);
    }, 500);
  }, [currentPage, speakText]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Escape - Close modals
      if (e.key === 'Escape') {
        if (showCommandPicker) {
          setShowCommandPicker(false);
          return;
        }
        if (showHelp) {
          setShowHelp(false);
          return;
        }
      }

      // F1 - Voice command
      if (e.key === 'F1') {
        e.preventDefault();
        startVoiceCommand();
        return;
      }

      // H - Help
      if (e.key === 'h' || e.key === 'H') {
        if (!e.target || (e.target as HTMLElement).tagName !== 'INPUT' && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
          e.preventDefault();
          setShowHelp(true);
          speakText('Showing keyboard shortcuts and voice commands help');
        }
        return;
      }

      // Alt + Number shortcuts
      if (e.altKey) {
        e.preventDefault();
        switch (e.key) {
          case '1':
            speakText('Going to document upload');
            onNavigate('document-upload');
            break;
          case '2':
            speakText('Going to braille evaluation');
            onNavigate('braille');
            break;
          case '3':
            speakText('Going to quiz system');
            onNavigate('quiz');
            break;
          case '4':
            speakText('Going to history lessons');
            onNavigate('history');
            break;
          case '0':
            speakText('Going to home page');
            onNavigate('home');
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [onNavigate, startVoiceCommand, speakText, showHelp, showCommandPicker]);

  return (
    <>
      {/* Voice Command Button - Fixed Position */}
      <div 
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2"
        role="region"
        aria-label="Voice command controls"
      >
        <Button
          onClick={() => setShowHelp(!showHelp)}
          size="icon"
          variant="secondary"
          className="h-14 w-14 rounded-full shadow-lg"
          aria-label="Toggle help menu - Press H or F1 for voice commands"
          title="Help (Press H)"
        >
          <HelpCircle className="h-6 w-6" />
        </Button>
        
        <Button
          onClick={startVoiceCommand}
          disabled={isListening}
          size="icon"
          className={`h-14 w-14 rounded-full shadow-lg transition-all ${
            isListening 
              ? 'animate-pulse bg-red-500 hover:bg-red-600 scale-110' 
              : 'bg-primary hover:bg-primary/90'
          }`}
          aria-label={isListening ? 'Listening for voice command...' : 'Activate voice command - Press F1'}
          title="Voice Command (Press F1)"
        >
          {isListening ? (
            <div className="relative">
              <Mic className="h-6 w-6 animate-pulse" />
              <span className="absolute -top-1 -right-1 flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-white"></span>
              </span>
            </div>
          ) : (
            <Mic className="h-6 w-6" />
          )}
        </Button>
        
        {/* Listening Indicator */}
        {isListening && (
          <div className="absolute -top-16 right-0 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg animate-pulse">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-white rounded-full animate-ping"></div>
              <span className="text-sm font-medium">Listening...</span>
            </div>
          </div>
        )}
      </div>

      {/* Help Modal */}
      {showHelp && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="help-title"
        >
          <Card className="max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h2 id="help-title" className="text-2xl">Accessibility Help</h2>
              <Button
                onClick={() => setShowHelp(false)}
                variant="ghost"
                size="icon"
                aria-label="Close help menu"
              >
                <X className="h-6 w-6" />
              </Button>
            </div>

            <div className="space-y-6">
              {/* Keyboard Shortcuts */}
              <section aria-labelledby="keyboard-shortcuts">
                <h3 id="keyboard-shortcuts" className="text-xl mb-3">Keyboard Shortcuts</h3>
                <ul className="space-y-2 text-sm">
                  <li><kbd className="px-2 py-1 bg-secondary rounded">F1</kbd> - Activate voice commands</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">H</kbd> - Show this help menu</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Alt + 0</kbd> - Go to Home</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Alt + 1</kbd> - Document Upload & Q&A</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Alt + 2</kbd> - Braille Evaluation</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Alt + 3</kbd> - Quiz System</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Alt + 4</kbd> - History Lessons</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Tab</kbd> - Navigate between elements</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Enter</kbd> - Activate buttons/links</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Space</kbd> - Play/pause audio</li>
                  <li><kbd className="px-2 py-1 bg-secondary rounded">Escape</kbd> - Close dialogs or go back</li>
                </ul>
              </section>

              {/* Voice Commands */}
              <section aria-labelledby="voice-commands">
                <h3 id="voice-commands" className="text-xl mb-3">Voice Commands</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  Press F1 or click the microphone button, then say:
                </p>
                <ul className="space-y-2 text-sm">
                  <li><strong>"Go to document upload"</strong> - Open document module</li>
                  <li><strong>"Go to braille"</strong> - Open braille evaluation</li>
                  <li><strong>"Take quiz"</strong> - Open quiz system</li>
                  <li><strong>"Go to history"</strong> - Open history lessons</li>
                  <li><strong>"Go home"</strong> - Return to main menu</li>
                  <li><strong>"Help"</strong> - Show this help menu</li>
                </ul>
              </section>

              {/* Screen Reader Tips */}
              <section aria-labelledby="screen-reader-tips">
                <h3 id="screen-reader-tips" className="text-xl mb-3">Screen Reader Tips</h3>
                <ul className="space-y-2 text-sm">
                  <li>All buttons and interactive elements are properly labeled</li>
                  <li>Use headings navigation to jump between sections</li>
                  <li>Form inputs have descriptive labels</li>
                  <li>Audio content auto-announces when available</li>
                  <li>Status updates are announced via live regions</li>
                </ul>
              </section>
            </div>

            <Button
              onClick={() => setShowHelp(false)}
              size="lg"
              className="w-full min-h-[56px]"
              aria-label="Close help and return to page"
            >
              Close Help
            </Button>
          </Card>
        </div>
      )}

      {/* Command Picker Modal */}
      {showCommandPicker && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="command-picker-title"
        >
          <Card className="max-w-lg w-full p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h2 id="command-picker-title" className="text-2xl">🎤 Voice Commands</h2>
                <p className="text-sm text-muted-foreground">
                  Select a command to navigate
                </p>
              </div>
              <Button
                onClick={() => setShowCommandPicker(false)}
                variant="ghost"
                size="icon"
                aria-label="Close command picker"
              >
                <X className="h-6 w-6" />
              </Button>
            </div>

            <div className="space-y-3">
              {quickCommands.map((cmd, index) => (
                <Button
                  key={index}
                  onClick={() => {
                    processCommand(cmd.command);
                    setShowCommandPicker(false);
                  }}
                  size="lg"
                  variant={cmd.text.includes('Home') ? 'secondary' : 'default'}
                  className="w-full min-h-[56px] justify-start text-left"
                  aria-label={`Execute command: ${cmd.text}`}
                >
                  <span className="mr-2">
                    {cmd.text.includes('Document') && '📄'}
                    {cmd.text.includes('Braille') && '⠿'}
                    {cmd.text.includes('Quiz') && '❓'}
                    {cmd.text.includes('History') && '📚'}
                    {cmd.text.includes('Home') && '🏠'}
                    {cmd.text.includes('Help') && '❔'}
                  </span>
                  {cmd.text}
                </Button>
              ))}
            </div>

            <div className="pt-4 border-t">
              <p className="text-xs text-center text-muted-foreground">
                💡 Tip: Use keyboard shortcuts Alt+0 to Alt+4 for quick navigation
              </p>
            </div>
          </Card>
        </div>
      )}

      {/* Last Command Display (for debugging) */}
      {lastCommand && (
        <div 
          className="fixed bottom-24 right-4 z-40 max-w-xs"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <Card className="p-3 bg-secondary text-sm">
            Last command: "{lastCommand}"
          </Card>
        </div>
      )}
    </>
  );
};