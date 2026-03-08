import { BookOpen, Headphones, GraduationCap, Mic, MicOff } from 'lucide-react';
import { Card } from '../ui/card';
import { useEffect, useRef, useState } from 'react';
import { safeSpeak, safeCancel } from '../../utils/mockSpeech';

interface HistoryHomeProps {
  onSelectGrade: (grade: number) => void;
}

export const HistoryHome = ({ onSelectGrade }: HistoryHomeProps) => {
  // useRef instead of state to avoid re-renders causing repeat speech
  const hasAnnounced = useRef(false);
  const recognitionRef = useRef<any>(null);
  const [isListening, setIsListening] = useState(false);
  const isListeningRef = useRef(false);
  const listeningTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const restartListeningTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoListenEnabledRef = useRef(true);

  const scheduleAutoListenRestart = (delay = 500) => {
    if (!autoListenEnabledRef.current) return;
    if (restartListeningTimeoutRef.current) {
      clearTimeout(restartListeningTimeoutRef.current);
    }

    restartListeningTimeoutRef.current = setTimeout(() => {
      if (recognitionRef.current && !isListeningRef.current && autoListenEnabledRef.current) {
        try {
          recognitionRef.current.start();
          listeningTimeoutRef.current = setTimeout(() => {
            if (recognitionRef.current) {
              recognitionRef.current.stop();
              isListeningRef.current = false;
              setIsListening(false);
            }
          }, 10000);
        } catch (error) {
          console.error('Error auto-restarting voice recognition:', error);
        }
      }
    }, delay);
  };

  // Initialize voice recognition
  useEffect(() => {
    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        isListeningRef.current = true;
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript.toLowerCase().trim();
        console.log('Voice input detected:', transcript);

        if (transcript.includes('hello')) {
          safeCancel();
          safeSpeak('Yes, say dear.', () => {
            setTimeout(() => startListening(), 500);
          });
          return;
        }

        if (transcript.includes('stop speech')) {
          safeCancel();
          safeSpeak("Okay, I'm silance now, say me what to do?", () => {
            setTimeout(() => startListening(), 500);
          });
          return;
        }

        // Check for grade commands
        if (transcript.includes('grade 10') || transcript.includes('ten') || transcript.match(/\b10\b/)) {
          safeCancel();
          safeSpeak('Grade 10 selected. Loading lessons.', () => {
            setTimeout(() => onSelectGrade(10), 400);
          });
        } else if (transcript.includes('grade 11') || transcript.includes('eleven') || transcript.match(/\b11\b/)) {
          safeCancel();
          safeSpeak('Grade 11 selected. Loading lessons.', () => {
            setTimeout(() => onSelectGrade(11), 400);
          });
        } else {
          safeCancel();
          safeSpeak('Invalid speech. Please select Grade 10 or Grade 11.', () => {
            setTimeout(() => startListening(), 1000);
          });
        }
      };

      recognition.onerror = (event: any) => {
        console.error('Voice recognition error:', event.error);
        isListeningRef.current = false;
        setIsListening(false);

        if (event.error === 'no-speech') {
          console.log('No speech detected, asking again...');
          safeCancel();
          safeSpeak('No speech detected. Please say Grade 10 or Grade 11.', () => {
            setTimeout(() => startListening(), 500);
          });
        }
      };

      recognition.onend = () => {
        isListeningRef.current = false;
        setIsListening(false);
        scheduleAutoListenRestart(300);
      };

      recognitionRef.current = recognition;
    }

    return () => {
      if (listeningTimeoutRef.current) {
        clearTimeout(listeningTimeoutRef.current);
      }
      if (restartListeningTimeoutRef.current) {
        clearTimeout(restartListeningTimeoutRef.current);
      }
      autoListenEnabledRef.current = false;
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, [onSelectGrade]);

  const startListening = () => {
    if (recognitionRef.current && !isListeningRef.current) {
      try {
        if (listeningTimeoutRef.current) {
          clearTimeout(listeningTimeoutRef.current);
        }
        recognitionRef.current.start();
        listeningTimeoutRef.current = setTimeout(() => {
          if (recognitionRef.current) {
            recognitionRef.current.stop();
            isListeningRef.current = false;
            setIsListening(false);
          }
        }, 10000);
      } catch (error) {
        console.error('Error starting voice recognition:', error);
      }
    }
  };

  // Voice announcement on page load and auto-start listening
  useEffect(() => {
    safeCancel();

    if (!hasAnnounced.current) {
      hasAnnounced.current = true;

      const timer = setTimeout(() => {
        safeSpeak(
          'AI History Teacher. Please select your grade.',
          () => {
            setTimeout(() => {
              safeSpeak('Say Grade 10 or Grade 11.', () => {
                setTimeout(() => startListening(), 500);
              });
            }, 500);
          }
        );
      }, 500);

      return () => clearTimeout(timer);
    }

    return () => {
      safeCancel();
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.repeat) return;

      if (e.key === '1') {
        e.preventDefault();
        safeCancel();
        if (recognitionRef.current) {
          recognitionRef.current.abort();
          isListeningRef.current = false;
          setIsListening(false);
        }
        safeSpeak('Grade 10 selected.', () => {
          setTimeout(() => onSelectGrade(10), 400);
        });
      }

      if (e.key === '2') {
        e.preventDefault();
        safeCancel();
        if (recognitionRef.current) {
          recognitionRef.current.abort();
          isListeningRef.current = false;
          setIsListening(false);
        }
        safeSpeak('Grade 11 selected.', () => {
          setTimeout(() => onSelectGrade(11), 400);
        });
      }

      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        safeCancel();
        safeSpeak('Press 1 for Grade 10. Press 2 for Grade 11. Or say Grade 10 or Grade 11.');
      }

      if (e.key === 'F1') {
        e.preventDefault();
        if (isListening) {
          autoListenEnabledRef.current = false;
          if (recognitionRef.current) {
            recognitionRef.current.stop();
            isListeningRef.current = false;
            setIsListening(false);
          }
          safeCancel();
          safeSpeak('Microphone stopped.');
        } else {
          autoListenEnabledRef.current = true;
          safeCancel();
          safeSpeak('Listening for grade selection.');
          setTimeout(() => startListening(), 500);
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [onSelectGrade, isListening]);

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-orange-500 p-6">
            <BookOpen className="h-12 w-12 text-white" aria-hidden />
          </div>
        </div>

        <h1 className="text-2xl">AI History Teacher</h1>
        <p className="text-muted-foreground">
          Press 1 for Grade 10 • Press 2 for Grade 11 • Press H for help • Press F1 for microphone
        </p>
      </div>

      {/* Microphone Status Indicator */}
      {isListening && (
        <Card className="bg-blue-50 border-blue-200 p-4">
          <div className="flex items-center gap-3 justify-center">
            <Mic className="h-5 w-5 text-blue-600 animate-pulse" />
            <span className="text-sm text-blue-600 font-medium">Listening for voice command...</span>
          </div>
        </Card>
      )}

      {/* Features */}
      <Card className="p-6 space-y-4">
        <div className="flex items-center gap-3">
          <Headphones className="h-6 w-6 text-secondary" />
          <div>
            <h3 className="text-sm">AI-Generated Audio Lessons</h3>
            <p className="text-xs text-muted-foreground">
              Listen to comprehensive history lessons
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <GraduationCap className="h-6 w-6 text-secondary" />
          <div>
            <h3 className="text-sm">Curriculum Aligned</h3>
            <p className="text-xs text-muted-foreground">
              Follows Grade 10 & 11 syllabus
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Mic className="h-6 w-6 text-secondary" />
          <div>
            <h3 className="text-sm">Voice Commands</h3>
            <p className="text-xs text-muted-foreground">
              Say your grade to navigate automatically
            </p>
          </div>
        </div>
      </Card>

      {/* Grade Selection */}
      <div className="space-y-3">
        <h2 className="text-center">Select Your Grade</h2>

        {[10, 11].map((grade) => (
          <Card key={grade} className="overflow-hidden hover:shadow-lg">
            <button
              onClick={() => onSelectGrade(grade)}
              className="w-full p-6 text-left"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-primary text-2xl text-primary-foreground">
                  {grade}
                </div>

                <div className="flex-1 space-y-1">
                  <h3 className="text-xl">Grade {grade}</h3>
                  <p className="text-sm text-muted-foreground">
                    {grade === 10
                      ? 'Ancient Civilizations • World History • Cultural Studies'
                      : 'Modern History • World Wars • Contemporary Issues'}
                  </p>
                </div>
              </div>
            </button>
          </Card>
        ))}
      </div>

      {/* Info */}
      <Card className="border-secondary bg-secondary/10 p-4">
        <p className="text-center text-sm">
          Each lesson is narrated by AI and includes key topics and historical context. Say your grade or press 1 or 2 to begin.
        </p>
      </Card>
    </div>
  );
};
