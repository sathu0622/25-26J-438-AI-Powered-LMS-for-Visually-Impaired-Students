import { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Sparkles, Layers, User, FileText } from 'lucide-react';
import { useTTS } from '../../contexts/TTSContext';

interface QuizModeSelectProps {
  onSelectGenerative: () => void;
  onSelectAdaptive: () => void;
  onSelectPastPaper: () => void;
  onViewProfile: () => void;
  username?: string;
}

export const QuizModeSelect = ({ onSelectGenerative, onSelectAdaptive, onSelectPastPaper, onViewProfile, username }: QuizModeSelectProps) => {
  const { speak, cancel } = useTTS();
  const [focusedOption, setFocusedOption] = useState<number>(0); // 0: generative, 1: adaptive, 2: pastpaper, 3: profile

  useEffect(() => {
    // Cancel any previous speech and ensure we start fresh
    cancel();
    
    // Wait longer to ensure any previous speech from other components is completely stopped
    const startAnnouncement = () => {
      // Cancel again right before speaking
      cancel();
      
      const instructions = `
        Quiz Mode Selection Page loaded. Welcome ${username}!
        
        You are currently on option 1 of 4: Generative Quiz.
        
        Available options:
        1: Generative Quiz - AI creates unique questions from your study material
        2: Adaptive Quiz - Questions adjust difficulty based on your performance  
        3: Past Paper Quiz - Practice with real exam questions from previous years
        4: View Profile - See your quiz history and statistics
        
        To navigate:
        Press Down arrow to move to next option
        Press Up arrow to move to previous option
        Press Enter or Space bar to select the current option
        Press A at any time to repeat these instructions
        
        You are currently on Generative Quiz. Press Enter to start, or use arrow keys to explore other options.
      `;
      
      // Force interrupt any ongoing speech
      speak(instructions, { interrupt: true });
    };

    // Progressive delays to ensure we override any other speech
    const timer1 = setTimeout(() => {
      cancel();
    }, 100);
    
    const timer2 = setTimeout(() => {
      cancel();
      startAnnouncement();
    }, 500);
    
    // Backup announcement in case the first one doesn't work
    const timer3 = setTimeout(() => {
      if (speak) {
        cancel();
        startAnnouncement();
      }
    }, 1000);

    // Cleanup function
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      cancel();
    };
  }, [speak, cancel, username]);

  useEffect(() => {
    // Enhanced keyboard navigation
    const handleKeys = (e: KeyboardEvent) => {
      // Help instructions
      if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        cancel();
        const instructions = `
          Quiz Mode Selection Help:
          Option 1: Generative Quiz - Uses AI to create unique questions from study material
          Option 2: Adaptive Quiz - Adjusts question difficulty based on your responses
          Option 3: View Profile - Check your performance history and statistics
          
          Navigation:
          Up/Down arrows: Navigate between options
          Enter/Space: Select current option
          Tab: Move through buttons
          A: Repeat these instructions
        `;
        speak(instructions, { interrupt: true });
        return;
      }

      // Arrow key navigation with clear selection feedback
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        const newFocus = (focusedOption - 1 + 4) % 4;
        setFocusedOption(newFocus);
        announceOptionSelection(newFocus, 'previous');
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        const newFocus = (focusedOption + 1) % 4;
        setFocusedOption(newFocus);
        announceOptionSelection(newFocus, 'next');
      }

      // Enter or Space to select with confirmation
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        const optionNames = ['Generative Quiz', 'Adaptive Quiz', 'Past Paper Quiz', 'View Profile'];
        cancel();
        speak(`You selected ${optionNames[focusedOption]}. Starting now.`, { interrupt: false });
        handleSelection(focusedOption);
      }
    };

    document.addEventListener('keydown', handleKeys);
    return () => {
      document.removeEventListener('keydown', handleKeys);
      cancel();
    };
  }, [focusedOption, speak, cancel]);

  const announceOptionSelection = (optionIndex: number, direction: 'next' | 'previous' | 'focus') => {
    cancel(); // Cancel any previous announcement
    
    const optionNames = ['Generative Quiz', 'Adaptive Quiz', 'Past Paper Quiz', 'View Profile'];
    const optionNumbers = ['1', '2', '3', '4'];
    const currentOption = optionNames[optionIndex];
    const optionNumber = optionNumbers[optionIndex];
    
    const directionText = direction === 'next' ? 'Moving to next option:' : 
                         direction === 'previous' ? 'Moving to previous option:' : '';
    
    // Clear selection announcement
    const selectionAnnouncement = `
      ${directionText} Option ${optionNumber} of 4 is now selected: ${currentOption}.
    `;
    
    // Detailed descriptions
    const descriptions = [
      `You selected Generative Quiz. It uses artificial intelligence to create unique questions from your study chapters. Each quiz session will have different questions. Press Enter to start Generative Quiz.`,
      `You selected Adaptive Quiz. It automatically adjusts question difficulty based on your performance. If you answer correctly, questions get harder. If you struggle, they get easier. Press Enter to start Adaptive Quiz.`,
      `You selected Past Paper Quiz. It provides real examination questions from previous years, with year announcements for each question. Your answers are evaluated using advanced similarity matching. Press Enter to start Past Paper Quiz.`,
      `You selected View Profile. It lets you know about your complete quiz history, performance statistics, and learning progress. Press Enter to view your profile.`
    ];
    
    speak(selectionAnnouncement, { interrupt: true });
    
    // Follow with detailed description after a brief pause
    setTimeout(() => {
      speak(descriptions[optionIndex], { interrupt: false });
    }, 1000);
  };

  const handleSelection = (optionIndex: number) => {
    cancel();
    const selectionMessages = [
      'Starting Generative Quiz mode. Loading AI question generator.',
      'Starting Adaptive Quiz mode. Initializing personalized difficulty system.',
      'Starting Past Paper Quiz mode. Loading examination questions from previous years.',
      'Opening your profile. Loading performance data and quiz history.'
    ];
    
    speak(selectionMessages[optionIndex], { interrupt: true });
    
    // Small delay to let announcement play before navigation
    setTimeout(() => {
      switch(optionIndex) {
        case 0: onSelectGenerative(); break;
        case 1: onSelectAdaptive(); break;
        case 2: onSelectPastPaper(); break;
        case 3: onViewProfile(); break;
      }
    }, 1500);
  };
  return (
    <main className="mx-auto max-w-3xl p-4 space-y-6 pb-24" role="main" aria-labelledby="page-title">
      {/* Skip to main content link for screen readers */}
      <a href="#quiz-options" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-white p-2 rounded">
        Skip to quiz options
      </a>
      
      {/* Page Header */}
      <header className="text-center space-y-2" role="banner">
        <h1 id="page-title" className="text-2xl">Welcome, {username}!</h1>
        <p className="text-muted-foreground" aria-describedby="page-instructions">
          Choose a Quiz Mode or View Your Progress
        </p>
        <div className="sr-only" id="page-instructions">
          Use arrow keys to navigate, Enter to select, H for help
        </div>
      </header>

      {/* Navigation Help Text */}
      <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg" role="complementary" aria-labelledby="nav-help">
        <h2 id="nav-help" className="text-sm font-semibold mb-1">Keyboard Navigation</h2>
        <p className="text-xs text-muted-foreground">
          Up/Down arrows to navigate options, Enter/Space to select, H for help, Tab for buttons
        </p>
      </div>
      
      {/* Quiz Options */}
      <section className="grid gap-4 md:grid-cols-3" id="quiz-options" role="region" aria-labelledby="quiz-modes-heading">
        <h2 id="quiz-modes-heading" className="sr-only">Available Quiz Modes</h2>
        
        <Card 
          className={`p-6 space-y-4 ${focusedOption === 0 ? 'ring-2 ring-blue-500 bg-blue-50' : ''}`}
          role="option"
          aria-labelledby="generative-title"
          aria-describedby="generative-desc"
          tabIndex={0}
          onFocus={() => {
            setFocusedOption(0);
            announceOptionSelection(0, 'focus');
          }}
        >
          <div className="flex items-center gap-3">
            <Sparkles className="h-6 w-6 text-primary" aria-hidden="true" />
            <h3 id="generative-title" className="text-xl font-semibold">Generative Quiz</h3>
          </div>
          <p id="generative-desc" className="text-sm text-muted-foreground">
            AI-generated questions from selected chapters. Perfect for diverse practice with unique questions each time.
          </p>
          <Button 
            onClick={() => handleSelection(0)} 
            className="w-full"
            aria-label="Start Generative Quiz - AI-generated questions"
            onFocus={() => {
              cancel();
              speak('Generative Quiz start button focused. This is option 1 of 4. Press Enter or Space to begin AI-generated quiz, or use arrow keys to explore other options.', { interrupt: true });
            }}
          >
            Start Generative
          </Button>
        </Card>
        
        <Card 
          className={`p-6 space-y-4 ${focusedOption === 1 ? 'ring-2 ring-blue-500 bg-blue-50' : ''}`}
          role="option"
          aria-labelledby="adaptive-title"
          aria-describedby="adaptive-desc"
          tabIndex={0}
          onFocus={() => {
            setFocusedOption(1);
            announceOptionSelection(1, 'focus');
          }}
        >
          <div className="flex items-center gap-3">
            <Layers className="h-6 w-6 text-primary" aria-hidden="true" />
            <h3 id="adaptive-title" className="text-xl font-semibold">Adaptive Quiz</h3>
          </div>
          <p id="adaptive-desc" className="text-sm text-muted-foreground">
            Personalized difficulty that adapts to your performance. Questions become easier or harder based on your answers.
          </p>
          <Button 
            variant="outline" 
            onClick={() => handleSelection(1)} 
            className="w-full"
            aria-label="Start Adaptive Quiz - Personalized difficulty adjustment"
            onFocus={() => {
              cancel();
              speak('Adaptive Quiz start button focused. This is option 2 of 4. Press Enter or Space to begin personalized difficulty quiz, or use arrow keys to explore other options.', { interrupt: true });
            }}
          >
            Start Adaptive
          </Button>
        </Card>
        
        <Card 
          className={`p-6 space-y-4 ${focusedOption === 2 ? 'ring-2 ring-blue-500 bg-blue-50' : ''}`}
          role="option"
          aria-labelledby="pastpaper-title"
          aria-describedby="pastpaper-desc"
          tabIndex={0}
          onFocus={() => {
            setFocusedOption(2);
            announceOptionSelection(2, 'focus');
          }}
        >
          <div className="flex items-center gap-3">
            <FileText className="h-6 w-6 text-primary" aria-hidden="true" />
            <h3 id="pastpaper-title" className="text-xl font-semibold">Past Paper Quiz</h3>
          </div>
          <p id="pastpaper-desc" className="text-sm text-muted-foreground">
            Practice with real examination questions from previous years. Questions include year announcements and use advanced evaluation.
          </p>
          <Button 
            variant="outline" 
            onClick={() => handleSelection(2)} 
            className="w-full"
            aria-label="Start Past Paper Quiz - Real exam questions with year announcements"
            onFocus={() => {
              cancel();
              speak('Past Paper Quiz start button focused. This is option 3 of 4. Press Enter or Space to begin practicing with real exam questions from previous years, or use arrow keys to explore other options.', { interrupt: true });
            }}
          >
            Start Past Papers
          </Button>
        </Card>
      </section>
      
      {/* Profile Section */}
      <section role="region" aria-labelledby="profile-section-heading">
        <h2 id="profile-section-heading" className="sr-only">Profile and Performance</h2>
        <Card 
          className={`p-6 ${focusedOption === 3 ? 'ring-2 ring-blue-500 bg-blue-50' : ''}`}
          role="option"
          aria-labelledby="profile-title"
          aria-describedby="profile-desc"
          tabIndex={0}
          onFocus={() => {
            setFocusedOption(3);
            announceOptionSelection(3, 'focus');
          }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <User className="h-6 w-6 text-secondary-foreground" aria-hidden="true" />
              <div>
                <h3 id="profile-title" className="text-lg font-semibold">Your Profile</h3>
                <p id="profile-desc" className="text-sm text-muted-foreground">
                  View detailed quiz history, performance statistics, and learning progress
                </p>
              </div>
            </div>
            <Button 
              variant="secondary" 
              onClick={() => handleSelection(3)}
              aria-label="View Profile - See quiz history and performance statistics"
              onFocus={() => {
                cancel();
                speak('View Profile button focused. This is option 4 of 4. Press Enter or Space to access your performance history and statistics, or use arrow keys to explore other options.', { interrupt: true });
              }}
            >
              View Profile
            </Button>
          </div>
        </Card>
      </section>
    </main>
  );
};
