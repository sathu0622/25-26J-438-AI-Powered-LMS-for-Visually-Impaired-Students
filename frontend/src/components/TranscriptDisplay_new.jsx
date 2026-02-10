import { useState, useEffect } from 'react';
import './TranscriptDisplay.css';

const TranscriptDisplay = ({ isListening, lastCommand }) => {
  const [displayText, setDisplayText] = useState('');
  const [fadeOut, setFadeOut] = useState(false);
  const [fadeTimer, setFadeTimer] = useState(null);

  useEffect(() => {
    if (lastCommand && lastCommand.trim()) {
      console.log('Displaying transcript:', lastCommand);
      setDisplayText(lastCommand);
      setFadeOut(false);
      
      // Clear any existing timer
      if (fadeTimer) clearTimeout(fadeTimer);
      
      // Auto-fade out after 4 seconds
      const timer = setTimeout(() => {
        setFadeOut(true);
        setTimeout(() => setDisplayText(''), 500);
      }, 4000);
      
      setFadeTimer(timer);
      
      return () => {
        if (timer) clearTimeout(timer);
      };
    }
  }, [lastCommand]);

  return (
    <div className={`transcript-display ${fadeOut ? 'fade-out' : ''} ${isListening || displayText ? 'show' : ''}`}>
      {isListening && !displayText && (
        <div className="transcript-listening">
          <span className="listening-dot"></span>
          Listening...
        </div>
      )}
      {displayText && (
        <div className="transcript-text">
          📝 {displayText}
        </div>
      )}
    </div>
  );
};

export default TranscriptDisplay;
