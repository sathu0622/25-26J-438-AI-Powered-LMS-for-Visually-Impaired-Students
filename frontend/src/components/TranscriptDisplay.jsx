import { useState, useEffect } from 'react';
import './TranscriptDisplay.css';

const TranscriptDisplay = ({ isListening, lastCommand }) => {
  const [displayText, setDisplayText] = useState('');
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    if (lastCommand) {
      setDisplayText(lastCommand);
      setFadeOut(false);
      
      // Auto-fade out after 3 seconds
      const timer = setTimeout(() => {
        setFadeOut(true);
        setTimeout(() => setDisplayText(''), 500);
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [lastCommand]);

  if (!displayText && !isListening) {
    return null;
  }

  return (
    <div className={`transcript-display ${fadeOut ? 'fade-out' : ''}`}>
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
