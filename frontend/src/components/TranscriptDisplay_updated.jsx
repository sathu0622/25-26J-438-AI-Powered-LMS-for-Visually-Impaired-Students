import { useState, useEffect } from 'react';
import './TranscriptDisplay.css';

const TranscriptDisplay = ({ isListening, lastCommand, transcript }) => {
  const [displayText, setDisplayText] = useState('');
  const [fadeOut, setFadeOut] = useState(false);

  // Use transcript if lastCommand is not available
  const textToDisplay = lastCommand || transcript;

  useEffect(() => {
    if (textToDisplay && textToDisplay.trim()) {
      console.log('📝 Displaying:', textToDisplay);
      setDisplayText(textToDisplay);
      setFadeOut(false);
      
      // Auto-fade out after 4 seconds
      const fadeTimer = setTimeout(() => {
        setFadeOut(true);
        setTimeout(() => {
          setDisplayText('');
          setFadeOut(false);
        }, 500);
      }, 4000);
      
      return () => clearTimeout(fadeTimer);
    }
  }, [textToDisplay]);

  // Always show if listening or has text
  const shouldShow = isListening || displayText;

  if (!shouldShow) {
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
