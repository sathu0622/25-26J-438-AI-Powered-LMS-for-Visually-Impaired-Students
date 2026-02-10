import './styles.css';

const VoiceControl = ({ isListening, onStartListening, onStopListening, isSupported }) => {
  return (
    <div className="voice-control">
      {isSupported && (
        <button
          className={`voice-button ${isListening ? 'listening' : ''}`}
          onClick={isListening ? onStopListening : onStartListening}
          title="Click to use voice commands"
        >
          <span className="voice-icon">🎙️</span>
          <span className="voice-text">{isListening ? 'Listening...' : 'Voice Control'}</span>
        </button>
      )}
      {!isSupported && (
        <p className="voice-warning">Voice commands not supported in your browser</p>
      )}
    </div>
  );
};

export default VoiceControl;
