import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import VoiceControl from '../components/VoiceControl';
import useVoiceCommand from '../hooks/useVoiceCommand';
import './styles.css';

const HomeScreen = () => {
  const navigate = useNavigate();
  const { isListening, startListening, stopListening, speak, registerCommand, isSupported } = useVoiceCommand();

  useEffect(() => {
    // Welcome message
    setTimeout(() => {
      speak('Welcome to AI History Teacher. Say Grade 10 or Grade 11 to start learning. You can also click the microphone button.');
    }, 500);

    // Register voice commands for grade selection
    registerCommand('selectGrade', (transcript) => {
      const lowerTranscript = transcript.toLowerCase();
      
      // Check for "10", "ten", "grade 10", "grade ten"
      if (lowerTranscript.includes('10') || lowerTranscript.includes('ten')) {
        speak('Navigating to Grade 10 History lessons');
        setTimeout(() => navigate('/lessons/10'), 800);
      } 
      // Check for "11", "eleven", "grade 11", "grade eleven"
      else if (lowerTranscript.includes('11') || lowerTranscript.includes('eleven')) {
        speak('Navigating to Grade 11 History lessons');
        setTimeout(() => navigate('/lessons/11'), 800);
      }
    });

    // Auto-start listening after welcome message (optional - can be removed if not desired)
    setTimeout(() => {
      if (isSupported && !isListening) {
        startListening();
      }
    }, 3000);
  }, [navigate, speak, registerCommand, isSupported, isListening, startListening]);

  const handleGradeSelect = (grade) => {
    speak(`Loading Grade ${grade} lessons`);
    navigate(`/lessons/${grade}`);
  };

  return (
    <div className="home-screen">
      <Header 
        title="AI History Teacher" 
        subtitle="Learn History with Smart Audio Lessons"
        showBackButton={false}
      />

      <VoiceControl 
        isListening={isListening}
        onStartListening={startListening}
        onStopListening={stopListening}
        isSupported={isSupported}
      />

      <div className="content">
        <div className="grade-cards">
          <button 
            className="grade-button grade-10"
            onClick={() => handleGradeSelect(10)}
          >
            <span className="grade-icon">📖</span>
            <span className="grade-text">Grade 10</span>
            <span className="grade-emoji">🎓</span>
          </button>

          <button 
            className="grade-button grade-11"
            onClick={() => handleGradeSelect(11)}
          >
            <span className="grade-icon">📚</span>
            <span className="grade-text">Grade 11</span>
            <span className="grade-emoji">🌟</span>
          </button>
        </div>

        <div className="home-info">
          <p className="info-text">
            🎤 Say "Grade 10" or "Grade 11" to begin your learning journey
            <br />
            <small style={{ opacity: 0.8 }}>Click the microphone button above to activate voice commands</small>
          </p>
        </div>
      </div>
    </div>
  );
};

export default HomeScreen;
