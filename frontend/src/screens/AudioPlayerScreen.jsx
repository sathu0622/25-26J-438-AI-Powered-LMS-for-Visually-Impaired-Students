import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Header from '../components/Header';
import AudioPlayer from '../components/AudioPlayer';
import VoiceControl from '../components/VoiceControl';
import useVoiceCommand from '../hooks/useVoiceCommand';
import { apiService } from '../services/api';
import './styles.css';

const AudioPlayerScreen = () => {
  const navigate = useNavigate();
  const { lessonId, subsectionId } = useParams();
  const [subsection, setSubsection] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generatingAudio, setGeneratingAudio] = useState(false);
  const [error, setError] = useState(null);
  const { isListening, startListening, stopListening, speak, registerCommand, isSupported } = useVoiceCommand();

  useEffect(() => {
    fetchSubsection();
  }, [lessonId, subsectionId]);

  useEffect(() => {
    if (subsection && !audioUrl && !generatingAudio) {
      generateAudio();
    }
  }, [subsection]);

  useEffect(() => {
    // Register voice commands for audio control
    registerCommand('play', () => {
      speak('Playing audio');
    });
    registerCommand('pause', () => {
      speak('Audio paused');
    });
    registerCommand('next', () => {
      speak('Skipping forward');
    });
    registerCommand('previous', () => {
      speak('Going back');
    });
    registerCommand('back', () => {
      handleBack();
    });
  }, [registerCommand, speak]);

  const fetchSubsection = async () => {
    try {
      setLoading(true);
      const data = await apiService.getSubsection(lessonId, subsectionId);
      setSubsection(data.subsection);
      setError(null);
    } catch (err) {
      setError('Failed to load subsection');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const generateAudio = async () => {
    try {
      setGeneratingAudio(true);
      const data = await apiService.generateAudio(lessonId, subsectionId);
      if (data.success) {
        setAudioUrl(data.audio_url);
        speak(`Audio for ${subsection.title} is ready. Press play to listen`);
      }
    } catch (err) {
      console.error('Error generating audio:', err);
      // Use placeholder audio
      setAudioUrl(null);
    } finally {
      setGeneratingAudio(false);
    }
  };

  const handleBack = () => {
    speak('Returning to topics');
    navigate(-1);
  };

  if (loading) {
    return (
      <div className="audio-player-screen">
        <Header title="Audio Player" onBack={handleBack} />
        <div className="content">
          <p className="loading">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="audio-player-screen">
      <Header 
        title="Audio Lesson" 
        onBack={handleBack}
      />

      <VoiceControl 
        isListening={isListening}
        onStartListening={startListening}
        onStopListening={stopListening}
        isSupported={isSupported}
      />

      <div className="content">
        {error && <p className="error">{error}</p>}
        
        {subsection && (
          <div className="player-wrapper">
            {generatingAudio && (
              <p className="generating-text">Generating audio... Please wait</p>
            )}
            
            <AudioPlayer
              audioUrl={audioUrl}
              subsectionTitle={subsection.title}
              duration={`${subsection.duration} minutes`}
            />

            <div className="subsection-info">
              <h3>{subsection.title}</h3>
              <p className="description">{subsection.description}</p>
              <div className="content-preview">
                <p className="content-text">{subsection.content}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioPlayerScreen;
