import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Header from '../components/Header';
import VoiceControl from '../components/VoiceControl';
import useVoiceCommand from '../hooks/useVoiceCommand';
import { apiService } from '../services/api';
import './AudioPlayer.css';

const TopicAudioPlayerScreen = () => {
  const navigate = useNavigate();
  const { grade, chapterId, topicId } = useParams();
  const [topic, setTopic] = useState(null);
  const [chapter, setChapter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [emotionIntensity, setEmotionIntensity] = useState(1.0);
  const [includeEffects, setIncludeEffects] = useState(true);
  const [effectsOnly, setEffectsOnly] = useState(false);
  const [generatingAudio, setGeneratingAudio] = useState(false);

  const audioRef = useRef(null);
  const { isListening, startListening, stopListening, speak, registerCommand, isSupported } = useVoiceCommand();

  useEffect(() => {
    fetchTopicAndChapter();
  }, [grade, chapterId, topicId]);

  useEffect(() => {
    // Register voice commands for audio player
    if (isSupported) {
      registerCommand('play', () => {
        if (audioRef.current) {
          audioRef.current.play();
          setPlaying(true);
          speak('Playing audio');
        }
      });

      registerCommand('pause', () => {
        if (audioRef.current) {
          audioRef.current.pause();
          setPlaying(false);
          speak('Audio paused');
        }
      });

      registerCommand('next', () => {
        if (audioRef.current) {
          audioRef.current.currentTime = Math.min(
            audioRef.current.currentTime + 10,
            duration
          );
        }
      });

      registerCommand('previous', () => {
        if (audioRef.current) {
          audioRef.current.currentTime = Math.max(
            audioRef.current.currentTime - 10,
            0
          );
        }
      });

      registerCommand('back', () => {
        handleBack();
      });
    }
  }, [isSupported, registerCommand, duration]);

  const fetchTopicAndChapter = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch topic details
      const topicData = await apiService.getTopic(parseInt(grade), chapterId, topicId);
      setTopic(topicData.topic);

      // Fetch chapter details
      const chapterData = await apiService.getChapter(parseInt(grade), chapterId);
      setChapter(chapterData.chapter);

      speak(`Loading ${topicData.topic.title}`);
      
    } catch (err) {
      setError('Failed to load topic');
      console.error('Error fetching topic:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateAudio = async () => {
    try {
      setGeneratingAudio(true);
      speak('Generating audio with emotional tone and sound effects. Please wait...');
      
      const audioData = await apiService.generateTopicAudio(
        parseInt(grade),
        chapterId,
        topicId,
        {
          emotionIntensity,
          includeEffects: includeEffects && !effectsOnly,
          effectsOnly
        }
      );

      if (audioData.success && audioData.audio_file) {
        const fullAudioUrl = apiService.getAudioUrl(audioData.audio_file);
        setAudioUrl(fullAudioUrl);
        speak(`Audio ready. Press play to listen with ${audioData.emotion} emotion.`);
      } else {
        speak('Could not generate audio. Please try again.');
      }
    } catch (err) {
      console.error('Error generating audio:', err);
      speak('Audio generation failed. The content is displayed below.');
      setError('Failed to generate audio');
    } finally {
      setGeneratingAudio(false);
    }
  };

  const handlePlayPause = () => {
    if (!audioRef.current) return;

    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
      speak('Paused');
    } else {
      audioRef.current.play();
      setPlaying(true);
      speak('Playing');
    }
  };

  const handleForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(
        audioRef.current.currentTime + 10,
        duration
      );
      speak('Skipped forward 10 seconds');
    }
  };

  const handleBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(
        audioRef.current.currentTime - 10,
        0
      );
      speak('Skipped backward 10 seconds');
    }
  };

  const handleProgressChange = (e) => {
    const newTime = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleBack = () => {
    speak('Returning to chapter topics');
    navigate(`/chapter/${grade}/${chapterId}`);
  };

  const formatTime = (time) => {
    if (!time || isNaN(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (loading) {
    return (
      <div className="audio-player-screen">
        <Header title="Loading Topic..." onBack={handleBack} />
        <div className="content">
          <p className="loading">📖 Loading topic content...</p>
        </div>
      </div>
    );
  }

  if (!topic) {
    return (
      <div className="audio-player-screen">
        <Header title="Topic Not Found" onBack={handleBack} />
        <div className="content">
          <p className="error">⚠️ {error || 'Topic not found'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="audio-player-screen">
      <Header 
        title={topic.title} 
        subtitle={chapter?.title}
        onBack={handleBack} 
      />

      <VoiceControl 
        isListening={isListening}
        onStartListening={startListening}
        onStopListening={stopListening}
        isSupported={isSupported}
      />

      <div className="content">
        <div className="topic-header">
          <h1>{topic.title}</h1>
          <div className="topic-meta">
            {topic.emotion && <span className="emotion">😊 {topic.emotion}</span>}
            {topic.sound_effects && <span className="effects">🔊 {topic.sound_effects}</span>}
          </div>
        </div>

        {/* Audio Generation Controls */}
        <div className="audio-controls-section">
          <h2>🎵 Audio Settings</h2>
          <div className="audio-settings">
            <div className="setting-group">
              <label htmlFor="emotion-slider">
                Emotion Intensity: {emotionIntensity.toFixed(1)}
              </label>
              <input
                id="emotion-slider"
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={emotionIntensity}
                onChange={(e) => setEmotionIntensity(parseFloat(e.target.value))}
                className="emotion-slider"
              />
            </div>

            <div className="setting-group">
              <label>Sound Effects:</label>
              <div className="toggle-buttons">
                <button
                  className={`toggle-btn ${includeEffects && !effectsOnly ? 'active' : ''}`}
                  onClick={() => {
                    setIncludeEffects(true);
                    setEffectsOnly(false);
                  }}
                >
                  🎵 With Effects
                </button>
                <button
                  className={`toggle-btn ${effectsOnly ? 'active' : ''}`}
                  onClick={() => {
                    setIncludeEffects(true);
                    setEffectsOnly(true);
                  }}
                >
                  🔊 Effects Only
                </button>
                <button
                  className={`toggle-btn ${!includeEffects ? 'active' : ''}`}
                  onClick={() => {
                    setIncludeEffects(false);
                    setEffectsOnly(false);
                  }}
                >
                  📢 Voice Only
                </button>
              </div>
            </div>

            <button
              className="btn-generate"
              onClick={generateAudio}
              disabled={generatingAudio}
            >
              {generatingAudio ? '⏳ Generating...' : '🎵 Generate Audio'}
            </button>
          </div>
        </div>

        {audioUrl && (
          <div className="audio-section">
            <h2>🎧 Listen to This Topic</h2>
            <audio
              ref={audioRef}
              src={audioUrl}
              onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
              onDurationChange={() => setDuration(audioRef.current?.duration || 0)}
              onEnded={() => setPlaying(false)}
            />

            <div className="player-container">
              <div className="progress-section">
                <span className="time">{formatTime(currentTime)}</span>
                <input
                  type="range"
                  min="0"
                  max={duration || 0}
                  value={currentTime}
                  onChange={handleProgressChange}
                  className="progress-bar"
                />
                <span className="time">{formatTime(duration)}</span>
              </div>

              <div className="controls">
                <button 
                  className="control-btn backward-btn" 
                  onClick={handleBackward}
                  title="Rewind 10 seconds (say 'previous')"
                >
                  ⏮️ -10s
                </button>
                
                <button 
                  className={`control-btn play-btn ${playing ? 'playing' : ''}`}
                  onClick={handlePlayPause}
                  title="Play/Pause (say 'play' or 'pause')"
                >
                  {playing ? '⏸️ Pause' : '▶️ Play'}
                </button>
                
                <button 
                  className="control-btn forward-btn"
                  onClick={handleForward}
                  title="Forward 10 seconds (say 'next')"
                >
                  +10s ⏭️
                </button>
              </div>
            </div>

            {generatingAudio && (
              <p className="generating">🎵 Generating audio with emotional tone...</p>
            )}
          </div>
        )}

        <div className="content-section">
          <h2>📖 Topic Content</h2>
          <div className="topic-content">
            {topic.content && (
              <p>{topic.content}</p>
            )}
          </div>
        </div>

        <div className="navigation-section">
          <button 
            className="btn-secondary"
            onClick={handleBack}
          >
            ← Back to Topics
          </button>
        </div>
      </div>
    </div>
  );
};

export default TopicAudioPlayerScreen;
