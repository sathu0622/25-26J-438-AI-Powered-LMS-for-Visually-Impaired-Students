import React from 'react';
import './AudioPlayer.css';

const AudioPlayer = ({ audioUrl, subsectionTitle, duration, onPlay, onPause, isPlaying }) => {
  const audioRef = React.useRef(null);
  const [currentTime, setCurrentTime] = React.useState(0);
  const [isAudioPlaying, setIsAudioPlaying] = React.useState(false);

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isAudioPlaying) {
        audioRef.current.pause();
        setIsAudioPlaying(false);
        onPause?.();
      } else {
        audioRef.current.play();
        setIsAudioPlaying(true);
        onPlay?.();
      }
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleProgressChange = (e) => {
    if (audioRef.current) {
      audioRef.current.currentTime = e.target.value;
      setCurrentTime(e.target.value);
    }
  };

  const handleForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(audioRef.current.duration, audioRef.current.currentTime + 10);
    }
  };

  const handleBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 10);
    }
  };

  const formatTime = (time) => {
    if (!time) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  return (
    <div className="audio-player">
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onEnded={() => setIsAudioPlaying(false)}
        style={{ display: 'none' }}
      >
        {audioUrl && <source src={audioUrl} type="audio/mpeg" />}
      </audio>

      <div className="player-content">
        <div className="player-icon">🎧</div>
        
        <h2 className="player-title">{subsectionTitle}</h2>
        <p className="player-label">AI-Generated Voice Lesson</p>

        <div className="progress-container">
          <span className="time">{formatTime(currentTime)}</span>
          <input
            type="range"
            className="progress-bar"
            min="0"
            max={audioRef.current?.duration || 100}
            value={currentTime}
            onChange={handleProgressChange}
          />
          <span className="time">{duration || formatTime(audioRef.current?.duration)}</span>
        </div>

        <div className="controls">
          <button className="control-btn backward" onClick={handleBackward} title="Backward 10s">
            ⏮
          </button>
          
          <button className="control-btn play-pause" onClick={handlePlayPause} title={isAudioPlaying ? "Pause" : "Play"}>
            {isAudioPlaying ? '⏸' : '▶'}
          </button>
          
          <button className="control-btn forward" onClick={handleForward} title="Forward 10s">
            ⏭
          </button>
        </div>

        <p className="emotion-indicator">🎧 Calm Storytelling</p>
      </div>
    </div>
  );
};

export default AudioPlayer;
