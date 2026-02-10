import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Header from '../components/Header';
import VoiceControl from '../components/VoiceControl';
import useVoiceCommand from '../hooks/useVoiceCommand';
import { apiService } from '../services/api';
import './styles.css';

const ChapterDetailsScreen = () => {
  const navigate = useNavigate();
  const { grade, chapterId } = useParams();
  const [chapter, setChapter] = useState(null);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { isListening, startListening, stopListening, speak, registerCommand, isSupported } = useVoiceCommand();

  useEffect(() => {
    fetchChapterAndTopics();
  }, [grade, chapterId]);

  const fetchChapterAndTopics = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch chapter details
      const chapterData = await apiService.getChapter(parseInt(grade), chapterId);
      setChapter(chapterData.chapter);

      // Fetch topics for the chapter
      const topicsData = await apiService.getChapterTopics(parseInt(grade), chapterId);
      setTopics(topicsData.topics || []);

      if (chapterData.chapter && chapterData.chapter.title) {
        speak(`Loaded ${chapterData.chapter.title}. Select a topic to learn.`);
      }
    } catch (err) {
      setError('Failed to load chapter details');
      console.error('Error fetching chapter:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTopicSelect = (topic) => {
    const topicTitle = topic.title || 'Topic';
    speak(`Opening ${topicTitle}`);
    
    // Navigate to topic details/audio player
    navigate(`/topic/${grade}/${chapterId}/${topic.id}`, {
      state: { topic, chapter }
    });
  };

  const handleBack = () => {
    speak('Returning to chapters');
    navigate(`/lessons/${grade}`);
  };

  if (loading) {
    return (
      <div className="lesson-list-screen">
        <Header title="Loading Chapter..." onBack={handleBack} />
        <div className="content">
          <p className="loading">📖 Loading chapter details...</p>
        </div>
      </div>
    );
  }

  if (!chapter) {
    return (
      <div className="lesson-list-screen">
        <Header title="Chapter Not Found" onBack={handleBack} />
        <div className="content">
          <p className="error">⚠️ {error || 'Chapter not found'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="lesson-list-screen">
      <Header 
        title={chapter.title || chapter.chapter_title} 
        onBack={handleBack} 
      />

      <VoiceControl 
        isListening={isListening}
        onStartListening={startListening}
        onStopListening={stopListening}
        isSupported={isSupported}
      />

      <div className="content">
        <div className="chapter-header">
          <p className="chapter-description">{chapter.description}</p>
          
          <div className="chapter-meta">
            <span className="meta-item">⏱️ {chapter.estimated_duration} minutes</span>
            <span className="meta-item">📚 {topics.length} topics</span>
            {chapter.ai_confidence && (
              <span className="meta-item">🤖 {(chapter.ai_confidence * 100).toFixed(0)}% confidence</span>
            )}
          </div>

          {chapter.learning_objectives && chapter.learning_objectives.length > 0 && (
            <div className="learning-objectives">
              <h3>📖 Learning Objectives</h3>
              <ul>
                {chapter.learning_objectives.map((objective, index) => (
                  <li key={index}>{objective}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="topics-section">
          <h2>📚 Topics in this Chapter</h2>
          
          <div className="topics-grid">
            {topics.map((topic, index) => (
              <div 
                key={topic.id} 
                className="topic-card"
                onClick={() => handleTopicSelect(topic)}
              >
                <div className="topic-icon">
                  {['🗣️', '📖', '🌍', '⚡'][index % 4]}
                </div>
                <div className="topic-content">
                  <h3>{topic.title}</h3>
                  <p className="topic-preview">
                    {topic.content?.substring(0, 80)}...
                  </p>
                  {topic.emotion && (
                    <span className="topic-emotion">😊 {topic.emotion}</span>
                  )}
                </div>
                <div className="topic-arrow">→</div>
              </div>
            ))}
          </div>

          {topics.length === 0 && (
            <p className="no-lessons">No topics available for this chapter</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChapterDetailsScreen;
