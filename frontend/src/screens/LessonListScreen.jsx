import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Header from '../components/Header';
import LessonCard from '../components/LessonCard';
import VoiceControl from '../components/VoiceControl';
import useVoiceCommand from '../hooks/useVoiceCommand';
import { apiService } from '../services/api';
import './styles.css';

const LessonListScreen = () => {
  const navigate = useNavigate();
  const { grade } = useParams();
  const [chapters, setChapters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState('');
  const { isListening, startListening, stopListening, speak, registerCommand, isSupported } = useVoiceCommand();

  const chapterIcons = ['🏛️', '🏰', '🧭', '⚙️', '🔬', '🌍'];

  useEffect(() => {
    fetchChapters();
  }, [grade]);

  useEffect(() => {
    if (chapters.length > 0) {
      speak(`Grade ${grade} History chapters loaded. Select a chapter to begin learning`);
    }
  }, [chapters, speak, grade]);

  const fetchChapters = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch chapters from the AI model
      const data = await apiService.getChaptersByGrade(parseInt(grade));
      
      setChapters(data.chapters || []);
      setModelInfo(data.model_used || 'AI-Generated Curriculum');
      
    } catch (err) {
      setError('Failed to load chapters from AI model');
      console.error('Error fetching chapters:', err);
      
      // Fallback to AI chapters endpoint
      try {
        const fallbackData = await apiService.getAIChapters(parseInt(grade));
        setChapters(fallbackData.chapters || []);
        setModelInfo('AI-Recommended (Fallback)');
      } catch (fallbackErr) {
        console.error('Fallback also failed:', fallbackErr);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleChapterSelect = (chapter) => {
    const title = chapter.title || chapter.chapter_title || 'Chapter';
    speak(`Opening ${title}`);
    
    // Navigate to chapter topics/details
    navigate(`/chapter/${grade}/${chapter.id}`, { 
      state: { chapter } 
    });
  };

  const handleBack = () => {
    speak('Returning to home screen');
    navigate('/');
  };

  if (loading) {
    return (
      <div className="lesson-list-screen">
        <Header title={`Grade ${grade} History`} onBack={handleBack} />
        <div className="content">
          <p className="loading">🤖 Loading chapters from AI model...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="lesson-list-screen">
      <Header 
        title={`Grade ${grade} History`} 
        onBack={handleBack} 
      />

      <VoiceControl 
        isListening={isListening}
        onStartListening={startListening}
        onStopListening={stopListening}
        isSupported={isSupported}
      />

      <div className="content">
        {error && <p className="error">⚠️ {error}</p>}
        
        {modelInfo && chapters.length > 0 && (
          <div className="ai-info">
            <span>🤖 {modelInfo} for Grade {grade}</span>
            <p className="model-source">Source: AI_History_Teacher_System.pth</p>
          </div>
        )}
        
        <div className="lessons-grid">
          {chapters.map((chapter, index) => (
            <LessonCard
              key={chapter.id}
              lesson={{
                id: chapter.id,
                title: chapter.title || chapter.chapter_title,
                description: chapter.description,
                topics: chapter.topics,
                duration: chapter.estimated_duration,
                aiConfidence: chapter.ai_confidence
              }}
              onSelect={handleChapterSelect}
              icon={chapterIcons[index % chapterIcons.length]}
            />
          ))}
        </div>

        {chapters.length === 0 && !error && (
          <p className="no-lessons">No chapters available for Grade {grade}</p>
        )}
      </div>
    </div>
  );
};

export default LessonListScreen;
