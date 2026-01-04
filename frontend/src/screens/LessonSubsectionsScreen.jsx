import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Header from '../components/Header';
import SubsectionCard from '../components/SubsectionCard';
import VoiceControl from '../components/VoiceControl';
import useVoiceCommand from '../hooks/useVoiceCommand';
import { apiService } from '../services/api';
import './styles.css';

const LessonSubsectionsScreen = () => {
  const navigate = useNavigate();
  const { lessonId } = useParams();
  const [subsections, setSubsections] = useState([]);
  const [lesson, setLesson] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { isListening, startListening, stopListening, speak, registerCommand, isSupported } = useVoiceCommand();

  useEffect(() => {
    fetchSubsections();
  }, [lessonId]);

  useEffect(() => {
    if (lesson) {
      speak(`${lesson.title} has ${subsections.length} topics. Select a topic to listen`);
    }
  }, [lesson, subsections, speak]);

  const fetchSubsections = async () => {
    try {
      setLoading(true);
      const data = await apiService.getSubsections(lessonId);
      setLesson({ id: lessonId, title: data.lesson_title });
      setSubsections(data.subsections || []);
      setError(null);
    } catch (err) {
      setError('Failed to load subsections');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubsectionSelect = (subsection) => {
    speak(`Playing ${subsection.title}`);
    navigate(`/subsection/${lessonId}/${subsection.id}`);
  };

  const handleBack = () => {
    speak('Returning to lessons');
    navigate(-1);
  };

  if (loading) {
    return (
      <div className="subsections-screen">
        <Header 
          title={lesson?.title || 'Loading...'} 
          onBack={handleBack}
        />
        <div className="content">
          <p className="loading">Loading topics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="subsections-screen">
      <Header 
        title={lesson?.title || 'Lesson'} 
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
        
        <div className="subsections-list">
          {subsections.map((subsection) => (
            <SubsectionCard
              key={subsection.id}
              subsection={subsection}
              onSelect={handleSubsectionSelect}
            />
          ))}
        </div>

        {subsections.length === 0 && !error && (
          <p className="no-subsections">No topics available for this lesson</p>
        )}
      </div>
    </div>
  );
};

export default LessonSubsectionsScreen;
