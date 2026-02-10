import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import HomeScreen from './screens/HomeScreen';
import LessonListScreen from './screens/LessonListScreen';
import ChapterDetailsScreen from './screens/ChapterDetailsScreen';
import TopicAudioPlayerScreen from './screens/TopicAudioPlayerScreen';
import LessonSubsectionsScreen from './screens/LessonSubsectionsScreen';
import AudioPlayerScreen from './screens/AudioPlayerScreen';
import TranscriptDisplay from './components/TranscriptDisplay';
import { useVoiceCommand } from './hooks/useVoiceCommand';
import './styles/global.css';

function App() {
  const { isListening, transcript } = useVoiceCommand();

  return (
    <Router>
      <div className="app">
        <Routes>
          {/* Home */}
          <Route path="/" element={<HomeScreen />} />
          
          {/* AI Model Chapters Routes */}
          <Route path="/lessons/:grade" element={<LessonListScreen />} />
          <Route path="/chapter/:grade/:chapterId" element={<ChapterDetailsScreen />} />
          <Route path="/topic/:grade/:chapterId/:topicId" element={<TopicAudioPlayerScreen />} />
          
          {/* Legacy Routes */}
          <Route path="/lesson/:lessonId" element={<LessonSubsectionsScreen />} />
          <Route path="/subsection/:lessonId/:subsectionId" element={<AudioPlayerScreen />} />
          
          {/* Default */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <TranscriptDisplay isListening={isListening} lastCommand={transcript} />
      </div>
    </Router>
  );
}

export default App;
