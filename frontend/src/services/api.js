import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Fetch available grades
  getGrades: async () => {
    try {
      const response = await api.get('/grades');
      return response.data;
    } catch (error) {
      console.error('Error fetching grades:', error);
      throw error;
    }
  },

  // ===== NEW: AI Model Endpoints =====
  
  // Get AI-generated chapters for a grade (using trained model)
  getChaptersByGrade: async (grade) => {
    try {
      const response = await api.get(`/chapters/${grade}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching chapters:', error);
      throw error;
    }
  },

  // Get a specific chapter with all topics
  getChapter: async (grade, chapterId) => {
    try {
      const response = await api.get(`/chapter/${grade}/${chapterId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching chapter:', error);
      throw error;
    }
  },

  // Get all topics for a chapter
  getChapterTopics: async (grade, chapterId) => {
    try {
      const response = await api.get(`/chapter/${grade}/${chapterId}/topics`);
      return response.data;
    } catch (error) {
      console.error('Error fetching chapter topics:', error);
      throw error;
    }
  },

  // Get a specific topic with content
  getTopic: async (grade, chapterId, topicId) => {
    try {
      const response = await api.get(`/topic/${grade}/${chapterId}/${topicId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching topic:', error);
      throw error;
    }
  },

  // Generate audio for a topic with AI content
  generateTopicAudio: async (grade, chapterId, topicId, options = {}) => {
    try {
      const response = await api.post(
        `/topic/${grade}/${chapterId}/${topicId}/generate-audio`,
        {
          emotion_intensity: options.emotionIntensity || 1.0,
          include_effects: options.includeEffects !== false,
          effects_only: options.effectsOnly || false
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error generating topic audio:', error);
      throw error;
    }
  },

  // Get generated audio file URL
  getAudioUrl: (filename) => {
    return `${API_BASE_URL}/audio/${filename}`;
  },

  // ===== LEGACY: Standard Lesson Endpoints =====

  // Fetch lessons for a specific grade
  getLessonsByGrade: async (grade) => {
    try {
      const response = await api.get(`/lessons/${grade}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching lessons:', error);
      throw error;
    }
  },

  // Fetch a specific lesson
  getLesson: async (lessonId) => {
    try {
      const response = await api.get(`/lesson/${lessonId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching lesson:', error);
      throw error;
    }
  },

  // Fetch subsections for a lesson
  getSubsections: async (lessonId) => {
    try {
      const response = await api.get(`/lesson/${lessonId}/subsections`);
      return response.data;
    } catch (error) {
      console.error('Error fetching subsections:', error);
      throw error;
    }
  },

  // Fetch a specific subsection
  getSubsection: async (lessonId, subsectionId) => {
    try {
      const response = await api.get(`/subsection/${lessonId}/${subsectionId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching subsection:', error);
      throw error;
    }
  },

  // Generate audio for a subsection
  generateAudio: async (lessonId, subsectionId) => {
    try {
      const response = await api.post(`/generate-audio/${lessonId}/${subsectionId}`);
      return response.data;
    } catch (error) {
      console.error('Error generating audio:', error);
      throw error;
    }
  },

  // Get AI-recommended chapters for a grade (legacy)
  getAIChapters: async (grade) => {
    try {
      const response = await api.get(`/ai/chapters/${grade}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching AI chapters:', error);
      throw error;
    }
  },

  // Get personalized lessons based on student profile
  getPersonalizedLessons: async (grade, profile) => {
    try {
      const response = await api.post(`/ai/personalized-lessons/${grade}`, profile);
      return response.data;
    } catch (error) {
      console.error('Error fetching personalized lessons:', error);
      throw error;
    }
  },

  // Check if a chapter is appropriate for the grade
  checkChapterDifficulty: async (grade, chapterId) => {
    try {
      const response = await api.get(`/ai/chapter-difficulty/${grade}/${chapterId}`);
      return response.data;
    } catch (error) {
      console.error('Error checking chapter difficulty:', error);
      throw error;
    }
  },
};

export default api;
