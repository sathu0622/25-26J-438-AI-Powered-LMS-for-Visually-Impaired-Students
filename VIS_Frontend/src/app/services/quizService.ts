import { api } from './api';

export interface ChapterResponse {
  chapters: string[];
}

export interface GenerateQuestionResponse {
  question: string;
  correct_answer: string;
  key_phrase: string;
}

export interface EvaluateAnswerResponse {
  score: number;
  feedback: string;
  correct: boolean;
}

export const quizService = {
  // Get all chapters
  async getChapters(): Promise<string[]> {
    const res = await api.request<ChapterResponse>('/chapters');
    return res.chapters;
  },

  // Generate new question
  async generateQuestion(chapter_name: string) {
    return api.post<GenerateQuestionResponse>(
      '/generate_question',
      { chapter_name }
    );
  },

  // Evaluate answer
  async evaluateAnswer(
    user_answer: string,
    correct_answer: string,
    key_phrase: string,
    chapter_name: string
  ) {
    return api.post<EvaluateAnswerResponse>(
      '/evaluate_answer',
      {
        user_answer,
        correct_answer,
        key_phrase,
        chapter_name,
      }
    );
  },
};