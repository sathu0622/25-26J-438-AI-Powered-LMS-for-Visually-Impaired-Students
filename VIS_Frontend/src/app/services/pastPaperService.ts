import { api } from './api';

export interface PastPaperChapterResponse {
  chapters: string[];
}

export interface PastPaperQuestion {
  question: string;
  correct_answer: string;
  unique_part: string;
  year: string;
  chapter: string;
}

export interface PastPaperQuestionsResponse {
  questions: PastPaperQuestion[];
}

export interface PastPaperEvaluateResponse {
  score: number;
  feedback: string;
  correct: boolean;
  similarity_score: number;
  correct_answer: string;
}

export interface PastPaperQuestionResult {
  question: string;
  user_answer: string;
  correct_answer: string;
  score: number;
  correct: boolean;
  year: string;
}

export interface SavePastPaperResultRequest {
  username: string;
  chapter_name: string;
  questions: PastPaperQuestionResult[];
  total_score: number;
  correct_count: number;
  total_questions: number;
}

export interface SavePastPaperResultResponse {
  message: string;
  session_id: string;
}

export const pastPaperService = {
  // Get all chapters from past paper data
  async getChapters(): Promise<string[]> {
    const res = await api.request<PastPaperChapterResponse>('/past-paper/chapters');
    return res.chapters;
  },

  // Get past paper questions for a specific chapter
  async getQuestions(chapter_name: string): Promise<PastPaperQuestion[]> {
    const res = await api.post<PastPaperQuestionsResponse>(
      '/past-paper/questions',
      { chapter_name }
    );
    return res.questions;
  },

  // Evaluate past paper answer using SBERT model
  async evaluateAnswer(
    user_answer: string,
    correct_answer: string,
    question: string,
    year: string
  ): Promise<PastPaperEvaluateResponse> {
    return api.post<PastPaperEvaluateResponse>(
      '/past-paper/evaluate',
      {
        user_answer,
        correct_answer,
        question,
        year,
      }
    );
  },

  // Save past paper quiz result to database
  async saveResult(
    request: SavePastPaperResultRequest
  ): Promise<SavePastPaperResultResponse> {
    return api.post<SavePastPaperResultResponse>(
      '/past-paper/save-result',
      request
    );
  },
};