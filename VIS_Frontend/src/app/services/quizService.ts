import { api } from './api';

export interface ChapterResponse {
  chapters: string[];
}

export interface GenerateQuestionResponse {
  year: string;
  question: string;
  correct_answer: string;
  key_phrase: string;
  options?: string[];      // MCQ options (4 choices)
  correct_index?: number;  // Index of correct answer in options array
}

export interface EvaluateAnswerResponse {
  score: number;
  feedback: string;
  correct: boolean;
}

export interface QuizSetSummary {
  correct_count: number;
  total_questions: number;
  average_score: number;
}

export interface QuizSetStartResponse {
  set_id: string;
  attempt_id: string;
  chapter_name: string;
  questions: GenerateQuestionResponse[];
  total_questions: number;
}

export interface QuizSetListItem {
  set_id: string;
  chapter_name: string;
  created_at?: string;
  questions_count: number;
  latest_attempt?: {
    attempt_id?: string;
    summary?: QuizSetSummary;
    completed_at?: string | null;
  } | null;
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

  async startQuizSet(username: string, chapter_name: string, set_id?: string) {
    return api.post<QuizSetStartResponse>('/quiz_sets/start', {
      username,
      chapter_name,
      set_id,
    });
  },

  async submitQuizSetAnswer(
    set_id: string,
    attempt_id: string,
    username: string,
    question_index: number,
    user_answer: string
  ) {
    return api.post<EvaluateAnswerResponse & { question_index: number }>(
      `/quiz_sets/${set_id}/attempts/${attempt_id}/answer`,
      {
        username,
        question_index,
        user_answer,
      }
    );
  },

  async completeQuizAttempt(set_id: string, attempt_id: string, username: string) {
    return api.post<{ set_id: string; attempt_id: string; summary: QuizSetSummary }>(
      `/quiz_sets/${set_id}/attempts/${attempt_id}/complete`,
      {
        username,
      }
    );
  },

  async getUserQuizSets(username: string) {
    return api.request<{ quiz_sets: QuizSetListItem[] }>(
      `/quiz_sets/user/${username}`
    );
  },
};