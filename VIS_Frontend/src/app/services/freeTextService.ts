import { api, isAbortError, AbortedRequestError } from './api';

export { isAbortError, AbortedRequestError };

export interface FreeTextQuestion {
  question: string;
  correct_answer: string;
  key_phrase: string;
}

export interface FreeTextStartResponse {
  session_id: string;
  attempt_id: string;
  chapter_name: string;
  question_index: number;
  current_question: FreeTextQuestion;
  total_questions: number;
  is_retake: boolean;
}

export interface FreeTextAnswerResponse {
  question_index: number;
  score: number;
  correct: boolean;
  feedback: string;
  correct_answer: string;
  user_answer: string;
}

export interface FreeTextNextResponse {
  question_index: number;
  current_question: FreeTextQuestion;
  total_questions: number;
  is_retake: boolean;
}

export interface FreeTextSummary {
  correct_count: number;
  total_questions: number;
  average_score: number;
}

export interface FreeTextFinishResponse {
  session_id: string;
  attempt_id: string;
  summary: FreeTextSummary;
  answers: FreeTextAnswerResponse[];
}

export interface FreeTextSessionListItem {
  session_id: string;
  chapter_name: string;
  created_at?: string;
  questions_count: number;
  attempts_count: number;
  latest_attempt?: {
    attempt_id?: string;
    summary?: FreeTextSummary;
    completed_at?: string | null;
  } | null;
}

export const freeTextService = {
  /**
   * Get available chapters for free-text quiz
   */
  async getChapters(): Promise<string[]> {
    const res = await api.request<{ chapters: string[] }>('/freetext/chapters');
    return res.chapters;
  },

  /**
   * Start a new free-text quiz session or resume existing one
   * @param signal - Optional AbortSignal for cancellation
   */
  async start(username: string, chapter_name: string, session_id?: string, signal?: AbortSignal): Promise<FreeTextStartResponse> {
    return api.post<FreeTextStartResponse>('/freetext/start', {
      username,
      chapter_name,
      session_id,
    }, signal);
  },

  /**
   * Submit an answer for evaluation
   */
  async submitAnswer(
    session_id: string,
    user_answer: string,
    username: string
  ): Promise<FreeTextAnswerResponse> {
    return api.post<FreeTextAnswerResponse>('/freetext/answer', {
      session_id,
      user_answer,
      username,
    });
  },

  /**
   * Get next question (generates new one or returns existing for retake)
   * @param signal - Optional AbortSignal for cancellation
   */
  async getNextQuestion(
    session_id: string,
    username: string,
    signal?: AbortSignal
  ): Promise<FreeTextNextResponse> {
    return api.post<FreeTextNextResponse>('/freetext/next', {
      session_id,
      user_answer: '', // Required by endpoint but not used
      username,
    }, signal);
  },

  /**
   * Finish the quiz session
   */
  async finish(session_id: string, username: string): Promise<FreeTextFinishResponse> {
    return api.post<FreeTextFinishResponse>('/freetext/finish', {
      session_id,
      username,
    });
  },

  /**
   * Get all sessions for a user
   */
  async getUserSessions(username: string): Promise<{ sessions: FreeTextSessionListItem[] }> {
    return api.request<{ sessions: FreeTextSessionListItem[] }>(`/freetext/sessions/${username}`);
  },
};
