import { api } from './api';

export interface TimedQuizQuestion {
  item_id: string;
  chapter_name: string;
  question: string;
  options: string[];
}

export interface TimedQuizStartResponse {
  session_id: string;
  duration_seconds: number;
  total_questions: number;
  questions: TimedQuizQuestion[];
}

export interface TimedQuizResultRow {
  item_id: string;
  chapter_name: string;
  question: string;
  correct: boolean;
  selected_index: number | null;
  correct_index: number;
  selected_text: string;
  correct_answer: string;
}

export interface TimedQuizEvaluateResponse {
  correct_count: number;
  total_questions: number;
  average_score: number;
  results: TimedQuizResultRow[];
}

export const timedQuizService = {
  async start(username: string): Promise<TimedQuizStartResponse> {
    return api.post<TimedQuizStartResponse>('/timed-quiz/start', { username });
  },

  /** Replay the same twenty questions from a completed timed session (no new LLM work). */
  async retake(
    username: string,
    template_session_id: string
  ): Promise<TimedQuizStartResponse> {
    return api.post<TimedQuizStartResponse>('/timed-quiz/retake', {
      username,
      template_session_id,
    });
  },

  async evaluate(
    username: string,
    sessionId: string,
    answers: { item_id: string; selected_index: number }[]
  ): Promise<TimedQuizEvaluateResponse> {
    return api.post<TimedQuizEvaluateResponse>('/timed-quiz/evaluate', {
      username,
      session_id: sessionId,
      answers,
    });
  },
};
