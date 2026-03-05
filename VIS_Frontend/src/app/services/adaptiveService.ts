import { api } from './api';

export interface AdaptiveItem {
  item_id: string;
  chapter_name: string;
  question: string;
  difficulty: number;
  difficulty_label: string;
  context?: string;
}

export interface AdaptiveStartResponse {
  session_id: string;
  theta: number;
  item: AdaptiveItem;
}

export interface AdaptiveAnswerResponse {
  correct: boolean;
  theta: number;
  correct_answer: string;
  probability: number;
  next_item: AdaptiveItem | null;
  done: boolean;
}

export const adaptiveService = {
  async getChapters(): Promise<string[]> {
    const res = await api.request<{ chapters: string[] }>('/adaptive/chapters');
    return res.chapters;
  },

  async start(username: string, chapter_name: string) {
    return api.post<AdaptiveStartResponse>('/adaptive/start', { username, chapter_name });
  },

  async answer(session_id: string, item_id: string, user_answer: string, username: string) {
    return api.post<AdaptiveAnswerResponse>('/adaptive/answer', {
      session_id,
      item_id,
      user_answer,
      username,
    });
  },

  async finish(session_id: string, username: string) {
    return api.post<{ status: string }>('/adaptive/finish', { session_id, username });
  },
};
