/**
 * User Service - API calls for user profile and quiz history
 */

import { api } from './api';

export interface QuizHistory {
  quiz_id?: string;
  session_id?: string;
  chapter_name: string;
  score: number;
  total_questions: number;
  completed_at: string;
  quiz_type: 'Generative' | 'Adaptive' | 'PastPaper' | 'FreeText' | 'TimedQuiz';
  correct_answers?: number;
  correct_count?: number;
  theta?: number;
  final_level?: string;
}

export interface ChapterStats {
  chapter_name: string;
  total_quizzes: number;
  average_score: number;
  best_score: number;
  last_attempted: string | null;
  quiz_types: {
    generative: number;
    adaptive: number;
    past_paper: number;
    freetext: number;
    timed_quiz?: number;
  };
}

export interface SavedQuiz {
  id: string;
  chapter_name: string;
  quiz_type: string;
  created_at: string;
  total_questions: number;
  attempts_count: number;
  last_score: number | null;
  can_retake: boolean;
}

export interface UserProfile {
  username: string;
  total_quizzes: number;
  generative_quizzes: number;
  adaptive_quizzes: number;
  past_paper_quizzes: number;
  freetext_quizzes: number;
  timed_quizzes: number;
  average_score: number;
  recent_activity: QuizHistory[];
  quiz_history: {
    generative: QuizHistory[];
    adaptive: QuizHistory[];
    past_paper: QuizHistory[];
    freetext: QuizHistory[];
    timed?: QuizHistory[];
  };
  chapter_stats: ChapterStats[];
  saved_quizzes: SavedQuiz[];
}

export interface UserStats {
  username: string;
  total_quizzes: number;
  generative_quizzes: number;
  adaptive_quizzes: number;
  past_paper_quizzes: number;
  timed_quizzes: number;
}

class UserService {
  /**
   * Register a new user
   */
  async register(username: string, password: string): Promise<{message: string}> {
    return api.post('/register', { username, password });
  }

  /**
   * Login user
   */
  async login(username: string, password: string): Promise<{message: string}> {
    return api.post('/login', { username, password });
  }

  /**
   * Get comprehensive user profile with quiz history
   */
  async getUserProfile(username: string): Promise<UserProfile> {
    return api.request<UserProfile>(`/profile/${username}`);
  }

  /**
   * Get quick user statistics
   */
  async getUserStats(username: string): Promise<UserStats> {
    return api.request<UserStats>(`/profile/${username}/stats`);
  }

  /**
   * Add quiz history (legacy method if needed)
   */
  async addQuizHistory(username: string, quiz_result: any): Promise<{message: string}> {
    return api.post('/add_quiz_history', { username, quiz_result });
  }
}

export const userService = new UserService();