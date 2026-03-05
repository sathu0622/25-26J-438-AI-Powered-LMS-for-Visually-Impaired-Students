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
  quiz_type: 'Generative' | 'Adaptive';
  correct_answers?: number;
  theta?: number;
  final_level?: string;
}

export interface UserProfile {
  username: string;
  total_quizzes: number;
  generative_quizzes: number;
  adaptive_quizzes: number;
  average_score: number;
  recent_activity: QuizHistory[];
  quiz_history: {
    generative: QuizHistory[];
    adaptive: QuizHistory[];
  };
}

export interface UserStats {
  username: string;
  total_quizzes: number;
  generative_quizzes: number;
  adaptive_quizzes: number;
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