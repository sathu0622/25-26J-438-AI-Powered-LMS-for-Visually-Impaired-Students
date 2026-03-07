/**
 * API Configuration and Base Setup
 * Five microservice base URLs: voice, braille, document, history, quiz.
 * Voice is used in ttsService.ts via VITE_API_URL_VOICE.
 */

const env = (import.meta as any).env;

const VITE_API_URL_DOCUMENT =
  env?.VITE_API_URL_DOCUMENT || env?.VITE_API_URL || 'http://localhost:8000';
const VITE_API_URL_BRAILLE =
  env?.VITE_API_URL_BRAILLE || 'http://localhost:8000';
const VITE_API_URL_HISTORY =
  env?.VITE_API_URL_HISTORY || 'http://localhost:8000';
const VITE_API_URL_QUIZ = env?.VITE_API_URL_QUIZ || 'http://localhost:8000';

export const api = {
  baseURL: API_BASE_URL,

  /**
   * Generic fetch wrapper with error handling
   */
  async request<T>(
    endpoint: string,
    options: RequestInit & { method?: string } = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, options);

      if (!response.ok) {
        let errorMessage = `API Error: ${response.statusText}`;
        try {
          const errorData = await response.json();
          if (errorData?.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // Ignore JSON parse errors
        }
        throw new Error(errorMessage);
      }

    return response.json();
  },

  /**
   * POST request with form data (for file uploads)
   */
  async postFormData<T>(endpoint: string, formData: FormData): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: formData,
    });
  },

  /**
   * POST request with JSON
   */
  async post<T>(
    endpoint: string,
    data: Record<string, any>
  ): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
  },

  /**
   * POST request with URL encoded form data
   */
  async postForm<T>(
    endpoint: string,
    data: Record<string, string>
  ): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams(data),
    });
  },
};
