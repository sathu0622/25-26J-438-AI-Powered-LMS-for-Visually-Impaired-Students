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

export const apiBaseUrls = {
  document: VITE_API_URL_DOCUMENT,
  braille: VITE_API_URL_BRAILLE,
  history: VITE_API_URL_HISTORY,
  quiz: VITE_API_URL_QUIZ,
};

export interface ApiClient {
  baseURL: string;
  request<T>(endpoint: string, options?: RequestInit & { method?: string }): Promise<T>;
  postFormData<T>(endpoint: string, formData: FormData): Promise<T>;
  post<T>(endpoint: string, data: Record<string, any>): Promise<T>;
  postForm<T>(endpoint: string, data: Record<string, string>): Promise<T>;
}

function createApi(baseURL: string): ApiClient {
  const request = async <T>(
    endpoint: string,
    options: RequestInit & { method?: string } = {}
  ): Promise<T> => {
    const url = `${baseURL}${endpoint}`;
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
  };

  return {
    baseURL,
    request,
    async postFormData<T>(endpoint: string, formData: FormData): Promise<T> {
      return request<T>(endpoint, { method: 'POST', body: formData });
    },
    async post<T>(endpoint: string, data: Record<string, any>): Promise<T> {
      return request<T>(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
    },
    async postForm<T>(endpoint: string, data: Record<string, string>): Promise<T> {
      return request<T>(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams(data),
      });
    },
  };
}

/** Document microservice (process, summarize, Q&A) */
export const documentApi = createApi(VITE_API_URL_DOCUMENT);

/** Braille microservice (decode, evaluate) */
export const brailleApi = createApi(VITE_API_URL_BRAILLE);

/** History microservice (audio lessons) */
export const historyApi = createApi(VITE_API_URL_HISTORY);

/** Quiz microservice (voice quiz) */
export const quizApi = createApi(VITE_API_URL_QUIZ);

/** @deprecated Use documentApi for document, brailleApi for braille, etc. */
export const api = documentApi;
