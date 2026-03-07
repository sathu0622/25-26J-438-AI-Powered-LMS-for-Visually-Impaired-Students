/**
 * API Configuration and Base Setup
 * Four backends: voice (ttsService), braille, document, quiz.
 * Voice uses VITE_API_URL_VOICE in ttsService.ts.
 */

const env = (import.meta as any).env;

const VITE_API_URL_DOCUMENT =
  env?.VITE_API_URL_DOCUMENT || env?.VITE_API_URL || 'http://localhost:8000';
const VITE_API_URL_BRAILLE =
  env?.VITE_API_URL_BRAILLE || 'http://localhost:8000';
const VITE_API_URL_QUIZ = env?.VITE_API_URL_QUIZ || 'http://localhost:8000';

type ApiClient = {
  baseURL: string;
  request<T>(endpoint: string, options?: RequestInit & { method?: string }): Promise<T>;
  postFormData<T>(endpoint: string, formData: FormData): Promise<T>;
  post<T>(endpoint: string, data: Record<string, any>): Promise<T>;
  postForm<T>(endpoint: string, data: Record<string, string>): Promise<T>;
};

function createApi(baseURL: string): ApiClient {
  return {
    baseURL,

    async request<T>(
      endpoint: string,
      options: RequestInit & { method?: string } = {}
    ): Promise<T> {
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
    },

    async postFormData<T>(endpoint: string, formData: FormData): Promise<T> {
      return this.request<T>(endpoint, {
        method: 'POST',
        body: formData,
      });
    },

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
}

/** Quiz backend: user, quiz, past paper, free text, adaptive. */
export const api = createApi(VITE_API_URL_QUIZ);

/** Document microservice: process, summarize, Q&A. */
export const documentApi = createApi(VITE_API_URL_DOCUMENT);

/** Braille microservice: decode, evaluate. */
export const brailleApi = createApi(VITE_API_URL_BRAILLE);

/** Thrown when a request is aborted (e.g. AbortController). */
export class AbortedRequestError extends Error {
  constructor(message = 'Request was aborted') {
    super(message);
    this.name = 'AbortedRequestError';
  }
}

/** Returns true if the error is from an aborted request. */
export function isAbortError(err: unknown): boolean {
  if (err instanceof AbortedRequestError) return true;
  if (err instanceof DOMException && err.name === 'AbortError') return true;
  return false;
}
