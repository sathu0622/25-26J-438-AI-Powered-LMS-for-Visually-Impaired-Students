/**
 * API Configuration and Base Setup
 * Centralized API URL and utility functions for all API calls
 */

const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

/**
 * Custom error class for cancelled/aborted requests
 */
export class AbortedRequestError extends Error {
  constructor(message: string = 'Request was cancelled') {
    super(message);
    this.name = 'AbortedRequestError';
  }
}

/**
 * Check if an error is an AbortedRequestError or AbortError
 */
export function isAbortError(error: unknown): boolean {
  if (error instanceof AbortedRequestError) return true;
  if (error instanceof Error && error.name === 'AbortError') return true;
  if (error instanceof DOMException && error.name === 'AbortError') return true;
  return false;
}

export const api = {
  baseURL: API_BASE_URL,

  /**
   * Generic fetch wrapper with error handling
   * @param options.signal - Optional AbortSignal for request cancellation
  */
  async request<T>(
    endpoint: string,
    options: RequestInit & { method?: string; signal?: AbortSignal } = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    try {
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
    } catch (error) {
      // Re-throw abort errors with our custom type for consistent handling
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new AbortedRequestError('Request was cancelled');
      }
      throw error;
    }
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
   * @param signal - Optional AbortSignal for request cancellation
   */
  async post<T>(
    endpoint: string,
    data: Record<string, any>,
    signal?: AbortSignal
  ): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      signal,
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
