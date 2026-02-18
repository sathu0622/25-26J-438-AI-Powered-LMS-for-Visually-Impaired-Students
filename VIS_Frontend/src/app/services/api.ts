/**
 * API Configuration and Base Setup
 * Centralized API URL and utility functions for all API calls
 */

const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

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
