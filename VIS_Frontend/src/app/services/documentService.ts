/**
 * Document Service
 * Handles all API calls related to document processing, summarization, and Q&A
 * Uses VITE_API_URL_DOCUMENT (document microservice). Set VITE_API_DOCUMENT_PREFIX if your backend
 * uses a path prefix (e.g. /api for /api/process).
 */

import { documentApi } from './api';

const DOCUMENT_PREFIX =
  (import.meta as any).env?.VITE_API_DOCUMENT_PREFIX ?? '';

export interface DocumentProcessResponse {
  document_id: string;
  summaries?: Array<{
    summary: string;
  }>;
  article_list?: ArticleInfo[];
}

export interface ArticleInfo {
  article_id: string;
  index?: number;
  heading?: string;
  subheading?: string;
  column?: string;
  word_count?: number;
}

export interface SummaryResponse {
  summary: string;
  article_heading?: string;
}

export interface QAResponse {
  answer: string;
  confidence?: number;
  article_heading?: string;
  context_preview?: string;
}

export const documentService = {
  /**
   * Upload and process a document
   * Sends the file to the backend for text extraction and initial summarization
   */
  async uploadDocument(file: File): Promise<DocumentProcessResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return documentApi.postFormData<DocumentProcessResponse>(
      `${DOCUMENT_PREFIX}/process`,
      formData
    );
  },

  /**
   * Get a summary for a specific article in the document
   */
  async summarizeArticle(
    documentId: string,
    articleId: string
  ): Promise<SummaryResponse> {
    return documentApi.postForm<SummaryResponse>(`${DOCUMENT_PREFIX}/summarize-article`, {
      document_id: documentId,
      article_id: articleId,
    });
  },

  /**
   * Ask a question about the document/article
   */
  async askQuestion(
    documentId: string,
    articleId: string,
    question: string,
    maxAnswerLen: number = 128,
    scoreThreshold: number = 0.08
  ): Promise<QAResponse> {
    return documentApi.post<QAResponse>(`${DOCUMENT_PREFIX}/ask-question`, {
      document_id: documentId,
      article_id: articleId,
      question,
      max_answer_len: maxAnswerLen,
      score_threshold: scoreThreshold,
    });
  },
};
