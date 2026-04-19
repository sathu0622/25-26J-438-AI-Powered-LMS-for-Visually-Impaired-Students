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

/** GET /articles/{document_id} */
export interface ArticlesListResponse {
  document_id: string;
  resource_type?: string;
  num_articles?: number;
  articles: Array<{
    index: number;
    article_id: string;
    column?: string;
    heading?: string;
    subheading?: string;
    body_preview?: string;
    word_count?: number;
    paragraph_count?: number;
  }>;
  timestamp?: string;
  supports_qa?: boolean;
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
   * List articles for a document still held in the processor (e.g. after /process or when reopening a favorite).
   */
  async getArticlesList(documentId: string): Promise<ArticlesListResponse> {
    const encoded = encodeURIComponent(documentId);
    return documentApi.get<ArticlesListResponse>(
      `${DOCUMENT_PREFIX}/articles/${encoded}`
    );
  },

  articlesResponseToArticleList(data: ArticlesListResponse): ArticleInfo[] {
    const items = data.articles ?? [];
    return items.map((a) => ({
      article_id: a.article_id,
      index: a.index,
      heading: a.heading,
      subheading: a.subheading,
      column: a.column,
      word_count: a.word_count,
    }));
  },

  /**
   * Ask a question about the document/article
   * Ask a question about the document/article.
   * When `fullContentFromStore` is set (e.g. opened from Mongo-backed favorites), it is sent as `full_content`
   * so the document service can run Q&A on stored text without requiring the document to still be in memory.
   */
  async askQuestion(
    documentId: string,
    articleId: string,
    question: string,
    maxAnswerLen: number = 128,
    scoreThreshold: number = 0.08
    scoreThreshold: number = 0.08,
    fullContentFromStore?: string
  ): Promise<QAResponse> {
    const body: Record<string, any> = {
      document_id: documentId,
      article_id: articleId,
      question,
      max_answer_len: maxAnswerLen,
      score_threshold: scoreThreshold,
    };
    if (fullContentFromStore?.trim()) {
      body.full_content = fullContentFromStore.trim();
    }
    return documentApi.post<QAResponse>(`${DOCUMENT_PREFIX}/ask-question`, body);
  },
};
