/**
 * useDocumentModule Hook
 * Manages all state and logic for the document processing module
 */

import { useState, useCallback } from 'react';
import {
  documentService,
  DocumentProcessResponse,
  ArticleInfo,
} from '../services/documentService';

export type DocumentScreen = 'upload' | 'processing' | 'summary' | 'qa';

export interface DocumentModuleState {
  // Current screen
  screen: DocumentScreen;

  // File and processing
  uploadedFile: File | null;
  isLoading: boolean;

  // Document data
  documentResult: DocumentProcessResponse | null;
  documentSummary: string;
  selectedArticleId: string | null;

  // Q&A mode
  qaMode: 'voice' | 'text';

  // Error handling
  error: string | null;
}

export const useDocumentModule = () => {
  const [state, setState] = useState<DocumentModuleState>({
    screen: 'upload',
    uploadedFile: null,
    isLoading: false,
    documentResult: null,
    documentSummary: '',
    selectedArticleId: null,
    qaMode: 'voice',
    error: null,
  });

  /**
   * Upload a document and start processing
   */
  const handleUpload = useCallback(async (file: File) => {
    setState((prev) => ({
      ...prev,
      uploadedFile: file,
      screen: 'processing',
      isLoading: true,
      error: null,
      documentResult: null,
      documentSummary: '',
      selectedArticleId: null,
    }));

    try {
      const result = await documentService.uploadDocument(file);

      // Extract initial summary
      let initialSummary = '';
      if (Array.isArray(result.summaries) && result.summaries.length > 0) {
        initialSummary = result.summaries[0]?.summary || '';
      }

      // Get default article
      const defaultArticleId =
        result.article_list?.[0]?.article_id || 'full_document';

      setState((prev) => ({
        ...prev,
        documentResult: result,
        documentSummary: initialSummary,
        selectedArticleId: defaultArticleId,
        screen: 'summary',
        isLoading: false,
      }));
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Unable to process document. Please try again.';

      setState((prev) => ({
        ...prev,
        error: message,
        screen: 'upload',
        isLoading: false,
      }));
    }
  }, []);

  /**
   * Select an article and fetch its summary
   */
  const handleSelectArticle = useCallback(async (articleId: string) => {
    setState((prev) => ({
      ...prev,
      selectedArticleId: articleId,
    }));

    if (!state.documentResult?.document_id) return;

    try {
      const summaryData = await documentService.summarizeArticle(
        state.documentResult.document_id,
        articleId
      );

      setState((prev) => ({
        ...prev,
        documentSummary: summaryData.summary || '',
      }));
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Failed to summarize selected article.';

      setState((prev) => ({
        ...prev,
        error: message,
      }));
    }
  }, [state.documentResult]);

  /**
   * Start Q&A mode
   */
  const handleStartQA = useCallback((mode: 'voice' | 'text') => {
    setState((prev) => ({
      ...prev,
      qaMode: mode,
      screen: 'qa',
    }));
  }, []);

  /**
   * Return to summary screen
   */
  const handleBackToSummary = useCallback(() => {
    setState((prev) => ({
      ...prev,
      screen: 'summary',
    }));
  }, []);

  /**
   * Reset document module state
   */
  const reset = useCallback(() => {
    setState({
      screen: 'upload',
      uploadedFile: null,
      isLoading: false,
      documentResult: null,
      documentSummary: '',
      selectedArticleId: null,
      qaMode: 'voice',
      error: null,
    });
  }, []);

  /**
   * Clear error message
   */
  const clearError = useCallback(() => {
    setState((prev) => ({
      ...prev,
      error: null,
    }));
  }, []);

  return {
    // State
    ...state,

    // Handlers
    handleUpload,
    handleSelectArticle,
    handleStartQA,
    handleBackToSummary,
    reset,
    clearError,
  };
};
