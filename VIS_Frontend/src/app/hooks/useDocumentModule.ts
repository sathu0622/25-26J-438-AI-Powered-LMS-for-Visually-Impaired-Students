/**
 * useDocumentModule Hook
 * Manages all state and logic for the document processing module
 */

import { useState, useCallback } from 'react';
import {
  documentService,
  DocumentProcessResponse,
} from '../services/documentService';
import type { FavoriteArticle } from '../components/document/favoritesApi';

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

  /**
   * When non-null, Q&A uses this stored passage (e.g. Mongo `full_content` from favorites)
   * and the summary view does not call /summarize-article when switching articles.
   */
  qaContextFullText: string | null;

  /** True when a favorite was opened with a summary but no `full_content` (Q&A may not work). */
  favoriteStoredPassageMissing: boolean;

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
    qaContextFullText: null,
    favoriteStoredPassageMissing: false,
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
      qaContextFullText: null,
      favoriteStoredPassageMissing: false,
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
        qaContextFullText: null,
        favoriteStoredPassageMissing: false,
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
    let documentId: string | undefined;

    setState((prev) => {
      documentId = prev.documentResult?.document_id;
      return {
        ...prev,
        selectedArticleId: articleId,
      };
    });

    if (!documentId) return;
    let skipSummarize = false;

    setState((prev) => {
      documentId = prev.documentResult?.document_id;
      skipSummarize =
        typeof prev.qaContextFullText === 'string' &&
        prev.qaContextFullText.length > 0;
      return {
        ...prev,
        selectedArticleId: articleId,
      };
    });

    if (!documentId || skipSummarize) return;

    try {
      const summaryData = await documentService.summarizeArticle(
        documentId,
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
  }, []);

  /**
   * Open a favorite when the document is still available on the document server (in-memory session).
   */
  const handleOpenFavorite = useCallback(async (favorite: FavoriteArticle) => {
   * Open a favorite using persisted summary and full text from the favorites API (Mongo),
   * without calling /articles or /summarize-article on the document service.
   */
  const handleOpenFavorite = useCallback((favorite: FavoriteArticle) => {
    setState((prev) => ({
      ...prev,
      error: null,
      isLoading: true,
    }));

    try {
      const listData = await documentService.getArticlesList(
        favorite.document_id
      );
      const articleList = documentService.articlesResponseToArticleList(listData);

      const stillThere = articleList.some(
        (a) => a.article_id === favorite.article_id
      );
      if (!stillThere) {
        throw new Error(
          'This article is no longer available for that document on the server. Upload and process the file again, then save the favorite again.'
        );
      }

      const summaryData = await documentService.summarizeArticle(
        favorite.document_id,
        favorite.article_id
      );

      const documentResult: DocumentProcessResponse = {
        document_id: listData.document_id,
        article_list: articleList,
        summaries: [{ summary: summaryData.summary || '' }],
      const summaryText = (favorite.summary ?? '').trim();
      const fullText = (favorite.full_content ?? '').trim();

      if (!summaryText && !fullText) {
        throw new Error(
          'This favorite has no saved summary or article text. Save the article again from a processed document.'
        );
      }

      const wordCount = fullText
        ? fullText.split(/\s+/).filter(Boolean).length
        : 0;

      const documentResult: DocumentProcessResponse = {
        document_id: favorite.document_id,
        article_list: [
          {
            index: 1,
            article_id: favorite.article_id,
            heading: favorite.heading || 'Saved article',
            subheading: favorite.subheading || '',
            column: 'favorite',
            word_count: wordCount,
          },
        ],
        summaries: [{ summary: summaryText }],
      };

      setState({
        screen: 'summary',
        uploadedFile: null,
        isLoading: false,
        documentResult,
        documentSummary: summaryData.summary || '',
        selectedArticleId: favorite.article_id,
        documentSummary: summaryText || 'No summary was stored for this favorite.',
        selectedArticleId: favorite.article_id,
        qaContextFullText: fullText || null,
        favoriteStoredPassageMissing: Boolean(summaryText) && !fullText,
        qaMode: 'voice',
        error: null,
      });
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Could not open this favorite. The document may have expired on the server; upload it again.';
          : 'Could not open this favorite.';

      setState((prev) => ({
        ...prev,
        error: message,
        isLoading: false,
        screen: 'upload',
      }));
    }
  }, []);

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
      qaContextFullText: null,
      favoriteStoredPassageMissing: false,
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
    handleOpenFavorite,
    handleStartQA,
    handleBackToSummary,
    reset,
    clearError,
  };
};
