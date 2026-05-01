/**
 * useDocumentModule Hook
 * Manages all state and logic for the document processing module
 */

import { useState, useCallback } from 'react';
import {
  documentService,
  DocumentProcessResponse,
  SyllabusMatchResponse,
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
  syllabusMatchMessage: string | null;

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
    syllabusMatchMessage: null,
    qaContextFullText: null,
    favoriteStoredPassageMissing: false,
    qaMode: 'voice',
    error: null,
  });

  /**
   * Build friendly syllabus text for students (shown only when in-syllabus).
   */
  const toSyllabusMessage = useCallback((data: SyllabusMatchResponse): string | null => {
    if (!data?.result?.in_syllabus || !data.result.match) return null;
    const topic = data.result.match.grade_topic?.trim();
    const chapter = data.result.match.chapter?.trim();
    if (topic && chapter) {
      return `This article is under syllabus topic "${topic}" (Chapter: ${chapter}).`;
    }
    if (topic) {
      return `This article is under syllabus topic "${topic}".`;
    }
    if (chapter) {
      return `This article is under syllabus chapter "${chapter}".`;
    }
    return null;
  }, []);

  /**
   * Fetch syllabus classification for the selected article.
   * If article is not in syllabus (or unavailable), nothing is shown.
   */
  const resolveSyllabusMessage = useCallback(
    async (documentId: string, articleId: string): Promise<string | null> => {
      try {
        const matchData = await documentService.matchSyllabus(documentId, articleId);
        return toSyllabusMessage(matchData);
      } catch {
        return null;
      }
    },
    [toSyllabusMessage]
  );

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
      syllabusMatchMessage: null,
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
      const syllabusMessage = await resolveSyllabusMessage(
        result.document_id,
        defaultArticleId
      );

      setState((prev) => ({
        ...prev,
        documentResult: result,
        documentSummary: initialSummary,
        selectedArticleId: defaultArticleId,
        syllabusMatchMessage: syllabusMessage,
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
  }, [resolveSyllabusMessage]);

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
        syllabusMatchMessage: null,
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
        syllabusMatchMessage: null,
      };
    });

    if (!documentId || skipSummarize) return;

    try {
      const summaryData = await documentService.summarizeArticle(
        documentId,
        articleId
      );
      const syllabusMessage = await resolveSyllabusMessage(documentId, articleId);

      setState((prev) => ({
        ...prev,
        documentSummary: summaryData.summary || '',
        syllabusMatchMessage:
          prev.selectedArticleId === articleId ? syllabusMessage : prev.syllabusMatchMessage,
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
  }, [resolveSyllabusMessage]);

  /**
   * Open a favorite using persisted summary and full text from favorites storage.
   */
  const handleOpenFavorite = useCallback((favorite: FavoriteArticle) => {
    setState((prev) => ({
      ...prev,
      error: null,
      isLoading: true,
    }));

    try {
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
        documentSummary: summaryText || 'No summary was stored for this favorite.',
        selectedArticleId: favorite.article_id,
        syllabusMatchMessage: null,
        qaContextFullText: fullText || null,
        favoriteStoredPassageMissing: Boolean(summaryText) && !fullText,
        qaMode: 'voice',
        error: null,
      });
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
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
      syllabusMatchMessage: null,
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
