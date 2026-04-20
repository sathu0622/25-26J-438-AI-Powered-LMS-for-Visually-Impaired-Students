/**
 * DocumentModule Container
 * Orchestrates the document processing workflow
 * Handles state management and passes down only necessary props to child components
 */

import { useDocumentModule } from '../../hooks/useDocumentModule';
import { DocumentUpload } from './DocumentUpload';
import { DocumentProcessing } from './DocumentProcessing';
import { DocumentSummary } from './DocumentSummary';
import { DocumentQA } from './DocumentQA';

export const DocumentModule = () => {
  const {
    screen,
    uploadedFile,
    documentResult,
    documentSummary,
    selectedArticleId,
    syllabusMatchMessage,
    qaContextFullText,
    favoriteStoredPassageMissing,
    qaMode,
    error,
    isLoading,
    handleUpload,
    handleSelectArticle,
    handleOpenFavorite,
    handleStartQA,
    handleBackToSummary,
  } = useDocumentModule();

  return (
    <>
      {/* Error Display */}
      {error && (
        <div className="mx-auto max-w-2xl p-4">
          <div
            className="rounded-lg border border-destructive bg-destructive/10 p-4"
            role="alert"
            aria-live="assertive"
          >
            <p className="font-medium">Document processing error</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Upload Screen */}
      {screen === 'upload' && (
        <DocumentUpload
          onUpload={handleUpload}
          onOpenFavorite={handleOpenFavorite}
          isOpeningFavorite={isLoading}
        />
      )}

      {/* Processing Screen */}
      {screen === 'processing' && uploadedFile && (
        <DocumentProcessing fileName={uploadedFile.name} />
      )}

      {/* Summary Screen */}
      {screen === 'summary' && (
        <DocumentSummary
          documentId={documentResult?.document_id ?? ''}
          summary={documentSummary}
          syllabusMatchMessage={syllabusMatchMessage}
          onAskQuestion={handleStartQA}
          articles={documentResult?.article_list}
          selectedArticleId={selectedArticleId}
          onSelectArticle={handleSelectArticle}
        />
      )}

      {/* Q&A Screen */}
      {screen === 'qa' && (
        <DocumentQA
          mode={qaMode}
          onBack={handleBackToSummary}
          documentId={documentResult?.document_id ?? ''}
          articleId={selectedArticleId ?? null}
          articleHeading={documentResult?.article_list?.find(
            (article: { article_id?: string }) =>
              article.article_id === selectedArticleId
          )?.heading}
          qaContextFullText={qaContextFullText}
          favoriteStoredPassageMissing={favoriteStoredPassageMissing}
        />
      )}
    </>
  );
};
