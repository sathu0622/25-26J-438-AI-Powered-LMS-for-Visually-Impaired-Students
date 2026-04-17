import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  FileText,
  Image as ImageIcon,
  Bookmark,
  RefreshCw,
  Loader2,
  BookOpen,
} from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { useTTS } from '../../contexts/TTSContext';
import {
  listFavoriteArticles,
  type FavoriteArticle,
} from './favoritesApi';

interface DocumentUploadProps {
  onUpload: (file: File) => void;
  onOpenFavorite: (favorite: FavoriteArticle) => void | Promise<void>;
  isOpeningFavorite?: boolean;
}

export const DocumentUpload = ({
  onUpload,
  onOpenFavorite,
  isOpeningFavorite = false,
}: DocumentUploadProps) => {
  const { announce, cancel, speak } = useTTS();
  const speakRef = useRef(speak);
  speakRef.current = speak;

  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [favorites, setFavorites] = useState<FavoriteArticle[]>([]);
  const [favoritesLoading, setFavoritesLoading] = useState(false);
  const [favoritesError, setFavoritesError] = useState<string | null>(null);

  const loadFavorites = useCallback(async (opts?: { announceResult?: boolean }) => {
    setFavoritesLoading(true);
    setFavoritesError(null);
    try {
      const data = await listFavoriteArticles();
      const list = Array.isArray(data.favorites) ? data.favorites : [];
      setFavorites(list);
      if (opts?.announceResult) {
        const n = list.length;
        const line =
          n === 0
            ? 'No favorite articles yet. Save articles from the summary or question pages with the S key or save button.'
            : `Loaded ${n} favorite article${n === 1 ? '' : 's'}.`;
        speakRef.current(line, { interrupt: true });
      }
    } catch (e) {
      const message =
        e instanceof Error ? e.message : 'Could not load favorite articles.';
      setFavoritesError(message);
      if (opts?.announceResult) {
        speakRef.current(message, { interrupt: true });
      }
    } finally {
      setFavoritesLoading(false);
    }
  }, []);

  useEffect(() => {
    announce(
      'Upload document. Drag and drop a file here, or choose Upload PDF or Upload Image below. Favorite articles are listed below. Press number keys 1 to 9 to open a favorite and go to its summary and questions, or use the Open summary and Q and A button on each item. Refresh favorites updates the list.'
    );
    return () => cancel();
  }, [announce, cancel]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isOpeningFavorite || favoritesLoading) return;
      const t = e.target as HTMLElement | null;
      if (t?.closest('input, textarea, select, [contenteditable="true"]')) return;

      const num = parseInt(e.key, 10);
      if (Number.isNaN(num) || num < 1 || num > 9) return;
      const idx = num - 1;
      if (idx >= favorites.length) return;
      e.preventDefault();
      const fav = favorites[idx];
      speakRef.current(
        `Opening favorite ${num}: ${fav.heading || 'article'}.`,
        { interrupt: true }
      );
      void onOpenFavorite(fav);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [
    favorites,
    favoritesLoading,
    isOpeningFavorite,
    onOpenFavorite,
  ]);

  useEffect(() => {
    void loadFavorites();
  }, [loadFavorites]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      onUpload(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  const handleButtonClick = (type: 'pdf' | 'image') => {
    if (fileInputRef.current) {
      fileInputRef.current.accept = type === 'pdf' ? '.pdf' : 'image/*';
      fileInputRef.current.click();
    }
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-2 text-center">
        <h1 className="text-2xl">Upload Document</h1>
        <p className="text-muted-foreground">
          Upload a PDF or image to hear and read its summary
        </p>
      </div>

      {/* Upload Area */}
      <Card
        className={`border-2 border-dashed p-8 transition-all ${
          isDragging ? 'border-primary bg-primary/5' : 'border-border'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="space-y-6 text-center">
          <div className="flex justify-center">
            <Upload
              className="h-16 w-16 text-muted-foreground"
              aria-hidden="true"
            />
          </div>
          <div className="space-y-2">
            <p>Drag and drop your file here</p>
            <p className="text-sm text-muted-foreground">or choose an option below</p>
          </div>
        </div>
      </Card>

      {/* Favorite articles */}
      <Card className="p-4 space-y-4" aria-labelledby="favorites-heading">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-primary/10 p-2">
              <Bookmark className="h-6 w-6 text-primary" aria-hidden="true" />
            </div>
            <div>
              <h2 id="favorites-heading" className="text-lg">
                Favorite articles
              </h2>
              <p className="text-sm text-muted-foreground">
                Open a favorite to go to summary and Q and A if that document is still loaded on the server. Otherwise upload the file again. Keys 1 to 9 open the matching favorite in order.
              </p>
            </div>
          </div>
          <Button
            type="button"
            variant="secondary"
            size="lg"
            className="min-h-[52px] gap-2 shrink-0 w-full sm:w-auto"
            disabled={favoritesLoading || isOpeningFavorite}
            onClick={() => void loadFavorites({ announceResult: true })}
            aria-label="Refresh favorite articles list"
          >
            {favoritesLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
            ) : (
              <RefreshCw className="h-5 w-5" aria-hidden="true" />
            )}
            {favoritesLoading ? 'Loading…' : 'Refresh favorites'}
          </Button>
        </div>

        {favoritesError && (
          <div
            className="rounded-md border border-destructive bg-destructive/10 px-3 py-2 text-sm"
            role="alert"
          >
            {favoritesError}
          </div>
        )}

        {isOpeningFavorite && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
            <Loader2 className="h-4 w-4 animate-spin shrink-0" aria-hidden="true" />
            Opening favorite and loading summary…
          </p>
        )}

        <div role="region" aria-live="polite" aria-relevant="additions text">
          {favoritesLoading && favorites.length === 0 && !favoritesError ? (
            <p className="text-sm text-muted-foreground">Loading favorites…</p>
          ) : favorites.length === 0 && !favoritesError ? (
            <p className="text-sm text-muted-foreground">
              No favorites yet. After you process a document, press S or use Save to favorites on the summary or question page.
            </p>
          ) : (
            <ol className="space-y-3 list-decimal pl-5 marker:text-muted-foreground">
              {favorites.map((fav, index) => {
                const n = index + 1;
                const shortcutHint = n <= 9 ? ` Shortcut: ${n}.` : '';
                return (
                  <li key={`${fav.document_id}:${fav.article_id}:${index}`}>
                    <div className="rounded-lg border border-border bg-card p-3 text-left space-y-3">
                      <p className="font-medium leading-snug">
                        {fav.heading || 'Untitled article'}
                      </p>
                      {fav.subheading ? (
                        <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                          {fav.subheading}
                        </p>
                      ) : null}
                      {fav.body_preview ? (
                        <p className="mt-2 text-sm leading-relaxed line-clamp-3">
                          {fav.body_preview}
                        </p>
                      ) : null}
                      <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                        {fav.resource_type ? (
                          <div>
                            <dt className="inline font-medium text-foreground/80">
                              Type:{' '}
                            </dt>
                            <dd className="inline">{fav.resource_type}</dd>
                          </div>
                        ) : null}
                        <div>
                          <dt className="sr-only">Article</dt>
                          <dd>Article ID: {fav.article_id}</dd>
                        </div>
                      </dl>
                      <Button
                        type="button"
                        variant="default"
                        size="lg"
                        className="w-full min-h-[52px] gap-2"
                        disabled={isOpeningFavorite}
                        onClick={() => void onOpenFavorite(fav)}
                        aria-label={`Open summary and questions for ${fav.heading || 'article'}.${shortcutHint}`}
                      >
                        <BookOpen className="h-5 w-5" aria-hidden="true" />
                        Open summary and Q and A
                        {n <= 9 ? (
                          <span className="text-xs opacity-90">({n})</span>
                        ) : null}
                      </Button>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      </Card>

      {/* Upload Buttons */}
      <div className="grid gap-4 sm:grid-cols-2">
        <Button
          onClick={() => handleButtonClick('pdf')}
          size="lg"
          className="min-h-[80px] flex-col gap-2"
          aria-label="Upload PDF document"
        >
          <FileText className="h-8 w-8" aria-hidden="true" />
          <span>Upload PDF</span>
        </Button>
        <Button
          onClick={() => handleButtonClick('image')}
          size="lg"
          className="min-h-[80px] flex-col gap-2"
          aria-label="Upload image"
        >
          <ImageIcon className="h-8 w-8" aria-hidden="true" />
          <span>Upload Image</span>
        </Button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileSelect}
        aria-label="File input"
      />

      {/* Guidelines */}
      <Card className="bg-muted/50 p-4">
        <h3 className="mb-2 text-sm">Supported Formats:</h3>
        <ul className="space-y-1 text-xs text-muted-foreground">
          <li>• PDF documents</li>
          <li>• Images (JPG, PNG, JPEG)</li>
          <li>• Maximum file size: 10MB</li>
        </ul>
      </Card>
    </div>
  );
};
