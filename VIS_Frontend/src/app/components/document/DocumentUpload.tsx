import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  FileText,
  Image as ImageIcon,
  Bookmark,
  RefreshCw,
  Loader2,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Volume2,
} from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { useTTS } from '../../contexts/TTSContext';
import {
  listFavoriteArticles,
  type FavoriteArticle,
} from './favoritesApi';

type HomeTab = 'upload' | 'favorites';

/** Keys 1–9 map to one slot per page; paginate when there are more favorites. */
const FAVORITES_PAGE_SIZE = 9;

/** One spoken block listing titles on the current page (keys 1–9) for screen-reader / TTS users. */
function buildFavoriteTitlesUtterance(
  items: FavoriteArticle[],
  pageIndex: number,
  totalCount: number
): string {
  if (items.length === 0 || totalCount === 0) return '';
  const pageStart = pageIndex * FAVORITES_PAGE_SIZE + 1;
  const pageEnd = pageStart + items.length - 1;
  const parts: string[] = [
    `Reading titles for items ${pageStart} through ${pageEnd} of ${totalCount}.`,
  ];
  items.forEach((fav, i) => {
    const keySlot = i + 1;
    const globalNum = pageIndex * FAVORITES_PAGE_SIZE + i + 1;
    const title = (fav.heading || 'Untitled article').replace(/\s+/g, ' ').trim();
    let sub = (fav.subheading || '').replace(/\s+/g, ' ').trim();
    if (sub.length > 100) sub = `${sub.slice(0, 100)}…`;
    if (sub) {
      parts.push(
        `Number ${globalNum}, press ${keySlot}: ${title}. Subheading: ${sub}.`
      );
    } else {
      parts.push(`Number ${globalNum}, press ${keySlot}: ${title}.`);
    }
  });
  parts.push('End of list for this page.');
  return parts.join(' ');
}

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

  const [activeTab, setActiveTab] = useState<HomeTab>('upload');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [favorites, setFavorites] = useState<FavoriteArticle[]>([]);
  const [favoritesLoading, setFavoritesLoading] = useState(false);
  const [favoritesError, setFavoritesError] = useState<string | null>(null);
  const [favoritesPageIndex, setFavoritesPageIndex] = useState(0);

  const favoritesRef = useRef<FavoriteArticle[]>([]);
  favoritesRef.current = favorites;
  const favoritesPageIndexRef = useRef(0);
  favoritesPageIndexRef.current = favoritesPageIndex;
  const favoritesLoadingRef = useRef(false);
  favoritesLoadingRef.current = favoritesLoading;

  const loadFavorites = useCallback(async (opts?: { announceResult?: boolean }) => {
    setFavoritesLoading(true);
    setFavoritesError(null);
    try {
      const data = await listFavoriteArticles();
      const list = Array.isArray(data.favorites) ? data.favorites : [];
      setFavorites(list);
      if (opts?.announceResult) {
        const n = list.length;
        if (n === 0) {
          speakRef.current(
            'No favorite articles yet. Save articles from the summary or question pages with the S key or save button.',
            { interrupt: true }
          );
        } else {
          const line =
            n > FAVORITES_PAGE_SIZE
              ? `Loaded ${n} favorites. Reading titles for items 1 through ${Math.min(FAVORITES_PAGE_SIZE, n)}.`
              : `Loaded ${n} favorite article${n === 1 ? '' : 's'}. Reading titles.`;
          const slice = list.slice(0, FAVORITES_PAGE_SIZE);
          const titles = buildFavoriteTitlesUtterance(slice, 0, n);
          speakRef.current(line, {
            interrupt: true,
            onEnd: () => {
              if (titles) void speakRef.current(titles, { interrupt: false });
            },
          });
        }
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
    const maxPage = Math.max(
      0,
      Math.ceil(favorites.length / FAVORITES_PAGE_SIZE) - 1
    );
    setFavoritesPageIndex((p) => Math.min(p, maxPage));
  }, [favorites]);

  const goFavoritesPage = useCallback((delta: number) => {
    setFavoritesPageIndex((p) => {
      const list = favoritesRef.current;
      const maxPage = Math.max(
        0,
        Math.ceil(list.length / FAVORITES_PAGE_SIZE) - 1
      );
      const next = Math.min(maxPage, Math.max(0, p + delta));
      if (next !== p && list.length > 0) {
        const from = next * FAVORITES_PAGE_SIZE + 1;
        const to = Math.min(
          (next + 1) * FAVORITES_PAGE_SIZE,
          list.length
        );
        const slice = list.slice(
          next * FAVORITES_PAGE_SIZE,
          next * FAVORITES_PAGE_SIZE + FAVORITES_PAGE_SIZE
        );
        const titles = buildFavoriteTitlesUtterance(slice, next, list.length);
        speakRef.current(
          `Favorites page ${next + 1} of ${maxPage + 1}. Showing items ${from} to ${to}.`,
          {
            interrupt: true,
            onEnd: () => {
              if (titles) void speakRef.current(titles, { interrupt: false });
            },
          }
        );
      }
      return next;
    });
  }, []);

  const speakFavoritesIntroAndTitles = useCallback(() => {
    const list = favoritesRef.current;
    const n = list.length;
    if (n === 0) {
      void speak('Saved favorites tab. No favorites yet.', { interrupt: true });
      return;
    }
    if (favoritesLoadingRef.current) {
      void speak('Favorites are loading. Please wait.', { interrupt: true });
      return;
    }
    const pageIdx = favoritesPageIndexRef.current;
    const maxPage = Math.max(0, Math.ceil(n / FAVORITES_PAGE_SIZE) - 1);
    const intro =
      n <= FAVORITES_PAGE_SIZE
        ? `Saved favorites tab. ${n} article${n === 1 ? '' : 's'}. Press 1 through ${n} to open one. Now reading their titles.`
        : `Saved favorites tab. ${n} articles, page ${pageIdx + 1} of ${maxPage + 1}. Press 1 through 9 for this page. Page Up or Page Down to change pages. Now reading titles on this page.`;
    const slice = list.slice(
      pageIdx * FAVORITES_PAGE_SIZE,
      pageIdx * FAVORITES_PAGE_SIZE + FAVORITES_PAGE_SIZE
    );
    const titles = buildFavoriteTitlesUtterance(slice, pageIdx, n);
    void speak(intro, {
      interrupt: true,
      onEnd: () => {
        if (titles) void speak(titles, { interrupt: false });
      },
    });
  }, [speak]);

  const speakCurrentPageTitlesOnly = useCallback(
    (pageIndex: number) => {
      const list = favoritesRef.current;
      const n = list.length;
      if (n === 0) return;
      const slice = list.slice(
        pageIndex * FAVORITES_PAGE_SIZE,
        pageIndex * FAVORITES_PAGE_SIZE + FAVORITES_PAGE_SIZE
      );
      const titles = buildFavoriteTitlesUtterance(slice, pageIndex, n);
      if (titles) void speak(titles, { interrupt: true });
    },
    [speak]
  );

  useEffect(() => {
    announce(
      'Document home. Use the Upload file or Saved favorites tabs, or press U for upload and F for favorites. On the favorites tab, article titles are read aloud when you open the tab or change pages. Press R to hear titles again. Press 1 through 9 to open an item on the current page. If you have more than nine favorites, use Page Up and Page Down, bracket keys, or the previous and next buttons to change pages. Upload a PDF or image from the upload tab.'
    );
    return () => cancel();
  }, [announce, cancel]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isOpeningFavorite || favoritesLoading) return;
      const t = e.target as HTMLElement | null;
      if (t?.closest('input, textarea, select, [contenteditable="true"]')) return;

      if (e.key === 'u' || e.key === 'U') {
        e.preventDefault();
        setActiveTab('upload');
        speakRef.current('Upload file tab.', { interrupt: true });
        return;
      }
      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        setActiveTab('favorites');
        speakFavoritesIntroAndTitles();
        return;
      }

      if (activeTab !== 'favorites') return;

      if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        speakCurrentPageTitlesOnly(favoritesPageIndex);
        return;
      }

      const totalPages = Math.max(
        1,
        Math.ceil(favorites.length / FAVORITES_PAGE_SIZE)
      );

      if (e.key === 'PageDown') {
        if (favoritesPageIndex < totalPages - 1) {
          e.preventDefault();
          goFavoritesPage(1);
        }
        return;
      }
      if (e.key === 'PageUp') {
        if (favoritesPageIndex > 0) {
          e.preventDefault();
          goFavoritesPage(-1);
        }
        return;
      }
      if (e.key === '[') {
        if (favoritesPageIndex > 0) {
          e.preventDefault();
          goFavoritesPage(-1);
        }
        return;
      }
      if (e.key === ']') {
        if (favoritesPageIndex < totalPages - 1) {
          e.preventDefault();
          goFavoritesPage(1);
        }
        return;
      }

      const num = parseInt(e.key, 10);
      if (Number.isNaN(num) || num < 1 || num > 9) return;
      const globalIdx = favoritesPageIndex * FAVORITES_PAGE_SIZE + num - 1;
      if (globalIdx >= favorites.length) return;
      e.preventDefault();
      const fav = favorites[globalIdx];
      const displayNum = globalIdx + 1;
      speakRef.current(
        `Opening favorite ${displayNum}: ${fav.heading || 'article'}.`,
        { interrupt: true }
      );
      void onOpenFavorite(fav);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [
    activeTab,
    favorites,
    favoritesPageIndex,
    favoritesLoading,
    isOpeningFavorite,
    onOpenFavorite,
    goFavoritesPage,
    speakFavoritesIntroAndTitles,
    speakCurrentPageTitlesOnly,
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

  const favCount = favorites.length;
  const favoritesTotalPages = Math.max(
    1,
    Math.ceil(favCount / FAVORITES_PAGE_SIZE)
  );
  const favStart = favoritesPageIndex * FAVORITES_PAGE_SIZE;
  const favoritesOnPage = favorites.slice(
    favStart,
    favStart + FAVORITES_PAGE_SIZE
  );
  const favShowingFrom = favCount === 0 ? 0 : favStart + 1;
  const favShowingTo = Math.min(favStart + FAVORITES_PAGE_SIZE, favCount);

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
      <div className="space-y-2 text-center">
        <h1 className="text-2xl">Documents</h1>
        <p className="text-muted-foreground">
          Upload a new file or open something you saved earlier
        </p>
        <p className="text-xs text-muted-foreground" aria-hidden="true">
          Shortcuts: U upload · F favorites and hear titles · R hear titles again on favorites
          · 1–9 open · Page Up / Down or [ / ] pages
        </p>
      </div>

      <Tabs
        value={activeTab}
        onValueChange={(v) => {
          const tab = v as HomeTab;
          setActiveTab(tab);
          if (tab === 'favorites') {
            speakFavoritesIntroAndTitles();
          } else {
            void speak('Upload file tab.', { interrupt: true });
          }
        }}
        className="w-full gap-4"
      >
        <TabsList
          className="grid h-auto w-full grid-cols-2 gap-1 p-1 sm:max-w-md sm:mx-auto"
          aria-label="Document home sections"
        >
          <TabsTrigger
            value="upload"
            className="min-h-[52px] gap-2 px-3 py-3 text-sm sm:text-base"
            aria-label="Upload file tab. Shortcut: U."
          >
            <Upload className="h-5 w-5 shrink-0 opacity-80" aria-hidden="true" />
            Upload file
          </TabsTrigger>
          <TabsTrigger
            value="favorites"
            className="min-h-[52px] gap-2 px-3 py-3 text-sm sm:text-base"
            aria-label={`Saved favorites tab. ${favCount} items. Shortcut F opens and reads article titles. Shortcut R reads titles again.`}
          >
            <Bookmark className="h-5 w-5 shrink-0 opacity-80" aria-hidden="true" />
            <span className="inline-flex items-center gap-1.5">
              Favorites
              {favCount > 0 ? (
                <span className="rounded-full bg-primary/15 px-2 py-0.5 text-xs font-semibold tabular-nums text-foreground">
                  {favCount}
                </span>
              ) : null}
            </span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="mt-4 space-y-6 outline-none">
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

          <Card className="bg-muted/50 p-4">
            <h3 className="mb-2 text-sm font-medium">Supported formats</h3>
            <ul className="space-y-1 text-xs text-muted-foreground">
              <li>• PDF documents</li>
              <li>• Images (JPG, PNG, JPEG)</li>
              <li>• Maximum file size: 10MB</li>
            </ul>
          </Card>
        </TabsContent>

        <TabsContent value="favorites" className="mt-4 space-y-4 outline-none">
          <Card className="p-4 space-y-4" aria-labelledby="favorites-heading">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex items-start gap-3">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Bookmark className="h-6 w-6 text-primary" aria-hidden="true" />
                </div>
                <div>
                  <h2 id="favorites-heading" className="text-lg">
                    Saved favorites
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    Titles are read aloud when you open this tab or change pages. Press R
                    or use Listen to titles to hear them again. Press 1 through 9 for the
                    current page. Use Previous and Next or Page Up and Page Down when you
                    have more than nine favorites.
                  </p>
                </div>
              </div>
              <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-stretch">
                <Button
                  type="button"
                  variant="outline"
                  size="lg"
                  className="min-h-[52px] gap-2 shrink-0 w-full sm:w-auto"
                  disabled={
                    favoritesLoading ||
                    isOpeningFavorite ||
                    favorites.length === 0
                  }
                  onClick={() => speakCurrentPageTitlesOnly(favoritesPageIndex)}
                  aria-label="Listen to article titles on this page. Shortcut: R on favorites tab."
                >
                  <Volume2 className="h-5 w-5" aria-hidden="true" />
                  Listen to titles
                </Button>
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
                {favoritesLoading ? 'Loading…' : 'Refresh list'}
              </Button>
              </div>
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
                Opening favorite from saved data…
              </p>
            )}

            {favCount > FAVORITES_PAGE_SIZE && !favoritesLoading && !favoritesError ? (
              <div
                className="flex flex-col gap-3 rounded-lg border border-border bg-muted/30 p-3 sm:flex-row sm:items-center sm:justify-between"
                role="navigation"
                aria-label="Favorite pages"
              >
                <p className="text-sm text-center sm:text-left" aria-live="polite">
                  <span className="font-medium tabular-nums">
                    Page {favoritesPageIndex + 1} of {favoritesTotalPages}
                  </span>
                  <span className="text-muted-foreground">
                    {' '}
                    · Items {favShowingFrom}–{favShowingTo} of {favCount}
                  </span>
                </p>
                <div className="flex gap-2 justify-center sm:justify-end">
                  <Button
                    type="button"
                    variant="outline"
                    size="lg"
                    className="min-h-[48px] min-w-[44px] gap-1 px-3"
                    disabled={favoritesPageIndex <= 0 || isOpeningFavorite}
                    onClick={() => goFavoritesPage(-1)}
                    aria-label="Previous favorites page. Shortcut: Page Up."
                  >
                    <ChevronLeft className="h-5 w-5" aria-hidden="true" />
                    Previous
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="lg"
                    className="min-h-[48px] min-w-[44px] gap-1 px-3"
                    disabled={
                      favoritesPageIndex >= favoritesTotalPages - 1 ||
                      isOpeningFavorite
                    }
                    onClick={() => goFavoritesPage(1)}
                    aria-label="Next favorites page. Shortcut: Page Down."
                  >
                    Next
                    <ChevronRight className="h-5 w-5" aria-hidden="true" />
                  </Button>
                </div>
              </div>
            ) : null}

            <div role="region" aria-live="polite" aria-relevant="additions text">
              {favoritesLoading && favorites.length === 0 && !favoritesError ? (
                <p className="text-sm text-muted-foreground">Loading favorites…</p>
              ) : favorites.length === 0 && !favoritesError ? (
                <p className="text-sm text-muted-foreground">
                  No favorites yet. Process a document, then press S or Save on the summary
                  or question page.
                </p>
              ) : (
                <ol
                  className="space-y-3 list-decimal pl-5 marker:text-muted-foreground"
                  start={favStart + 1}
                >
                  {favoritesOnPage.map((fav, index) => {
                    const globalIndex = favStart + index;
                    const slotKey = index + 1;
                    const shortcutHint = ` Shortcut key ${slotKey} on this page.`;
                    return (
                      <li key={`${fav.document_id}:${fav.article_id}:${globalIndex}`}>
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
                            aria-label={`Open summary and questions for item ${globalIndex + 1} of ${favCount}: ${fav.heading || 'article'}.${shortcutHint}`}
                          >
                            <BookOpen className="h-5 w-5" aria-hidden="true" />
                            Open summary and Q and A
                            <span className="text-xs opacity-90">({slotKey})</span>
                          </Button>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              )}
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

