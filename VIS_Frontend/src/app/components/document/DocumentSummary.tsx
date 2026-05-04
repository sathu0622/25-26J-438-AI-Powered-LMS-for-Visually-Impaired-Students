import { useState, useEffect, useRef, useCallback } from 'react';
import { ZoomIn, ZoomOut, MessageSquare, Keyboard, Bookmark } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { AudioPlayer } from '../AudioPlayer';
import { useTTS } from '../../contexts/TTSContext';
import { addFavoriteArticle } from './favoritesApi';

interface ArticleInfo {
  article_id: string;
  index?: number;
  heading?: string;
  subheading?: string;
  column?: string;
  word_count?: number;
}

interface DocumentSummaryProps {
  documentId: string;
  summary: string;
  syllabusMatchMessage?: string | null;
  onAskQuestion: (mode: 'voice' | 'text') => void;
  articles?: ArticleInfo[];
  selectedArticleId?: string | null;
  onSelectArticle?: (articleId: string) => void;
}

export const DocumentSummary = ({
  documentId,
  summary,
  syllabusMatchMessage = null,
  onAskQuestion,
  articles,
  selectedArticleId,
  onSelectArticle,
}: DocumentSummaryProps) => {
  const { speak, cancel } = useTTS();
  const [textSize, setTextSize] = useState(100);
  const [favoriteSaving, setFavoriteSaving] = useState(false);
  const [favoriteStatus, setFavoriteStatus] = useState<string | null>(null);
  const favoriteInFlightRef = useRef(false);
  const initialSpeakDoneRef = useRef(false);
  const prevSelectedArticleIdRef = useRef<string | null | undefined>(undefined);
  const hasScheduledInitialRef = useRef(false);
  /** Track last spoken summary so we only read when API returns new content (not the stale one). */
  const lastSpokenSummaryRef = useRef<string>('');
  const lastSpokenSyllabusRef = useRef<string>('');
  const narrationStepRef = useRef<
    'idle' | 'intro' | 'articles' | 'syllabus' | 'summary' | 'done'
  >('idle');
  const hasArticlesNarrationRef = useRef(false);
  const summaryNarrationTextRef = useRef('');
  const articlesNarrationTextRef = useRef('');
  const pendingSyllabusRef = useRef('');
  const narrationSessionRef = useRef(0);

  const speakNarrationStep = useCallback((sessionId: number) => {
    if (sessionId !== narrationSessionRef.current) return;
    const step = narrationStepRef.current;

    if (step === 'intro') {
      const introText =
        'Document Summary page. Press A to replay summary, Press N to skip current voice and go to next, Press S to save the current article to favorites, Press V for voice question, Press T for text question, Press Plus to increase text size, Press Minus to decrease text size.';

      speak(introText, {
        interrupt: true,
        onEnd: () => {
          if (sessionId !== narrationSessionRef.current) return;
          if (hasArticlesNarrationRef.current) {
            narrationStepRef.current = 'articles';
          } else if (pendingSyllabusRef.current.trim()) {
            narrationStepRef.current = 'syllabus';
          } else {
            narrationStepRef.current = 'summary';
          }
          speakNarrationStep(sessionId);
        },
      });
      return;
    }

    if (step === 'articles') {
      speak(articlesNarrationTextRef.current, {
        interrupt: true,
        onEnd: () => {
          if (sessionId !== narrationSessionRef.current) return;
          narrationStepRef.current = pendingSyllabusRef.current.trim()
            ? 'syllabus'
            : 'summary';
          speakNarrationStep(sessionId);
        },
      });
      return;
    }

    if (step === 'syllabus') {
      const msg = pendingSyllabusRef.current.trim();
      if (!msg) {
        narrationStepRef.current = 'summary';
        speakNarrationStep(sessionId);
        return;
      }
      if (msg === lastSpokenSyllabusRef.current) {
        pendingSyllabusRef.current = '';
        narrationStepRef.current = 'summary';
        speakNarrationStep(sessionId);
        return;
      }
      lastSpokenSyllabusRef.current = msg;
      pendingSyllabusRef.current = '';
      speak(msg, {
        interrupt: true,
        onEnd: () => {
          if (sessionId !== narrationSessionRef.current) return;
          narrationStepRef.current = 'summary';
          speakNarrationStep(sessionId);
        },
      });
      return;
    }

    if (step === 'summary') {
      lastSpokenSummaryRef.current = summaryNarrationTextRef.current;
      speak('Reading the latest summary for the selected article.', {
        interrupt: true,
        onEnd: () => {
          if (sessionId !== narrationSessionRef.current) return;
          speak(summaryNarrationTextRef.current, {
            interrupt: false,
            onEnd: () => {
              if (sessionId !== narrationSessionRef.current) return;
              narrationStepRef.current = 'done';
            },
          });
        },
      });
    }
  }, [speak]);

  const skipToNextNarration = useCallback(() => {
    const step = narrationStepRef.current;
    if (step === 'idle' || step === 'done') return;

    narrationSessionRef.current += 1;
    const sessionId = narrationSessionRef.current;
    cancel();
    if (step === 'intro') {
      if (hasArticlesNarrationRef.current) {
        narrationStepRef.current = 'articles';
      } else if (pendingSyllabusRef.current.trim()) {
        narrationStepRef.current = 'syllabus';
      } else {
        narrationStepRef.current = 'summary';
      }
      speakNarrationStep(sessionId);
      return;
    }
    if (step === 'articles') {
      narrationStepRef.current = pendingSyllabusRef.current.trim()
        ? 'syllabus'
        : 'summary';
      speakNarrationStep(sessionId);
      return;
    }
    if (step === 'syllabus') {
      narrationStepRef.current = 'summary';
      speakNarrationStep(sessionId);
      return;
    }
    if (step === 'summary') {
      narrationStepRef.current = 'done';
    }
  }, [cancel, speakNarrationStep]);

  // Single initial voice sequence: intro → articles (if any) → "Reading summary..." → summary (once on mount)
  useEffect(() => {
    cancel();

    if (!summary.trim() || hasScheduledInitialRef.current) {
      return () => cancel();
    }
    hasScheduledInitialRef.current = true;
    initialSpeakDoneRef.current = true;
    const trimmed = summary.trim();
    lastSpokenSummaryRef.current = trimmed;

    summaryNarrationTextRef.current = trimmed;
    if (articles && articles.length > 0) {
      const parts: string[] = [];
      parts.push(`There are ${articles.length} articles in this document.`);
      articles.forEach((article, index) => {
        const number = index + 1;
        const heading = article.heading || `Article ${number}`;
        parts.push(`Number ${number} article: ${heading}.`);
      });
      parts.push(
        'To summarize an article, press the number key that matches the article. For example, press 1 for article 1, press 2 for article 2, and so on.'
      );
      hasArticlesNarrationRef.current = true;
      articlesNarrationTextRef.current = parts.join(' ');
    } else {
      hasArticlesNarrationRef.current = false;
      articlesNarrationTextRef.current = '';
    }

    const timer = setTimeout(() => {
      narrationSessionRef.current += 1;
      const sessionId = narrationSessionRef.current;
      narrationStepRef.current = 'intro';
      speakNarrationStep(sessionId);
    }, 500);

    return () => {
      clearTimeout(timer);
      narrationSessionRef.current += 1;
      narrationStepRef.current = 'idle';
      cancel();
    };
  }, [articles, summary, speak, cancel, speakNarrationStep]);

  // When user selects a different article: only say "Loading summary" — do NOT read the old summary
  useEffect(() => {
    if (!initialSpeakDoneRef.current) return;
    if (prevSelectedArticleIdRef.current === undefined) {
      prevSelectedArticleIdRef.current = selectedArticleId;
      return;
    }
    if (prevSelectedArticleIdRef.current === selectedArticleId) return;
    prevSelectedArticleIdRef.current = selectedArticleId;
    cancel();
    speak('Article selected. Loading summary.', { interrupt: true });
  }, [selectedArticleId, speak, cancel]);

  // When summary content changes (e.g. after API returns for the selected article), read the new summary
  useEffect(() => {
    const trimmed = summary.trim();
    if (!trimmed || !initialSpeakDoneRef.current) return;
    if (trimmed === lastSpokenSummaryRef.current) return;

    narrationSessionRef.current += 1;
    const sessionId = narrationSessionRef.current;
    cancel();
    summaryNarrationTextRef.current = trimmed;
    narrationStepRef.current = pendingSyllabusRef.current.trim()
      ? 'syllabus'
      : 'summary';
    speakNarrationStep(sessionId);
  }, [summary, cancel, speakNarrationStep]);

  // Announce syllabus topic only when article is in syllabus (non-empty message).
  useEffect(() => {
    const msg = (syllabusMatchMessage || '').trim();
    if (!msg) {
      // Reset so the same message can be announced again after article changes.
      lastSpokenSyllabusRef.current = '';
      pendingSyllabusRef.current = '';
      return;
    }
    pendingSyllabusRef.current = msg;
  }, [syllabusMatchMessage]);

  const saveCurrentArticleToFavorites = useCallback(async () => {
    if (!documentId || !selectedArticleId || favoriteInFlightRef.current) return;

    favoriteInFlightRef.current = true;
    setFavoriteSaving(true);
    setFavoriteStatus(null);
    cancel();

    const heading =
      articles?.find((a) => a.article_id === selectedArticleId)?.heading ||
      'this article';

    try {
      await addFavoriteArticle(documentId, selectedArticleId);
      const ok = `Saved to favorites: ${heading}.`;
      setFavoriteStatus(ok);
      speak(ok, { interrupt: true });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Could not save favorite.';
      setFavoriteStatus(message);
      speak(message, { interrupt: true });
    } finally {
      favoriteInFlightRef.current = false;
      setFavoriteSaving(false);
    }
  }, [documentId, selectedArticleId, articles, speak, cancel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target?.closest('input, textarea, select, [contenteditable="true"]')
      ) {
        return;
      }

      // A key to replay summary
      if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        cancel();
        speak(summary, { interrupt: true });
        return;
      }

      // N key: skip current narration and move to next spoken step
      if (e.key === 'n' || e.key === 'N') {
        e.preventDefault();
        skipToNextNarration();
        return;
      }

      // S key: save current article to favorites
      if (e.key === 's' || e.key === 'S') {
        if (!documentId || !selectedArticleId) return;
        e.preventDefault();
        void saveCurrentArticleToFavorites();
        return;
      }

      // V key for voice question
      if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        onAskQuestion('voice');
        return;
      }

      // T key for text question
      if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        onAskQuestion('text');
        return;
      }

      // Plus/Equals key to increase text size
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        increaseTextSize();
        return;
      }

      // Minus key to decrease text size
      if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        decreaseTextSize();
        return;
      }

      // Number keys 1-9 to select articles directly
      if (onSelectArticle && articles && articles.length > 0) {
        const num = parseInt(e.key, 10);
        if (!Number.isNaN(num) && num >= 1 && num <= 9) {
          const index = num - 1;
          if (index < articles.length) {
            e.preventDefault();
            const targetArticle = articles[index];
            onSelectArticle(targetArticle.article_id);

            // Optional spoken confirmation; new summary will be read when API returns
            const confirmText = `Article ${num} selected: ${
              targetArticle.heading || 'no heading'
            }. Loading summary.`;
            cancel();
            speak(confirmText, { interrupt: true });
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [
    summary,
    onAskQuestion,
    articles,
    onSelectArticle,
    speak,
    cancel,
    documentId,
    selectedArticleId,
    saveCurrentArticleToFavorites,
    skipToNextNarration,
  ]);

  const increaseTextSize = () => {
    setTextSize((prev) => Math.min(prev + 10, 150));
  };

  const decreaseTextSize = () => {
    setTextSize((prev) => Math.max(prev - 10, 80));
  };

  const selectedArticle =
    articles && selectedArticleId
      ? articles.find((article) => article.article_id === selectedArticleId)
      : undefined;

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-2xl">Document Summary</h1>
        <p className="text-muted-foreground">
          A replay • N skip voice • S save favorite • V voice question • T text question • +/- text size
        </p>
        {selectedArticle && (
          <p className="text-sm text-secondary">
            Currently summarizing:{' '}
            <span className="font-medium">
              {selectedArticle.heading || `Article ${selectedArticle.index}`}
            </span>
          </p>
        )}
      </div>

      {/* Article selection (from processed document) */}
      {articles && articles.length > 0 && onSelectArticle && (
        <Card className="p-4 space-y-3" aria-label="Select article to summarize">
          <div className="flex items-center justify-between">
            <h2 className="text-lg">Articles in this document</h2>
            <p className="text-xs text-muted-foreground">
              Press number keys 1 to 9 to select an article
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {articles.map((article) => {
              const isSelected = article.article_id === selectedArticleId;
              return (
                <button
                  key={article.article_id}
                  type="button"
                  onClick={() => onSelectArticle(article.article_id)}
                  className={`text-left rounded-lg border p-3 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 transition-colors ${
                    isSelected
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:border-primary/60'
                  }`}
                  aria-pressed={isSelected}
                  aria-label={`Select article ${article.index || ''} ${
                    article.heading || ''
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium">
                        {article.heading || `Article ${article.index}`}
                      </p>
                      {article.subheading && (
                        <p className="text-xs text-muted-foreground line-clamp-2">
                          {article.subheading}
                        </p>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground text-right">
                      {article.column && article.column !== 'full' && (
                        <p>Column: {article.column}</p>
                      )}
                      {/* {article.word_count != null && article.word_count > 0 && (
                        <p>{article.word_count} words</p>
                      )} */}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </Card>
      )}

      {/* Audio Player */}
      <AudioPlayer text={summary} autoPlay={false} />

      {/* Syllabus Classification (shown only if article belongs to syllabus) */}
      {syllabusMatchMessage && (
        <Card className="border-emerald-500/50 bg-emerald-500/10 p-4">
          <p className="text-sm leading-relaxed" role="status" aria-live="polite">
            {syllabusMatchMessage}
          </p>
        </Card>
      )}

      {/* Save to favorites */}
      <Card className="p-4 space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg">Favorites</h2>
            <p className="text-sm text-muted-foreground">
              Save the currently selected article for quick access later. Keyboard: S
            </p>
          </div>
          <Button
            type="button"
            onClick={() => void saveCurrentArticleToFavorites()}
            disabled={
              favoriteSaving || !documentId || !selectedArticleId
            }
            size="lg"
            variant="secondary"
            className="min-h-[56px] gap-2 shrink-0"
            aria-label="Save current article to favorites. Keyboard shortcut S."
          >
            <Bookmark className="h-5 w-5" aria-hidden="true" />
            {favoriteSaving ? 'Saving…' : 'Save article to favorites'}
          </Button>
        </div>
        {favoriteStatus && (
          <p className="text-sm" role="status" aria-live="polite">
            {favoriteStatus}
          </p>
        )}
      </Card>

      {/* Text Controls */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">Text Size</p>
        <div className="flex gap-2">
          <Button
            onClick={decreaseTextSize}
            variant="outline"
            size="sm"
            aria-label="Decrease text size"
          >
            <ZoomOut className="h-4 w-4" aria-hidden="true" />
          </Button>
          <span className="flex items-center px-2 text-sm" aria-live="polite">
            {textSize}%
          </span>
          <Button
            onClick={increaseTextSize}
            variant="outline"
            size="sm"
            aria-label="Increase text size"
          >
            <ZoomIn className="h-4 w-4" aria-hidden="true" />
          </Button>
        </div>
      </div>

      {/* Summary Text */}
      <Card className="p-6">
        <div
          className="space-y-4"
          style={{ fontSize: `${textSize}%` }}
          role="article"
          aria-label="Document summary"
        >
          <h2 className="text-lg">Summary</h2>
          <p className="leading-relaxed">{summary}</p>
        </div>
      </Card>

      {/* Question Options */}
      <div className="space-y-3">
        <h2 className="text-center">Ask a Question About This Document</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <Button
            onClick={() => onAskQuestion('voice')}
            size="lg"
            className="min-h-[72px] flex-col gap-2"
          >
            <MessageSquare className="h-6 w-6" aria-hidden="true" />
            <span>Voice Question</span>
          </Button>
          <Button
            onClick={() => onAskQuestion('text')}
            size="lg"
            variant="outline"
            className="min-h-[72px] flex-col gap-2"
          >
            <Keyboard className="h-6 w-6" aria-hidden="true" />
            <span>Text Question</span>
          </Button>
        </div>
      </div>
    </div>
  );
};