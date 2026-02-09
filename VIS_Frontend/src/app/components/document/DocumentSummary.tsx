import { useState, useEffect } from 'react';
import { ZoomIn, ZoomOut, MessageSquare, Keyboard } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { AudioPlayer } from '../AudioPlayer';

interface ArticleInfo {
  article_id: string;
  index?: number;
  heading?: string;
  subheading?: string;
  column?: string;
  word_count?: number;
}

interface DocumentSummaryProps {
  summary: string;
  onAskQuestion: (mode: 'voice' | 'text') => void;
  articles?: ArticleInfo[];
  selectedArticleId?: string | null;
  onSelectArticle?: (articleId: string) => void;
}

export const DocumentSummary = ({
  summary,
  onAskQuestion,
  articles,
  selectedArticleId,
  onSelectArticle,
}: DocumentSummaryProps) => {
  const [textSize, setTextSize] = useState(100);
  const [hasAutoPlayed, setHasAutoPlayed] = useState(false);

  // Initial voice instructions + article list (run once)
  useEffect(() => {
    // STOP all previous speech immediately
    window.speechSynthesis.cancel();

    if (!hasAutoPlayed) {
      setHasAutoPlayed(true);

      setTimeout(() => {
        // Main instructions
        const introText =
          'Document Summary page. Press A to replay summary, Press V for voice question, Press T for text question, Press Plus to increase text size, Press Minus to decrease text size.';
        const introUtterance = new SpeechSynthesisUtterance(introText);
        window.speechSynthesis.speak(introUtterance);

        // If we have detected articles, read them out for visually impaired users
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

          const articlesText = parts.join(' ');
          const articlesUtterance = new SpeechSynthesisUtterance(articlesText);
          window.speechSynthesis.speak(articlesUtterance);
        }

      }, 500);
    }

    // Cleanup: stop speech when leaving page
    return () => {
      window.speechSynthesis.cancel();
    };
  }, [hasAutoPlayed, articles]);

  // Speak summary automatically whenever a new summary is available
  useEffect(() => {
    const trimmed = summary.trim();
    if (!trimmed) return;

    // Stop any ongoing speech and announce the new summary
    window.speechSynthesis.cancel();

    const preface = new SpeechSynthesisUtterance(
      'Reading the latest summary for the selected article.'
    );
    const content = new SpeechSynthesisUtterance(trimmed);

    window.speechSynthesis.speak(preface);
    window.speechSynthesis.speak(content);
  }, [summary, selectedArticleId]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // A key to replay summary
      if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        const utterance = new SpeechSynthesisUtterance(summary);
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
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

            // Optional spoken confirmation for visually impaired users
            const confirmText = `Article ${num} selected: ${
              targetArticle.heading || 'no heading'
            }. Summary will be shown below.`;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(
              new SpeechSynthesisUtterance(confirmText)
            );
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [summary, onAskQuestion, articles, onSelectArticle]);

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
          A to replay • V for voice question • T for text question • +/- to resize text
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
                      {article.word_count != null && article.word_count > 0 && (
                        <p>{article.word_count} words</p>
                      )}
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