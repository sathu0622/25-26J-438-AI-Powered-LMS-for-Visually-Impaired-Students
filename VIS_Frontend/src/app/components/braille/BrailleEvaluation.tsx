import { useState, useEffect, useRef } from 'react';
import { Loader2, CheckCircle2, XCircle, AlertCircle, Volume2 } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { AudioPlayer } from '../AudioPlayer';
import { brailleApi } from '../../services/api';
import { useTTS } from '../../contexts/TTSContext';

interface BrailleEvaluationProps {
  onBack: () => void;
  convertedData?: {
    question: string;
    answer: string;
    fullText: string;
    /** Pass true when coming from manual input to skip the "converted" pause */
    autoEvaluate?: boolean;
  };
}

interface EvaluationResponse {
  question: string;
  student_answer: string;
  model_answer: string;
  final_score: number;
  semantic_similarity: number;
  keyword_match: number;
  jaccard_similarity: number;
  status: string;
  feedback: string;
}

type EvaluationStatus = 'converting' | 'converted' | 'evaluating' | 'complete';
type ResultType = 'correct' | 'partial' | 'incorrect';

interface ParsedFeedback {
  summary: string;
  missingPoints: string[];
  closingNote: string;
  chapter: string;
  topic: string;
}

// ── Feedback parser ─────────────────────────────────────────────────────────
const parseFeedback = (raw: string): ParsedFeedback => {
  const lines = raw.split('\n').map((l) => l.trim()).filter(Boolean);

  let summary = '';
  const missingPoints: string[] = [];
  let closingNote = '';
  let chapter = '';
  let topic = '';
  let inFurtherStudy = false;
  let inMissing = false;

  for (const line of lines) {
    if (/further study recommendation/i.test(line)) {
      inFurtherStudy = true;
      inMissing = false;
      continue;
    }

    if (inFurtherStudy) {
      if (/chapter/i.test(line)) {
        chapter = line.replace(/[•\-*]?\s*chapter\s*:\s*/i, '').trim();
      } else if (/topic/i.test(line)) {
        topic = line.replace(/[•\-*]?\s*topic\s*:\s*/i, '').trim();
      }
      // skip the "Review this section..." line — we render it ourselves
      continue;
    }

    if (
      /^(try to include|important points|you identified|missing from)/i.test(line)
    ) {
      inMissing = true;
      continue;
    }

    if (inMissing) {
      const numbered = line.match(/^\d+\.\s+(.+)/);
      if (numbered) {
        missingPoints.push(numbered[1].trim());
        continue;
      }
      if (/^(improve|revise|review|make sure)/i.test(line)) {
        closingNote = line;
        inMissing = false;
        continue;
      }
    }

    if (!summary && line.length > 0) {
      summary = line;
    }
  }

  return { summary, missingPoints, closingNote, chapter, topic };
};

// ── Feedback renderer ───────────────────────────────────────────────────────
const FeedbackContent = ({
  raw,
  result,
}: {
  raw: string;
  result: ResultType | null;
}) => {
  const { summary, missingPoints, closingNote, chapter, topic } =
    parseFeedback(raw);

  const summaryStyle: React.CSSProperties =
    result === 'correct'
      ? {
          borderLeft: '3px solid var(--color-border-success, #639922)',
          background: 'var(--color-background-success, #EAF3DE)',
          color: 'var(--color-text-success, #3B6D11)',
        }
      : result === 'partial'
      ? {
          borderLeft: '3px solid var(--color-border-warning, #EF9F27)',
          background: 'var(--color-background-warning, #FAEEDA)',
          color: 'var(--color-text-warning, #633806)',
        }
      : {
          borderLeft: '3px solid var(--color-border-destructive, #E24B4A)',
          background: 'var(--color-background-danger, #FCEBEB)',
          color: 'var(--color-text-destructive, #791F1F)',
        };

  return (
    <div className="space-y-4">
      {/* Summary line */}
      {summary && (
        <p
          className="rounded-md px-4 py-3 text-sm leading-relaxed"
          style={summaryStyle}
        >
          {summary}
        </p>
      )}

      {/* Missing points */}
      {missingPoints.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            Points to include in your answer:
          </p>
          <ol className="space-y-2">
            {missingPoints.map((pt, i) => (
              <li key={i} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium text-muted-foreground">
                  {i + 1}
                </span>
                <span className="text-sm leading-relaxed">{pt}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Closing note */}
      {closingNote && (
        <p className="text-sm italic text-muted-foreground">{closingNote}</p>
      )}

      {/* Further study */}
      {(chapter || topic) && (
        <div className="mt-2 space-y-2 rounded-lg border bg-muted/40 p-4">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Further study recommendation
          </p>
          {chapter && (
            <div className="flex gap-3 text-sm">
              <span className="w-16 flex-shrink-0 font-medium">Chapter</span>
              <span className="text-muted-foreground">{chapter}</span>
            </div>
          )}
          {topic && (
            <div className="flex gap-3 text-sm">
              <span className="w-16 flex-shrink-0 font-medium">Topic</span>
              <span className="text-muted-foreground">{topic}</span>
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            Review this section in your textbook to strengthen your
            understanding.
          </p>
        </div>
      )}
    </div>
  );
};

// ── Main component ──────────────────────────────────────────────────────────
export const BrailleEvaluation = ({
  onBack,
  convertedData,
}: BrailleEvaluationProps) => {
  const [status, setStatus] = useState<EvaluationStatus>('converting');
  const [progress, setProgress] = useState(0);
  const [convertedText, setConvertedText] = useState('');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<ResultType | null>(null);
  const [score, setScore] = useState(0);
  const [rawFeedback, setRawFeedback] = useState('');
  const [modelAnswer, setModelAnswer] = useState('');
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [showDetailedReport, setShowDetailedReport] = useState(false);
  const [semanticScore, setSemanticScore] = useState(0);
  const [keywordScore, setKeywordScore] = useState(0);
  const [jaccardScore, setJaccardScore] = useState(0);

  const { speak, cancel } = useTTS();

  const questionRef = useRef('');
  const convertedTextRef = useRef('');

  useEffect(() => {
    questionRef.current = question;
    convertedTextRef.current = convertedText;
  }, [question, convertedText]);

  // ── TTS announcements ─────────────────────────────────────────────────────
  useEffect(() => {
    cancel();
    if (status === 'converted' && convertedText) {
      const t = setTimeout(
        () =>
          speak(
            'Answer converted from Braille. Press E to evaluate your answer, or Press A to hear your answer read aloud.',
            { interrupt: true }
          ),
        500
      );
      return () => clearTimeout(t);
    }
    if (status === 'complete' && !showDetailedReport) {
      const t = setTimeout(
        () =>
          speak(
            `Evaluation complete. Your score is ${score} percent. Press F to replay feedback, Press D for detailed report, Press B to upload another answer, or Press Escape to go back.`,
            { interrupt: true }
          ),
        10000
      );
      return () => clearTimeout(t);
    }
    if (showDetailedReport) {
      cancel();
      const t = setTimeout(
        () =>
          speak(
            'Detailed Report page. Press A to hear your answer, Press M to hear the model answer, Press B to go back to summary, or Press Escape.',
            { interrupt: true }
          ),
        500
      );
      return () => clearTimeout(t);
    }
    return () => cancel();
  }, [status, showDetailedReport, convertedText, score, speak, cancel]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if ((e.key === 'a' || e.key === 'A') && convertedText) {
        e.preventDefault();
        cancel();
        speak(convertedText, { interrupt: true });
      }
      if ((e.key === 'm' || e.key === 'M') && showDetailedReport) {
        e.preventDefault();
        cancel();
        speak(modelAnswer, { interrupt: true });
      }
      if (
        (e.key === 'f' || e.key === 'F') &&
        status === 'complete' &&
        rawFeedback
      ) {
        e.preventDefault();
        cancel();
        speak(`Your score is ${score}%. ${rawFeedback}`, { interrupt: true });
      }
      if ((e.key === 'e' || e.key === 'E') && status === 'converted') {
        e.preventDefault();
        handleEvaluate();
      }
      if (
        (e.key === 'd' || e.key === 'D') &&
        status === 'complete' &&
        !showDetailedReport
      ) {
        e.preventDefault();
        setShowDetailedReport(true);
      }
      if (e.key === 'b' || e.key === 'B') {
        e.preventDefault();
        if (showDetailedReport) {
          setShowDetailedReport(false);
        } else if (status === 'complete') {
          onBack();
        }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        if (showDetailedReport) {
          setShowDetailedReport(false);
        } else {
          onBack();
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [
    status,
    convertedText,
    rawFeedback,
    score,
    showDetailedReport,
    onBack,
    speak,
    cancel,
  ]);

  // ── Data initialisation ───────────────────────────────────────────────────
  useEffect(() => {
    if (convertedData) {
      setQuestion(convertedData.question);
      setConvertedText(convertedData.answer || convertedData.fullText);
      setProgress(50);

      if (convertedData.autoEvaluate) {
        setStatus('evaluating');
        setProgress(75);
        setTimeout(() => {
          handleEvaluateWithData(
            convertedData.question,
            convertedData.answer || convertedData.fullText
          );
        }, 0);
      } else {
        setStatus('converted');
      }
    } else {
      const timer = setTimeout(() => {
        setProgress(50);
        const mockQuestion =
          'Explain the importance of accessible education for students with visual impairments.';
        const mockConverted =
          'Accessible education ensures equal learning opportunities through adaptive technologies and inclusive teaching methods.';
        setQuestion(mockQuestion);
        setConvertedText(mockConverted);
        setStatus('converted');
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [convertedData]);

  // ── Evaluation helpers ────────────────────────────────────────────────────
  const handleEvaluate = async () => {
    await handleEvaluateWithData(
      questionRef.current,
      convertedTextRef.current
    );
  };

  const handleEvaluateWithData = async (q: string, ans: string) => {
    if (!q || !ans) {
      setEvaluationError('Question and answer are required for evaluation');
      return;
    }

    setStatus('evaluating');
    setProgress(75);
    setEvaluationError(null);

    try {
      const response = await brailleApi.post<EvaluationResponse>('/evaluate', {
        question: q,
        student_answer: ans,
      });

      setProgress(100);
      setModelAnswer(response.model_answer);
      setScore(Math.round(response.final_score));
      setSemanticScore(response.semantic_similarity || 0);
      setKeywordScore(response.keyword_match || 0);
      setJaccardScore(response.jaccard_similarity || 0);

      if (response.final_score >= 75) {
        setResult('correct');
      } else if (response.final_score >= 45) {
        setResult('partial');
      } else {
        setResult('incorrect');
      }

      // Store the raw feedback string — parsing happens at render time
      setRawFeedback(response.feedback || '');
      setStatus('complete');
    } catch (err) {
      setEvaluationError(
        err instanceof Error
          ? err.message
          : 'Failed to evaluate answer. Please try again.'
      );
      setStatus('converted');
      setProgress(50);
    }
  };

  // ── UI helpers ────────────────────────────────────────────────────────────
  const getResultIcon = () => {
    switch (result) {
      case 'correct':
        return <CheckCircle2 className="h-16 w-16 text-success" />;
      case 'partial':
        return <AlertCircle className="h-16 w-16 text-warning" />;
      case 'incorrect':
        return <XCircle className="h-16 w-16 text-destructive" />;
      default:
        return null;
    }
  };

  const getResultColor = () => {
    switch (result) {
      case 'correct':
        return 'border-success bg-success/10';
      case 'partial':
        return 'border-warning bg-warning/10';
      case 'incorrect':
        return 'border-destructive bg-destructive/10';
      default:
        return '';
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 pb-24">
      {!showDetailedReport ? (
        <>
          {/* Header */}
          <div className="space-y-2 text-center">
            <h1 className="text-2xl">Answer Evaluation</h1>
            <p className="text-muted-foreground">
              {status === 'converting' && 'Converting Braille to text...'}
              {status === 'converted' &&
                'Press E to evaluate • Press A to hear answer'}
              {status === 'evaluating' && 'Evaluating your answer...'}
              {status === 'complete' &&
                'Press F to replay feedback • Press D for details • Press B to upload another'}
            </p>
          </div>

          {/* Error Message */}
          {evaluationError && (
            <Card className="border-destructive bg-destructive/10 p-4">
              <p className="text-sm text-destructive">{evaluationError}</p>
            </Card>
          )}

          {/* Progress */}
          {status !== 'complete' && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <p className="text-center text-sm text-muted-foreground">
                {progress}% complete
              </p>
            </div>
          )}

          {/* Question */}
          <Card className="p-6">
            <div className="space-y-2">
              <h2 className="text-sm text-muted-foreground">Question:</h2>
              <p className="text-lg">{question}</p>
            </div>
          </Card>

          {/* Converted / Student Answer */}
          {status !== 'converting' && (
            <Card className="p-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm text-muted-foreground">
                    Your Answer (Converted from Braille):
                  </h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      const utterance = new SpeechSynthesisUtterance(
                        convertedText
                      );
                      window.speechSynthesis.speak(utterance);
                    }}
                    aria-label="Read answer aloud"
                  >
                    <Volume2 className="h-4 w-4" />
                  </Button>
                </div>
                <p className="leading-relaxed">{convertedText}</p>
              </div>
            </Card>
          )}

          {/* Evaluate Button */}
          {status === 'converted' && (
            <Button
              onClick={handleEvaluate}
              size="lg"
              className="w-full min-h-[56px]"
            >
              Evaluate Answer
            </Button>
          )}

          {/* Loading State */}
          {status === 'evaluating' && (
            <Card className="p-8">
              <div className="flex flex-col items-center gap-4 text-center">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <p>AI is evaluating your answer...</p>
              </div>
            </Card>
          )}

          {/* Results */}
          {status === 'complete' && result && (
            <>
              {/* Score Card */}
              <Card className={`border-2 p-8 ${getResultColor()}`}>
                <div className="flex flex-col items-center gap-4 text-center">
                  {getResultIcon()}
                  <div className="space-y-2">
                    <h2 className="text-2xl">Score: {score}%</h2>
                    <p className="text-lg">
                      {result === 'correct' && 'Excellent Work!'}
                      {result === 'partial' && 'Good Effort!'}
                      {result === 'incorrect' && 'Needs Improvement'}
                    </p>
                  </div>
                </div>
              </Card>

              {/* Feedback */}
              {rawFeedback && (
                <div className="space-y-4">
                  <h2 className="text-lg">Feedback</h2>
                  <AudioPlayer
                    text={`Your score is ${score}%. ${rawFeedback}`}
                    autoPlay={true}
                  />
                  <Card className="p-6">
                    <FeedbackContent raw={rawFeedback} result={result} />
                  </Card>
                </div>
              )}

              {/* Actions */}
              <div className="grid gap-3 sm:grid-cols-2">
                <Button
                  onClick={onBack}
                  variant="outline"
                  size="lg"
                  className="min-h-[56px]"
                >
                  Upload Another
                </Button>
                <Button
                  size="lg"
                  className="min-h-[56px]"
                  onClick={() => setShowDetailedReport(true)}
                >
                  View Detailed Report
                </Button>
              </div>
            </>
          )}
        </>
      ) : (
        /* ── Detailed Report ── */
        <div className="space-y-6">
          <h1 className="text-2xl">Detailed Report</h1>
          <p className="text-muted-foreground">
            Press A to hear your answer • Press M for model answer • Press B to
            go back
          </p>

          <Card className="p-6">
            <div className="space-y-4">
              <h2 className="text-sm text-muted-foreground">Question:</h2>
              <p className="text-lg">{question}</p>
            </div>
          </Card>

          <Card className="p-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm text-muted-foreground">Your Answer:</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const utterance = new SpeechSynthesisUtterance(
                      convertedText
                    );
                    window.speechSynthesis.speak(utterance);
                  }}
                  aria-label="Read your answer aloud"
                >
                  <Volume2 className="h-4 w-4" />
                </Button>
              </div>
              <p className="leading-relaxed">{convertedText}</p>
            </div>
          </Card>

          <Card className="p-6">
            <div className="space-y-2">
              <h2 className="text-sm text-muted-foreground">Score:</h2>
              <p className="text-lg font-medium">{score}%</p>
            </div>
          </Card>

          {/* <Card className="p-6 space-y-4">
            <h2 className="font-semibold">Similarity Breakdown</h2>
            <div>
              <p>Semantic Similarity: {Math.round(semanticScore)}%</p>
              <Progress value={semanticScore} />
            </div>
            <div>
              <p>Keyword Match: {Math.round(keywordScore)}%</p>
              <Progress value={keywordScore} />
            </div>
            <div>
              <p>Jaccard Similarity: {Math.round(jaccardScore)}%</p>
              <Progress value={jaccardScore} />
            </div>
          </Card> */}

          <Card className="border-2 border-success bg-success/5 p-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <h2 className="text-sm text-muted-foreground">
                    Model Answer (100%):
                  </h2>
                  <p className="text-xs text-muted-foreground">
                    This is what a perfect answer looks like
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const utterance = new SpeechSynthesisUtterance(modelAnswer);
                    window.speechSynthesis.speak(utterance);
                  }}
                  aria-label="Read model answer aloud"
                >
                  <Volume2 className="h-4 w-4" />
                </Button>
              </div>
              <p className="leading-relaxed">{modelAnswer}</p>
            </div>
          </Card>

          {/* Feedback in detailed report */}
          {rawFeedback && (
            <Card className="p-6">
              <div className="space-y-4">
                <h2 className="text-sm text-muted-foreground">Feedback:</h2>
                <FeedbackContent raw={rawFeedback} result={result} />
              </div>
            </Card>
          )}

          <Button
            onClick={() => setShowDetailedReport(false)}
            size="lg"
            className="min-h-[56px]"
          >
            Back to Summary
          </Button>
        </div>
      )}
    </div>
  );
};