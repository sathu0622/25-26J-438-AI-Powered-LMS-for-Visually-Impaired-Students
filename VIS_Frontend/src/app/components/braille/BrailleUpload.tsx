import React, { useState, useRef } from 'react';
import { Upload, FileText, X, Info, Loader2, PenLine } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { brailleApi } from '../../services/api';

interface BrailleUploadProps {
  onUpload: (data: { question: string; answer: string; fullText: string }) => void;
}

interface ConvertPdfResponse {
  status: string;
  question: string;
  answer: string;
  full_text: string;
}

// ── Manual Input Modal ──────────────────────────────────────────────────────
interface ManualInputModalProps {
  onClose: () => void;
  onSubmit: (data: { question: string; answer: string; fullText: string }) => void;
}

const ManualInputModal = ({ onClose, onSubmit }: ManualInputModalProps) => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = () => {
    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }
    if (!answer.trim()) {
      setError('Please enter an answer.');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    // onSubmit navigates to BrailleEvaluation, passing question + answer as
    // convertedData. BrailleEvaluation's useEffect picks this up, sets
    // status → 'converted', then the user (or auto-trigger) calls handleEvaluate.
    onSubmit({
      question: question.trim(),
      answer: answer.trim(),
      fullText: `Question: ${question.trim()}\nAnswer: ${answer.trim()}`,
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={(e) => e.target === e.currentTarget && !isSubmitting && onClose()}
    >
      <div className="w-full max-w-lg rounded-2xl bg-white shadow-2xl dark:bg-zinc-900">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-6 py-4 dark:border-zinc-700">
          <div className="flex items-center gap-2">
            <PenLine className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Enter Question & Answer</h2>
          </div>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Body */}
        <div className="space-y-4 px-6 py-5">
          {error && (
            <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-2">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          {/* Question */}
          <div className="space-y-1.5">
            <label
              htmlFor="manual-question"
              className="text-sm font-medium text-foreground"
            >
              Question
            </label>
            <textarea
              id="manual-question"
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                setError(null);
              }}
              rows={3}
              placeholder="Type or paste the question here…"
              disabled={isSubmitting}
              className="w-full resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
            />
          </div>

          {/* Answer */}
          <div className="space-y-1.5">
            <label
              htmlFor="manual-answer"
              className="text-sm font-medium text-foreground"
            >
              Student Answer
            </label>
            <textarea
              id="manual-answer"
              value={answer}
              onChange={(e) => {
                setAnswer(e.target.value);
                setError(null);
              }}
              rows={4}
              placeholder="Type or paste the student's answer here…"
              disabled={isSubmitting}
              className="w-full resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 border-t px-6 py-4 dark:border-zinc-700">
          <Button variant="outline" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Submitting…
              </>
            ) : (
              'Submit & Evaluate'
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

// ── Main Component ──────────────────────────────────────────────────────────
export const BrailleUpload = ({ onUpload }: BrailleUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      processFile(file);
    } else {
      setError('Please upload a PDF file');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        processFile(file);
        setError(null);
      } else {
        setError('Please upload a PDF file');
      }
    }
  };

  const processFile = (file: File) => {
    setFileName(file.name);
    setSelectedFile(file);
    setError(null);
  };

  const handleRemove = () => {
    setFileName(null);
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await brailleApi.postFormData<ConvertPdfResponse>(
        '/decode',
        formData
      );

      if (response.status === 'success') {
        onUpload({
          question: response.question,
          answer: response.answer,
          fullText: response.full_text,
        });
      } else {
        setError('Failed to convert PDF. Please try again.');
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to upload PDF. Please try again.'
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <>
      <div className="mx-auto max-w-2xl space-y-6 p-4 pb-24">
        {/* Header */}
        <div className="space-y-2 text-center">
          <h1 className="text-2xl">Braille Answer Sheet</h1>
          <p className="text-muted-foreground">
            Upload a photo of your Braille answer sheet for evaluation
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <Card className="border-destructive bg-destructive/10 p-4">
            <p className="text-sm text-destructive">{error}</p>
          </Card>
        )}

        {/* Upload Area */}
        {!selectedFile ? (
          <>
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
                  <Upload className="h-16 w-16 text-muted-foreground" aria-hidden="true" />
                </div>
                <div className="space-y-2">
                  <p>Drag and drop your PDF file here</p>
                  <p className="text-sm text-muted-foreground">or</p>
                </div>
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  size="lg"
                  className="min-h-[56px]"
                  disabled={isUploading}
                >
                  <FileText className="mr-2 h-5 w-5" aria-hidden="true" />
                  Browse PDF
                </Button>
              </div>
            </Card>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              className="hidden"
              onChange={handleFileSelect}
              aria-label="Braille PDF upload"
            />

            {/* Divider */}
            <div className="flex items-center gap-3">
              <div className="h-px flex-1 bg-border" />
              <span className="text-xs text-muted-foreground">or enter manually</span>
              <div className="h-px flex-1 bg-border" />
            </div>

            {/* Manual Input Trigger */}
            <Button
              variant="outline"
              size="lg"
              className="w-full min-h-[56px]"
              onClick={() => setShowModal(true)}
            >
              <PenLine className="mr-2 h-5 w-5" aria-hidden="true" />
              Enter Question &amp; Answer Manually
            </Button>
          </>
        ) : (
          <>
            {/* File Preview */}
            <Card className="overflow-hidden p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="h-8 w-8 text-primary" />
                  <p className="truncate text-sm font-medium">{fileName}</p>
                </div>
                <Button
                  onClick={handleRemove}
                  variant="ghost"
                  size="icon"
                  aria-label="Remove file"
                  disabled={isUploading}
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
            </Card>

            {/* Action Buttons */}
            <div className="grid gap-3 sm:grid-cols-2">
              <Button
                onClick={handleRemove}
                variant="outline"
                size="lg"
                className="min-h-[56px]"
                disabled={isUploading}
              >
                Replace PDF
              </Button>
              <Button
                onClick={handleSubmit}
                size="lg"
                className="min-h-[56px]"
                disabled={isUploading}
              >
                {isUploading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Convert & Evaluate'
                )}
              </Button>
            </div>
          </>
        )}

        {/* Guidelines */}
        <Card className="border-secondary bg-secondary/10 p-4">
          <div className="flex gap-3">
            <Info className="h-5 w-5 shrink-0 text-secondary" aria-hidden="true" />
            <div className="space-y-2">
              <h3 className="text-sm">Guidelines:</h3>
              <ul className="space-y-1 text-xs text-muted-foreground">
                <li>• Upload a PDF file containing Braille Unicode text</li>
                <li>• The PDF should contain both question and answer</li>
                <li>• Supported format: PDF (.pdf)</li>
                <li>• The system will extract and convert Braille to text automatically</li>
                <li>• Alternatively, enter the question and answer manually</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>

      {/* Manual Input Modal */}
      {showModal && (
        <ManualInputModal
          onClose={() => setShowModal(false)}
          onSubmit={(data) => {
            setShowModal(false);
            onUpload(data);
          }}
        />
      )}
    </>
  );
};