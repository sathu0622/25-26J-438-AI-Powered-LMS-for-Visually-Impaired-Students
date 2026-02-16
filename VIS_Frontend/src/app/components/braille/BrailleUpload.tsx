import React, { useState, useRef } from 'react';
import { Upload, FileText, X, Info, Loader2 } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { api } from '../../services/api';

interface BrailleUploadProps {
  onUpload: (data: { question: string; answer: string; fullText: string }) => void;
}

interface ConvertPdfResponse {
  status: string;
  question: string;
  answer: string;
  full_text: string;
}

export const BrailleUpload = ({ onUpload }: BrailleUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await api.postFormData<ConvertPdfResponse>(
        '/braille/convert-pdf',
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
                <Upload
                  className="h-16 w-16 text-muted-foreground"
                  aria-hidden="true"
                />
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
        </>
      ) : (
        <>
          {/* File Preview */}
          <Card className="overflow-hidden p-4">
            <div className="space-y-4">
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
            <h3 className="text-sm">Image Guidelines:</h3>
            <ul className="space-y-1 text-xs text-muted-foreground">
              <li>• Upload a PDF file containing Braille Unicode text</li>
              <li>• The PDF should contain both question and answer</li>
              <li>• Supported format: PDF (.pdf)</li>
              <li>• The system will extract and convert Braille to text automatically</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};