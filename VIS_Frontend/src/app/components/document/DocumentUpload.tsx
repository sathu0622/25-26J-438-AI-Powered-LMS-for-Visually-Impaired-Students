import { useState, useRef } from 'react';
import { Upload, FileText, Image as ImageIcon, Loader2 } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';

interface DocumentUploadProps {
  onUpload: (file: File) => void;
}

export const DocumentUpload = ({ onUpload }: DocumentUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
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
