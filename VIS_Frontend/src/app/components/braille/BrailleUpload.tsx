import { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, X, Info } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';

interface BrailleUploadProps {
  onUpload: (file: File) => void;
}

export const BrailleUpload = ({ onUpload }: BrailleUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
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
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    setFileName(file.name);
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleRemove = () => {
    setPreview(null);
    setFileName(null);
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onUpload(selectedFile);
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

      {/* Upload Area */}
      {!preview ? (
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
                <p>Drag and drop your image here</p>
                <p className="text-sm text-muted-foreground">or</p>
              </div>
              <Button
                onClick={() => fileInputRef.current?.click()}
                size="lg"
                className="min-h-[56px]"
              >
                <ImageIcon className="mr-2 h-5 w-5" aria-hidden="true" />
                Browse Image
              </Button>
            </div>
          </Card>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileSelect}
            aria-label="Braille image upload"
          />
        </>
      ) : (
        <>
          {/* Image Preview */}
          <Card className="overflow-hidden p-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="truncate text-sm">{fileName}</p>
                <Button
                  onClick={handleRemove}
                  variant="ghost"
                  size="icon"
                  aria-label="Remove image"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
              <img
                src={preview}
                alt="Braille answer sheet preview"
                className="w-full rounded-lg border"
              />
            </div>
          </Card>

          {/* Action Buttons */}
          <div className="grid gap-3 sm:grid-cols-2">
            <Button
              onClick={handleRemove}
              variant="outline"
              size="lg"
              className="min-h-[56px]"
            >
              Replace Image
            </Button>
            <Button onClick={handleSubmit} size="lg" className="min-h-[56px]">
              Process Answer Sheet
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
              <li>• Ensure good lighting</li>
              <li>• Keep the image flat and clear</li>
              <li>• Make sure all Braille dots are visible</li>
              <li>• Supported formats: JPG, PNG</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};