import React, { useRef } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function FileUpload({ onProcessComplete, onError, onUploadStart, loading }) {
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = [
      'application/pdf',
      'image/jpeg',
      'image/jpg',
      'image/png',
      'image/tiff',
      'image/bmp'
    ];

    if (!allowedTypes.includes(file.type)) {
      onError('Please upload a PDF or image file (JPG, PNG, TIFF, BMP)');
      return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      onError('File size must be less than 50MB');
      return;
    }

    onUploadStart();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes timeout for large files
      });

      onProcessComplete(response.data);
    } catch (err) {
      if (err.response) {
        onError(err.response.data.detail || 'Processing failed. Please try again.');
      } else if (err.request) {
        onError('Unable to connect to server. Please make sure the backend is running.');
      } else {
        onError(err.message || 'An unexpected error occurred');
      }
    } finally {
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && fileInputRef.current) {
      fileInputRef.current.files = e.dataTransfer.files;
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div className="w-full">
      <div
        className={`
          relative bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl
          border-3 border-dashed transition-all duration-300
          ${loading 
            ? 'border-primary-400 bg-gray-50 cursor-not-allowed' 
            : 'border-primary-400 hover:border-primary-500 hover:bg-white hover:shadow-3xl cursor-pointer transform hover:-translate-y-1'
          }
        `}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => !loading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.jpg,.jpeg,.png,.tiff,.bmp"
          onChange={handleFileChange}
          disabled={loading}
          className="hidden"
        />

        {loading ? (
          <div className="flex flex-col items-center justify-center py-16 px-8">
            <div className="relative mb-6">
              <div className="w-16 h-16 border-4 border-primary-200 border-t-primary-500 rounded-full animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-8 h-8 bg-primary-500 rounded-full animate-pulse"></div>
              </div>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">
              Processing document...
            </h3>
            <p className="text-sm text-gray-600 text-center max-w-md">
              This may take a few minutes depending on the file size
            </p>
            <div className="mt-4 w-64 h-1 bg-gray-200 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-primary-500 to-purple-500 rounded-full animate-pulse-slow"></div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-16 px-8">
            <div className="mb-6 relative">
              <div className="w-24 h-24 bg-gradient-to-br from-primary-400 to-purple-500 rounded-2xl flex items-center justify-center shadow-lg transform rotate-3 hover:rotate-6 transition-transform duration-300">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-400 rounded-full border-2 border-white animate-pulse"></div>
            </div>
            
            <h2 className="text-2xl font-bold text-gray-800 mb-3">
              Drop your file here or click to browse
            </h2>
            
            <div className="flex flex-wrap items-center justify-center gap-2 mb-4">
              <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                PDF
              </span>
              <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                JPG
              </span>
              <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                PNG
              </span>
              <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                TIFF
              </span>
              <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                BMP
              </span>
            </div>
            
            <p className="text-sm text-gray-500 flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Maximum file size: 50MB
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default FileUpload;



