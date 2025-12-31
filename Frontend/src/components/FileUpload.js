import React, { useRef } from 'react';
import axios from 'axios';
import './FileUpload.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
    <div className="file-upload-container">
      <div
        className={`upload-area ${loading ? 'loading' : ''}`}
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
          style={{ display: 'none' }}
        />

        {loading ? (
          <div className="loading-content">
            <div className="spinner"></div>
            <p>Processing document...</p>
            <p className="loading-subtitle">This may take a few minutes</p>
          </div>
        ) : (
          <div className="upload-content">
            <div className="upload-icon">📤</div>
            <h2>Drop your file here or click to browse</h2>
            <p>Supports PDF, JPG, PNG, TIFF, BMP</p>
            <p className="file-size-hint">Max file size: 50MB</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default FileUpload;





