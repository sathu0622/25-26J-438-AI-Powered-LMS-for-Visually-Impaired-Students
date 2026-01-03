import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import ResultsDisplay from './components/ResultsDisplay';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleProcessComplete = (data) => {
    setResults(data);
    setLoading(false);
    setError(null);
  };

  const handleError = (err) => {
    setError(err);
    setLoading(false);
    setResults(null);
  };

  const handleUploadStart = () => {
    setLoading(true);
    setError(null);
    setResults(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-500 via-primary-600 to-purple-700">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-primary-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob" style={{ animationDelay: '4s' }}></div>
      </div>

      <div className="relative z-10 min-h-screen px-4 py-8 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <header className="text-center mb-12 pt-8">
            <div className="inline-flex items-center justify-center w-20 h-20 mb-6 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 shadow-lg">
              <span className="text-4xl">📄</span>
            </div>
            <h1 className="text-5xl sm:text-6xl font-bold text-white mb-4 drop-shadow-lg">
              AI Document Processor
            </h1>
            <p className="text-xl sm:text-2xl text-white/90 max-w-2xl mx-auto leading-relaxed">
              Extract and summarize text from Books, Magazines, and Newspapers
            </p>
            <p className="text-sm text-white/70 mt-3">
              {/* Powered by AI • OCR • Smart Summarization */}
            </p>
          </header>

          {/* File Upload Section */}
          <FileUpload
            onProcessComplete={handleProcessComplete}
            onError={handleError}
            onUploadStart={handleUploadStart}
            loading={loading}
          />

          {/* Error Message */}
          {error && (
            <div className="mt-6 animate-fade-in">
              <div className="bg-red-50 border-l-4 border-red-500 rounded-lg shadow-lg p-6">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <svg className="h-6 w-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="ml-3 flex-1">
                    <h3 className="text-lg font-semibold text-red-800 mb-2">Error</h3>
                    <p className="text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Results Display */}
          {results && (
            <div className="mt-8 animate-fade-in">
              <ResultsDisplay results={results} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

