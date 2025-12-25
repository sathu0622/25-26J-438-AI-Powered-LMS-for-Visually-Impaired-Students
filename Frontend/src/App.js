import React, { useState } from 'react';
import './App.css';
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
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>📄 Document Processor</h1>
          <p>Extract and summarize text from PDFs, Books, Magazines, and Newspapers</p>
        </header>

        <FileUpload
          onProcessComplete={handleProcessComplete}
          onError={handleError}
          onUploadStart={handleUploadStart}
          loading={loading}
        />

        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {results && <ResultsDisplay results={results} />}
      </div>
    </div>
  );
}

export default App;





