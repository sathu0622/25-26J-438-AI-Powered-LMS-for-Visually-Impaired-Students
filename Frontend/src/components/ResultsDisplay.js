import React, { useState } from 'react';
import './ResultsDisplay.css';

function ResultsDisplay({ results }) {
  const [expandedSections, setExpandedSections] = useState({
    extractedText: false,
    summaries: true
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const getResourceTypeIcon = (type) => {
    switch(type.toLowerCase()) {
      case 'books':
        return '📚';
      case 'magazine':
        return '📖';
      case 'newspapers':
        return '📰';
      default:
        return '📄';
    }
  };

  const getResourceTypeLabel = (type) => {
    switch(type.toLowerCase()) {
      case 'books':
        return 'Book';
      case 'magazine':
        return 'Magazine';
      case 'newspapers':
        return 'Newspaper';
      default:
        return type;
    }
  };

  return (
    <div className="results-container">
      <div className="results-header">
        <h2>📊 Processing Results</h2>
      </div>

      <div className="result-card">
        <div className="result-section">
          <h3>Resource Type</h3>
          <div className="type-badge">
            <span className="type-icon">{getResourceTypeIcon(results.resource_type)}</span>
            <span className="type-label">{getResourceTypeLabel(results.resource_type)}</span>
            <span className="confidence">({(results.confidence * 100).toFixed(1)}% confidence)</span>
          </div>
        </div>

        {results.num_articles > 1 && (
          <div className="result-section">
            <h3>Articles Found</h3>
            <p className="article-count">{results.num_articles} articles detected</p>
          </div>
        )}

        <div className="result-section">
          <div className="section-header" onClick={() => toggleSection('summaries')}>
            <h3>
              {results.num_articles > 1 ? 'Summaries' : 'Summary'}
              <span className="toggle-icon">{expandedSections.summaries ? '▼' : '▶'}</span>
            </h3>
          </div>
          {expandedSections.summaries && (
            <div className="summaries-container">
              {results.summaries.map((summary, index) => (
                <div key={index} className="summary-item">
                  {results.num_articles > 1 && (
                    <div className="summary-header">Article {index + 1}</div>
                  )}
                  <p className="summary-text">{summary}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="result-section">
          <div className="section-header" onClick={() => toggleSection('extractedText')}>
            <h3>
              Extracted Text
              <span className="toggle-icon">{expandedSections.extractedText ? '▼' : '▶'}</span>
            </h3>
          </div>
          {expandedSections.extractedText && (
            <div className="extracted-text-container">
              <p className="extracted-text">{results.extracted_text}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ResultsDisplay;





