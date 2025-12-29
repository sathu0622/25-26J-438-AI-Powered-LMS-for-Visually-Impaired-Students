import React, { useState } from 'react';

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

  const getResourceTypeColor = (type) => {
    switch(type.toLowerCase()) {
      case 'books':
        return 'from-blue-500 to-blue-600';
      case 'magazine':
        return 'from-purple-500 to-purple-600';
      case 'newspapers':
        return 'from-orange-500 to-orange-600';
      default:
        return 'from-gray-500 to-gray-600';
    }
  };

  return (
    <div className="w-full animate-fade-in">
      <div className="text-center mb-8">
        <h2 className="text-4xl font-bold text-white mb-2 drop-shadow-lg">
          📊 Processing Results
        </h2>
        <div className="w-24 h-1 bg-white/30 rounded-full mx-auto"></div>
      </div>

      <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl overflow-hidden">
        {/* Resource Type Badge */}
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Detected Resource Type
          </h3>
          <div className={`
            inline-flex items-center gap-3 px-6 py-4 rounded-xl
            bg-gradient-to-r ${getResourceTypeColor(results.resource_type)}
            text-white shadow-lg
          `}>
            <span className="text-3xl">{getResourceTypeIcon(results.resource_type)}</span>
            <div className="flex flex-col">
              <span className="text-xl font-bold capitalize">
                {getResourceTypeLabel(results.resource_type)}
              </span>
              <span className="text-sm opacity-90">
                {(results.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>
          </div>
        </div>

        {/* Articles Count */}
        {results.num_articles > 1 && (
          <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-primary-50 to-purple-50">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">Articles Found</h3>
                <p className="text-2xl font-bold text-primary-600">{results.num_articles} articles detected</p>
              </div>
            </div>
          </div>
        )}

        {/* Summaries Section */}
        <div className="p-6 border-b border-gray-200">
          <div 
            className="flex items-center justify-between cursor-pointer group"
            onClick={() => toggleSection('summaries')}
          >
            <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
              <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              {results.num_articles > 1 ? 'Summaries' : 'Summary'}
            </h3>
            <button className="text-primary-500 group-hover:text-primary-600 transition-colors">
              <svg 
                className={`w-6 h-6 transform transition-transform ${expandedSections.summaries ? 'rotate-180' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
          
          {expandedSections.summaries && (
            <div className="mt-4 space-y-4 animate-fade-in">
              {results.summaries.map((summary, index) => (
                <div 
                  key={index} 
                  className="bg-gradient-to-r from-primary-50 to-purple-50 rounded-xl p-5 border-l-4 border-primary-500 shadow-sm hover:shadow-md transition-shadow"
                >
                  {results.num_articles > 1 && (
                    <div className="flex items-center gap-2 mb-3">
                      <span className="px-3 py-1 bg-primary-500 text-white rounded-full text-sm font-semibold">
                        Article {index + 1}
                      </span>
                    </div>
                  )}
                  <p className="text-gray-700 leading-relaxed text-base">
                    {summary}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Extracted Text Section */}
        <div className="p-6">
          <div 
            className="flex items-center justify-between cursor-pointer group"
            onClick={() => toggleSection('extractedText')}
          >
            <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
              <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Extracted Text
            </h3>
            <button className="text-primary-500 group-hover:text-primary-600 transition-colors">
              <svg 
                className={`w-6 h-6 transform transition-transform ${expandedSections.extractedText ? 'rotate-180' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
          
          {expandedSections.extractedText && (
            <div className="mt-4 animate-fade-in">
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 max-h-96 overflow-y-auto scrollbar-thin">
                <p className="text-gray-700 leading-relaxed whitespace-pre-wrap break-words text-sm">
                  {results.extracted_text}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ResultsDisplay;


