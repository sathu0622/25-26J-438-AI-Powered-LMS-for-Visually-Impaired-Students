import React, { useState, useEffect } from 'react';

function ResultsDisplay({ results, onArticleSelect, onAskQuestion, loading, activeTab, onTabChange }) {
  const [expandedSections, setExpandedSections] = useState({
    extractedText: false,
    summaries: true,
    articles: false,
    qa: false
  });

  const [selectedArticleId, setSelectedArticleId] = useState(null);
  const [question, setQuestion] = useState('');
  const [qaHistory, setQaHistory] = useState([]);
  const [isAsking, setIsAsking] = useState(false);

  useEffect(() => {
    // Initialize with main article
    if (results?.article_list?.[0]?.article_id && !selectedArticleId) {
      setSelectedArticleId(results.article_list[0].article_id);
    }
  }, [results]);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const getResourceTypeIcon = (type) => {
    switch(type?.toLowerCase()) {
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
    switch(type?.toLowerCase()) {
      case 'books':
        return 'Book';
      case 'magazine':
        return 'Magazine';
      case 'newspapers':
        return 'Newspaper';
      default:
        return type || 'Unknown';
    }
  };

  const getResourceTypeColor = (type) => {
    switch(type?.toLowerCase()) {
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

  const handleArticleClick = async (articleId, heading) => {
    if (loading) return;
    
    setSelectedArticleId(articleId);
    await onArticleSelect(articleId);
  };

// In ResultsDisplay.js, update the handleAskClick function:
const handleAskClick = async () => {
  if (!question.trim() || !selectedArticleId || loading || isAsking) return;
  
  setIsAsking(true);
  try {
    const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/ask-question`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        document_id: results.document_id,
        article_id: selectedArticleId,
        question: question,
        max_answer_len: 64,
        score_threshold: 0.15
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get answer');
    }
    
    const qaData = await response.json();
    
    // Check if it's an error response
    if (qaData.error) {
      setError(qaData.error + (qaData.suggestion ? ` - ${qaData.suggestion}` : ''));
      return null;
    }
    
    const selectedArticle = results.article_list?.find(a => a.article_id === selectedArticleId);
    
    setQaHistory(prev => [{
      id: Date.now(),
      question: question,
      answer: qaData.answer,
      confidence: qaData.confidence,
      articleHeading: selectedArticle?.heading || qaData.article_heading || 'Unknown Article',
      articleId: selectedArticleId,
      timestamp: new Date().toLocaleTimeString(),
      context: qaData.context_preview,
      suggestion: qaData.suggestion
    }, ...prev]);
    
    setQuestion('');
    return qaData;
    
  } catch (err) {
    setError(err.message || 'Failed to get answer. Please try rephrasing your question.');
    return null;
  } finally {
    setIsAsking(false);
  }
};

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskClick();
    }
  };

  const renderSummaries = () => {
    if (!results.summaries || !Array.isArray(results.summaries)) {
      return (
        <div className="bg-gradient-to-r from-primary-50 to-purple-50 rounded-xl p-5 border-l-4 border-primary-500 shadow-sm">
          <p className="text-gray-700">No summaries available</p>
        </div>
      );
    }

    return results.summaries.map((summary, index) => (
      <div 
        key={index} 
        className="bg-gradient-to-r from-primary-50 to-purple-50 rounded-xl p-5 border-l-4 border-primary-500 shadow-sm hover:shadow-md transition-shadow"
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 bg-primary-500 text-white rounded-full text-sm font-semibold">
              {summary.article_id || `Article ${index + 1}`}
            </span>
            {summary.column && summary.column !== 'full' && (
              <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                Column: {summary.column}
              </span>
            )}
          </div>
          <div className="text-sm text-gray-500">
            {summary.word_count && `${summary.word_count} words`}
          </div>
        </div>
        
        <div className="space-y-3">
          {summary.heading && (
            <h4 className="font-bold text-gray-800 text-lg">{summary.heading}</h4>
          )}
          
          {summary.subheading && (
            <p className="text-gray-600 text-sm italic">{summary.subheading}</p>
          )}
          
          <div className="bg-white p-4 rounded-lg border">
            <p className="text-gray-700 leading-relaxed text-base">
              {summary.summary || 'No summary available'}
            </p>
          </div>
        </div>
      </div>
    ));
  };

  const renderArticleList = () => {
    if (!results.article_list || !Array.isArray(results.article_list)) {
      return (
        <div className="text-center py-8 text-gray-500">
          No articles available for selection
        </div>
      );
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        {results.article_list.map((article) => (
          <div
            key={article.article_id}
            className={`border rounded-xl p-4 cursor-pointer transition-all duration-200 hover:shadow-md ${
              selectedArticleId === article.article_id
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-200 hover:border-primary-300'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={() => handleArticleClick(article.article_id, article.heading)}
          >
            <div className="flex items-start gap-3">
              <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                selectedArticleId === article.article_id
                  ? 'bg-primary-500 text-white'
                  : 'bg-gray-100 text-gray-600'
              }`}>
                {article.index}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-semibold text-gray-800 truncate">
                    {article.heading || `Article ${article.index}`}
                  </h4>
                  {article.is_main_article && (
                    <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded-full">
                      Main
                    </span>
                  )}
                </div>
                
                {article.subheading && (
                  <p className="text-sm text-gray-600 truncate mb-2">
                    {article.subheading}
                  </p>
                )}
                
                <div className="flex items-center gap-3 text-xs text-gray-500">
                  {article.column && article.column !== 'full' && (
                    <span className="flex items-center gap-1">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                      </svg>
                      {article.column}
                    </span>
                  )}
                  
                  {article.word_count > 0 && (
                    <span className="flex items-center gap-1">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      {article.word_count} words
                    </span>
                  )}
                </div>
                
                {article.body_preview && (
                  <p className="text-sm text-gray-500 mt-2 line-clamp-2">
                    {article.body_preview}
                  </p>
                )}
              </div>
              
              <div className="flex flex-col gap-2">
                <button
                  className={`flex-shrink-0 px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                    selectedArticleId === article.article_id
                      ? 'bg-primary-500 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleArticleClick(article.article_id, article.heading);
                    onTabChange('summary');
                  }}
                  disabled={loading}
                >
                  Summarize
                </button>
                
                <button
                  className="flex-shrink-0 px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-700 hover:bg-green-200 transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedArticleId(article.article_id);
                    onTabChange('qa');
                  }}
                  disabled={loading}
                >
                  Ask Q&A
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderQATab = () => {
    const selectedArticle = results.article_list?.find(a => a.article_id === selectedArticleId);
    
    return (
      <div className="space-y-6">
        {/* Selected Article Info */}
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-gray-800">Question & Answer</h3>
              <p className="text-sm text-gray-600">
                Asking about: <span className="font-medium">{selectedArticle?.heading || 'Unknown Article'}</span>
              </p>
            </div>
          </div>
        </div>

        {/* Question Input */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
          <div className="p-4 border-b">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Ask a question about this article:
            </label>
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your question here... (e.g., 'What is the main topic?', 'Who are the key people mentioned?')"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                  rows="3"
                  disabled={loading || isAsking}
                />
                <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                  Press Enter to ask
                </div>
              </div>
              <button
                onClick={handleAskClick}
                disabled={!question.trim() || loading || isAsking}
                className={`self-end px-6 py-3 rounded-lg font-medium transition-colors ${
                  !question.trim() || loading || isAsking
                    ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                    : 'bg-green-500 text-white hover:bg-green-600'
                }`}
              >
                {isAsking ? (
                  <span className="flex items-center gap-2">
                    <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span>
                    Asking...
                  </span>
                ) : (
                  'Ask Question'
                )}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Example questions: What is this about? Who is mentioned? When did this happen? Where did this occur? Why is this important?
            </p>
          </div>

          {/* Q&A History */}
          <div className="p-4">
            {qaHistory.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <svg className="w-12 h-12 mx-auto text-gray-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                <p>Ask a question to get started!</p>
                <p className="text-sm mt-1">Answers will appear here.</p>
              </div>
            ) : (
              <div className="space-y-4">
                <h4 className="font-medium text-gray-700">Previous Questions & Answers:</h4>
                {qaHistory.map((qa) => (
                  <div key={qa.id} className="border border-gray-200 rounded-lg overflow-hidden">
                    <div className="bg-gray-50 p-3 border-b">
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full">
                              Q
                            </span>
                            <span className="font-medium text-gray-800">{qa.question}</span>
                          </div>
                          <p className="text-xs text-gray-500">
                            About: {qa.articleHeading} • {qa.timestamp}
                          </p>
                        </div>
                        <span className={`px-2 py-0.5 text-xs rounded-full ${
                          qa.confidence > 0.7 ? 'bg-green-100 text-green-800' :
                          qa.confidence > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {Math.round(qa.confidence * 100)}% confidence
                        </span>
                      </div>
                    </div>
                    <div className="p-4 bg-white">
                      <div className="flex items-start gap-2">
                        <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full mt-1">
                          A
                        </span>
                        <div className="flex-1">
                          <p className="text-gray-700">{qa.answer}</p>
                          {qa.context && (
                            <div className="mt-3 p-2 bg-gray-50 rounded border border-gray-200">
                              <p className="text-xs text-gray-500 mb-1">Context used:</p>
                              <p className="text-sm text-gray-600">{qa.context}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
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
            bg-gradient-to-r ${getResourceTypeColor(results?.resource_type)}
            text-white shadow-lg
          `}>
            <span className="text-3xl">{getResourceTypeIcon(results?.resource_type)}</span>
            <div className="flex flex-col">
              <span className="text-xl font-bold capitalize">
                {getResourceTypeLabel(results?.resource_type)}
              </span>
              <span className="text-sm opacity-90">
                {results?.confidence ? `${(results.confidence * 100).toFixed(1)}% confidence` : 'Unknown confidence'}
              </span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="flex">
            <button
              onClick={() => onTabChange('summary')}
              className={`flex-1 py-4 text-center font-medium transition-colors ${
                activeTab === 'summary'
                  ? 'text-primary-600 border-b-2 border-primary-500'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Summary
              </div>
            </button>
            <button
              onClick={() => onTabChange('qa')}
              className={`flex-1 py-4 text-center font-medium transition-colors ${
                activeTab === 'qa'
                  ? 'text-green-600 border-b-2 border-green-500'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                Ask Questions
              </div>
            </button>
          </nav>
        </div>

        {/* Content based on active tab */}
        <div className="p-6">
          {activeTab === 'summary' ? (
            <div className="space-y-6">
              {/* Articles Count */}
              {results?.num_articles > 1 && (
                <div className="bg-gradient-to-r from-primary-50 to-purple-50 rounded-xl p-6">
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

              {/* Article Selection */}
              <div>
                <div 
                  className="flex items-center justify-between cursor-pointer group"
                  onClick={() => toggleSection('articles')}
                >
                  <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                    <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                    </svg>
                    Select Article to Summarize
                  </h3>
                  <button className="text-primary-500 group-hover:text-primary-600 transition-colors">
                    <svg 
                      className={`w-6 h-6 transform transition-transform ${expandedSections.articles ? 'rotate-180' : ''}`} 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                </div>
                
                {expandedSections.articles && (
                  <div className="mt-4 animate-fade-in">
                    {renderArticleList()}
                  </div>
                )}
              </div>

              {/* Generated Summary */}
              <div>
                <div 
                  className="flex items-center justify-between cursor-pointer group"
                  onClick={() => toggleSection('summaries')}
                >
                  <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                    <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Generated Summary
                    {selectedArticleId && (
                      <span className="text-sm font-normal text-primary-600">
                        (Selected Article)
                      </span>
                    )}
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
                    {renderSummaries()}
                  </div>
                )}
              </div>
            </div>
          ) : (
            // Q&A Tab
            renderQATab()
          )}

          {/* Extracted Text Preview (Always available) */}
          {results?.extracted_text_preview && (
            <div className="mt-8 pt-8 border-t border-gray-200">
              <div 
                className="flex items-center justify-between cursor-pointer group"
                onClick={() => toggleSection('extractedText')}
              >
                <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Full Text Preview
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
                      {results.extracted_text_preview}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ResultsDisplay;