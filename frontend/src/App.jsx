import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [question, setQuestion] = useState('');
  const [studentAnswer, setStudentAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleEvaluate = async () => {
    if (!question.trim() || !studentAnswer.trim()) {
      setError('Please enter both question and student answer');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/evaluate`, {
        question: question.trim(),
        student_answer: studentAnswer.trim(),
      });

      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'An error occurred while evaluating the answer'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setQuestion('');
    setStudentAnswer('');
    setResult(null);
    setError(null);
  };

  const getScoreColor = (score) => {
    if (score >= 60) return '#4caf50';
    if (score >= 50) return '#ff9800';
    return '#f44336';
  };

  const getStatusColor = (status) => {
    if (status === 'PASS') return '#4caf50';
    if (status === 'NEEDS IMPROVEMENT') return '#ff9800';
    return '#f44336';
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🎓 O/L History Answer Evaluation System</h1>
          <p>Evaluate student answers using AI-powered assessment</p>
        </header>

        <div className="form-section">
          <div className="form-group">
            <label htmlFor="question">📚 Question</label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter the history question here..."
              rows="4"
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="student-answer">✍️ Student Answer</label>
            <textarea
              id="student-answer"
              value={studentAnswer}
              onChange={(e) => setStudentAnswer(e.target.value)}
              placeholder="Enter the student's answer here..."
              rows="6"
              disabled={loading}
            />
          </div>

          <div className="button-group">
            <button
              onClick={handleEvaluate}
              disabled={loading}
              className="btn-evaluate"
            >
              {loading ? '⏳ Evaluating...' : '✅ Evaluate Answer'}
            </button>
            <button
              onClick={handleReset}
              disabled={loading}
              className="btn-reset"
            >
              🔄 Reset
            </button>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <strong>❌ Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="results-section">
            <h2>📊 Evaluation Results</h2>

            {/* FINAL SCORE ONLY */}
            <div
              className="score-card"
              style={{ borderColor: getScoreColor(result.final_score) }}
            >
              <div className="score-main">
                <span className="score-label">Final Score</span>
                <span
                  className="score-value"
                  style={{ color: getScoreColor(result.final_score) }}
                >
                  {result.final_score}%
                </span>
              </div>

              <div
                className="status-badge"
                style={{ backgroundColor: getStatusColor(result.status) }}
              >
                {result.status}
              </div>
            </div>

            {/* FEEDBACK */}
            <div className="feedback-section">
              <h3>💬 Feedback</h3>
              <div className="feedback-content">
                {result.feedback.split('\n').map((line, idx) => (
                  <p
                    key={idx}
                    className={
                      line.trim().startsWith('❗') ||
                      line.trim().startsWith('✔')
                        ? 'feedback-highlight'
                        : ''
                    }
                  >
                    {line}
                  </p>
                ))}
              </div>
            </div>

            {/* MODEL ANSWER ONLY IF NOT PASS */}
            {result.status !== 'PASS' && (
              <div className="model-answer-section">
                <h3>📝 Model Answer (Reference)</h3>
                <div className="model-answer-content">
                  {result.model_answer}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
