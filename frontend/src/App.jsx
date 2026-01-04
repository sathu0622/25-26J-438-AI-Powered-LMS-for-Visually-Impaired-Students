import { useEffect, useState } from "react";
import api from "./api"; 
import "./styles.css";

// --- VIEW 1: TOPIC SELECTION ---
const TopicList = ({ chapters, onSelect }) => {
  return (
    <>
      <header>
        <h1>Select a Topic</h1>
        <button className="header-btn">Help</button>
      </header>
      <div className="topic-container">
        {chapters.map((ch, index) => (
          <div key={ch} className="topic-card" onClick={() => onSelect(ch)}>
            <span className="topic-name">{index + 1}. {ch}</span>
            <span style={{color: '#9ca3af'}}>›</span>
          </div>
        ))}
      </div>
    </>
  );
};

// --- VIEW 2: QUESTION SCREEN (UPDATED) ---
const QuestionView = ({ data, loading, onSubmit, onSkip, onGoHome }) => {
  const [text, setText] = useState("");
  const [isRecording, setIsRecording] = useState(false);

  // Mock Microphone logic
  const toggleMic = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      setTimeout(() => {
        setIsRecording(false);
        setText("The primary purpose of education is to develop individuals intellectually and socially...");
      }, 1500);
    }
  };

  if (loading) return <div className="loading-text">Thinking...</div>;

  return (
    <>
      <header>
        {/* Updated "Back" button to actually go home */}
        <button className="header-btn" onClick={onGoHome}>Back</button>
        <h1>{data.chapter_name || "Quiz"}</h1>
        <div style={{width: 20}}></div> 
      </header>

      <div className="question-container">
        <div>
          <span className="q-badge">Question 1</span>
          <div className="q-content">{data.question}</div>
        </div>

        <div className="mic-section">
          <button 
            className={`mic-button ${isRecording ? 'recording' : ''}`} 
            onClick={toggleMic}
          >
            {isRecording ? '🛑' : '🎙️'}
          </button>
          
          <textarea 
            className="input-box"
            rows="3"
            placeholder="Tap microphone or type your answer..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <div className="action-row">
            <button className="btn btn-skip" onClick={onSkip}>Skip</button>
            <button className="btn btn-submit" onClick={() => onSubmit(text)}>Submit</button>
          </div>

          {/* NEW BUTTON ADDED HERE */}
          <button 
              className="btn btn-skip" 
              style={{ width: '100%', marginTop: '12px' }} 
              onClick={onGoHome}
          >
            Select Another Chapter
          </button>

        </div>
      </div>
    </>
  );
};

// --- VIEW 3: FEEDBACK SCREEN ---
const FeedbackView = ({ result, userAnswer, onNext, onGoHome }) => {
  const isCorrect = result.score >= 60;
  
  return (
    <>
      <header>
        <h1>Feedback</h1>
      </header>
      
      <div className="feedback-container">
        <div className={`feedback-banner ${isCorrect ? 'success' : 'error'}`}>
          <div className="score-circle" style={{background: isCorrect ? '#10b981' : '#ef4444'}}>
            {isCorrect ? '✓' : '✕'}
          </div>
          <h2 className="score-percent">{result.score}%</h2>
          <div className="score-label">{result.feedback}</div>
        </div>

        <div className="answer-block" style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div>
            <span className="block-label" style={{marginBottom: 0}}>Audio Playback</span>
            <span style={{fontSize: 13, color: '#6b7280'}}>Ready to play</span>
          </div>
          <button style={{background: '#1e1b4b', color: '#fff', border: 'none', borderRadius: 4, padding: '5px 10px'}}>▶</button>
        </div>

        <div className="answer-block">
          <span className="block-label">Your Answer</span>
          <div className="text-content">{userAnswer}</div>
        </div>

        <div className="answer-block" style={{borderColor: '#dbeafe', background: '#eff6ff'}}>
          <span className="block-label">Model Answer</span>
          <div className="text-content">{result.correct_answer}</div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <button className="btn btn-submit" style={{width: '100%'}} onClick={onNext}>
            Next Question ➡
            </button>
            
            <button 
                className="btn btn-skip" 
                style={{width: '100%'}} 
                onClick={onGoHome}
            >
            Select Another Chapter
            </button>
        </div>
      </div>
    </>
  );
};


export default function App() {
  const [currentView, setCurrentView] = useState("topics");
  const [chapters, setChapters] = useState([]);
  const [selectedChapter, setSelectedChapter] = useState("");
  const [questionData, setQuestionData] = useState(null);
  const [feedbackData, setFeedbackData] = useState(null);
  const [userAnswer, setUserAnswer] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    api.get("/chapters")
      .then(res => setChapters(res.data.chapters))
      .catch(err => console.error("API Error:", err));
  }, []);

  const startQuiz = async (chapter) => {
    setSelectedChapter(chapter);
    setIsLoading(true);
    setCurrentView("question");

    try {
      const res = await api.post("/generate_question", { chapter_name: chapter });
      setQuestionData({ ...res.data, chapter_name: chapter });
    } catch (err) {
      alert("Error loading question");
      setCurrentView("topics");
    }
    setIsLoading(false);
  };

  const handleAnswerSubmit = async (text) => {
    if (!text.trim()) return;
    setUserAnswer(text);
    setIsLoading(true);

    try {
      const res = await api.post("/evaluate_answer", {
        user_answer: text,
        correct_answer: questionData.correct_answer,
        key_phrase: questionData.key_phrase,
        chapter_name: selectedChapter
      });
      setFeedbackData({ 
        ...res.data, 
        correct_answer: questionData.correct_answer 
      });
      setCurrentView("feedback");
    } catch (err) {
      alert("Error submitting answer");
    }
    setIsLoading(false);
  };

  const handleNextQuestion = () => {
    setFeedbackData(null);
    setUserAnswer("");
    startQuiz(selectedChapter); 
  };

  // Reset state and go back to topics
  const handleGoHome = () => {
    setFeedbackData(null);
    setQuestionData(null);
    setUserAnswer("");
    setCurrentView("topics");
  };

  return (
    <div className="app-root">
      {currentView === "topics" && (
        <TopicList chapters={chapters} onSelect={startQuiz} />
      )}

      {currentView === "question" && questionData && (
        <QuestionView 
          data={questionData} 
          loading={isLoading}
          onSubmit={handleAnswerSubmit}
          onSkip={handleNextQuestion}
          onGoHome={handleGoHome} 
        />
      )}

      {currentView === "feedback" && feedbackData && (
        <FeedbackView 
          result={feedbackData} 
          userAnswer={userAnswer} 
          onNext={handleNextQuestion}
          onGoHome={handleGoHome}
        />
      )}
    </div>
  );
}