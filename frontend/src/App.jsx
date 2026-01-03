import { useEffect, useState } from "react";
import api from "./api";
import "./styles.css";

export default function App() {
  const [chapters, setChapters] = useState([]);
  const [chapter, setChapter] = useState("");
  const [currentData, setCurrentData] = useState(null);
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.get("/chapters").then(res => {
      setChapters(res.data.chapters);
    });
  }, []);

  const startQuiz = async () => {
    if (!chapter) return alert("Please select a chapter!");
    await nextQuestion();
  };

  const nextQuestion = async () => {
    setLoading(true);
    setFeedback(null);
    setAnswer("");

    const res = await api.post("/generate_question", {
      chapter_name: chapter,
    });

    setCurrentData(res.data);
    setLoading(false);
  };

  const submitAnswer = async () => {
    if (!answer) return;

    setLoading(true);
    const res = await api.post("/evaluate_answer", {
      user_answer: answer,
      correct_answer: currentData.correct_answer,
      key_phrase: currentData.key_phrase,
    });

    setFeedback(res.data);
    setLoading(false);
  };

  return (
    <>
      <h1>🎓 History Tutor AI</h1>

      {!currentData && (
        <div className="card">
          <h3>📖 Select a Chapter</h3>
          <select value={chapter} onChange={e => setChapter(e.target.value)}>
            <option value="">-- Choose a Topic --</option>
            {chapters.map(ch => (
              <option key={ch} value={ch}>{ch}</option>
            ))}
          </select>
          <button onClick={startQuiz}>Start Quiz</button>
        </div>
      )}

      {currentData && (
        <div className="card">
          <h3>
            {loading ? (
              <span className="loading">🤖 AI is thinking...</span>
            ) : (
              currentData.question
            )}
          </h3>

          <input
            type="text"
            placeholder="Type your answer here..."
            value={answer}
            onChange={e => setAnswer(e.target.value)}
          />

          {!feedback && (
            <button onClick={submitAnswer}>Submit Answer</button>
          )}

          {feedback && (
            <div
              id="feedback"
              className={feedback.correct ? "correct" : "incorrect"}
            >
              {feedback.feedback}
              <br />
              <small>Correct Answer: {currentData.correct_answer}</small>
            </div>
          )}

          {feedback && (
            <button
              onClick={nextQuestion}
              style={{ background: "#28a745" }}
            >
              Next Question ➡
            </button>
          )}

          <button
            onClick={() => setCurrentData(null)}
            style={{ background: "#6c757d", marginTop: "10px" }}
          >
            Exit to Menu
          </button>
        </div>
      )}
    </>
  );
}
