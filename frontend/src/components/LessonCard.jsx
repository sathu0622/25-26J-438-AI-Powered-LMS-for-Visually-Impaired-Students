import './styles.css';

const LessonCard = ({ lesson, onSelect, icon }) => {
  return (
    <div className="lesson-card" onClick={() => onSelect(lesson)}>
      <div className="lesson-icon">{icon || '📚'}</div>
      <h3 className="lesson-title">{lesson.title}</h3>
      <p className="lesson-description">{lesson.description}</p>
      <div className="lesson-action">
        <span className="action-arrow">→</span>
      </div>
    </div>
  );
};

export default LessonCard;
