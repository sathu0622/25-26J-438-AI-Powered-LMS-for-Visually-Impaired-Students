import './styles.css';

const SubsectionCard = ({ subsection, onSelect }) => {
  return (
    <div className="subsection-card" onClick={() => onSelect(subsection)}>
      <div className="subsection-header">
        <h4 className="subsection-title">{subsection.title}</h4>
        <span className="subsection-duration">⏱ {subsection.duration} min</span>
      </div>
      <p className="subsection-description">{subsection.description}</p>
      <div className="subsection-footer">
        <span className="audio-icon">🎧</span>
        <span className="play-icon">▶</span>
      </div>
    </div>
  );
};

export default SubsectionCard;
