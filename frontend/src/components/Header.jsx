import './styles.css';

const Header = ({ title, subtitle, onBack, showBackButton = true }) => {
  return (
    <header className="app-header">
      <div className="header-content">
        {showBackButton && onBack && (
          <button className="back-button" onClick={onBack} title="Go back">
            ← Back
          </button>
        )}
        <div className="header-text">
          <h1 className="header-title">{title}</h1>
          {subtitle && <p className="header-subtitle">{subtitle}</p>}
        </div>
      </div>
    </header>
  );
};

export default Header;
