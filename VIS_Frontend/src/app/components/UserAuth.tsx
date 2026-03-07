import React, { useState } from 'react';
import Register from './Register';
import Login from './Login';

interface UserAuthProps {
  onAuthSuccess: (username: string) => void;
}

const UserAuth: React.FC<UserAuthProps> = ({ onAuthSuccess }) => {
  const [showRegister, setShowRegister] = useState(true);
  const [username, setUsername] = useState('');

  // After registration, show login
  const handleRegistered = (uname: string) => {
    setShowRegister(false);
    setUsername(uname);
  };

  // After login, notify parent
  const handleLoggedIn = (uname: string) => {
    onAuthSuccess(uname);
  };

  const handleSwitchToLogin = () => setShowRegister(false);
  const handleSwitchToRegister = () => setShowRegister(true);

  return (
    <div>
      {showRegister ? (
        <Register onRegistered={handleRegistered} onSwitchToLogin={handleSwitchToLogin} />
      ) : (
        <Login onLoggedIn={handleLoggedIn} onSwitchToRegister={handleSwitchToRegister} />
      )}
    </div>
  );
};

export default UserAuth;
