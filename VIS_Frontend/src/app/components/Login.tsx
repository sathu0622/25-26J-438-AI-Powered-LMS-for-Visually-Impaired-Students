import React, { useState } from 'react';

interface LoginProps {
  onLoggedIn: (username: string) => void;
  onSwitchToRegister?: () => void;
}

const Login: React.FC<LoginProps> = ({ onLoggedIn, onSwitchToRegister }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      onLoggedIn(username);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleLogin} style={{ maxWidth: 350, margin: '2rem auto', padding: 24, borderRadius: 12, boxShadow: '0 2px 12px #0001', background: '#fff' }}>
      <h2 style={{ textAlign: 'center', marginBottom: 16 }}>Login</h2>
      <label style={{ display: 'block', marginBottom: 8 }}>
        Username
        <input
          type="text"
          placeholder="Enter username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          required
          style={{ width: '100%', padding: 8, marginTop: 4, marginBottom: 16, borderRadius: 6, border: '1px solid #ccc' }}
        />
      </label>
      <label style={{ display: 'block', marginBottom: 8 }}>
        Password
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <input
            type={showPassword ? 'text' : 'password'}
            placeholder="Enter password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
            style={{ width: '100%', padding: 8, marginTop: 4, borderRadius: 6, border: '1px solid #ccc' }}
          />
          <button
            type="button"
            onClick={() => setShowPassword(v => !v)}
            style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer' }}
            aria-label={showPassword ? 'Hide password' : 'Show password'}
          >
            {showPassword ? '🙈' : '👁️'}
          </button>
        </div>
      </label>
      <button type="submit" style={{ width: '100%', padding: 10, borderRadius: 6, background: '#007bff', color: '#fff', border: 'none', fontWeight: 600, marginTop: 12 }} disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
      <div style={{ marginTop: 16, textAlign: 'center' }}>
        Don't have an account?{' '}
        <button type="button" style={{ color: '#007bff', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }} onClick={onSwitchToRegister}>
          Register
        </button>
      </div>
      {error && <div style={{ color: 'red', marginTop: 12 }}>{error}</div>}
    </form>
  );
};

export default Login;
