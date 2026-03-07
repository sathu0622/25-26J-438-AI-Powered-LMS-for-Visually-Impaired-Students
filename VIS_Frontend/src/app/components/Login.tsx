import React, { useState, useEffect } from 'react';
import { User, Lock, Eye, EyeOff, LogIn, BookOpen, AlertCircle, HelpCircle } from 'lucide-react';
import { useTTS } from '../contexts/TTSContext';

interface LoginProps {
  onLoggedIn: (username: string) => void;
  onSwitchToRegister?: () => void;
}

const Login: React.FC<LoginProps> = ({ onLoggedIn, onSwitchToRegister }) => {
  const { speak, cancel } = useTTS();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [touched, setTouched] = useState({ username: false, password: false });

  // Announce page on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      speak('Welcome to the History Learning System login page. Need help? Get assistance from a guardian or teacher to login. Please enter your username and password to continue.', { interrupt: true });
    }, 500);
    return () => {
      clearTimeout(timer);
      cancel();
    };
  }, []);

  // Validation functions
  const validateUsername = (value: string) => {
    if (!value.trim()) return 'Username is required';
    if (value.length < 3) return 'Username must be at least 3 characters';
    if (value.length > 20) return 'Username must be less than 20 characters';
    return '';
  };

  const validatePassword = (value: string) => {
    if (!value) return 'Password is required';
    if (value.length < 6) return 'Password must be at least 6 characters';
    return '';
  };

  const usernameError = touched.username ? validateUsername(username) : '';
  const passwordError = touched.password ? validatePassword(password) : '';
  const isFormValid = !validateUsername(username) && !validatePassword(password);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouched({ username: true, password: true });
    
    if (!isFormValid) return;
    
    setError('');
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      onLoggedIn(username.trim());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      <div className="w-full max-w-md">
        {/* Logo/Brand Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 mb-4 shadow-lg">
            <BookOpen className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-800">History Learning System</h1>
          <p className="text-gray-500 mt-1">Accessible education for everyone</p>
        </div>

        {/* Login Card */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8">
          <h2 className="text-xl font-semibold text-center text-gray-800 mb-6">Welcome Back</h2>
          
          {/* Guardian Assistance Notice */}
          <div 
            className="flex items-start gap-3 bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 cursor-pointer hover:bg-blue-100 transition-colors"
            tabIndex={0}
            role="button"
            aria-label="Help notice: Get assistance from a guardian or teacher to login"
            onFocus={() => {
              cancel();
              speak('Need help? Get assistance from a guardian or teacher to login.', { interrupt: true });
            }}
            onClick={() => {
              cancel();
              speak('Need help? Get assistance from a guardian or teacher to login.', { interrupt: true });
            }}
          >
            <HelpCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-blue-700">
              <strong>Need help?</strong> Get assistance from a guardian or teacher to login.
            </p>
          </div>
          
          <form onSubmit={handleLogin} className="space-y-5">
            {/* Username Field */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                Username
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className={`h-5 w-5 ${usernameError ? 'text-red-400' : 'text-gray-400'}`} />
                </div>
                <input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  onBlur={() => setTouched(t => ({ ...t, username: true }))}
                  className={`w-full pl-10 pr-4 py-3 border rounded-xl focus:ring-2 focus:border-transparent transition-all outline-none text-gray-800 placeholder-gray-400 ${
                    usernameError 
                      ? 'border-red-300 focus:ring-red-500 bg-red-50' 
                      : 'border-gray-200 focus:ring-blue-500'
                  }`}
                  aria-label="Username input field"
                  aria-invalid={!!usernameError}
                  aria-describedby={usernameError ? 'username-error' : undefined}
                />
              </div>
              {usernameError && (
                <div id="username-error" className="flex items-center gap-1 mt-2 text-sm text-red-600" role="alert">
                  <AlertCircle className="h-4 w-4" />
                  {usernameError}
                </div>
              )}
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className={`h-5 w-5 ${passwordError ? 'text-red-400' : 'text-gray-400'}`} />
                </div>
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  onBlur={() => setTouched(t => ({ ...t, password: true }))}
                  className={`w-full pl-10 pr-12 py-3 border rounded-xl focus:ring-2 focus:border-transparent transition-all outline-none text-gray-800 placeholder-gray-400 ${
                    passwordError 
                      ? 'border-red-300 focus:ring-red-500 bg-red-50' 
                      : 'border-gray-200 focus:ring-blue-500'
                  }`}
                  aria-label="Password input field"
                  aria-invalid={!!passwordError}
                  aria-describedby={passwordError ? 'password-error' : undefined}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(v => !v)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 transition-colors"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {passwordError && (
                <div id="password-error" className="flex items-center gap-1 mt-2 text-sm text-red-600" role="alert">
                  <AlertCircle className="h-4 w-4" />
                  {passwordError}
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-xl text-sm" role="alert">
                <AlertCircle className="h-5 w-5 flex-shrink-0" />
                {error}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  <LogIn className="h-5 w-5" />
                  Sign In
                </>
              )}
            </button>
          </form>

          {/* Register Link */}
          <div className="mt-6 text-center">
            <span className="text-gray-500">Don't have an account? </span>
            <button
              type="button"
              onClick={onSwitchToRegister}
              className="text-blue-600 hover:text-blue-700 font-semibold hover:underline transition-colors"
            >
              Create Account
            </button>
          </div>
        </div>

        {/* Accessibility Note */}
        <p className="text-center text-gray-400 text-sm mt-6">
          Designed for accessibility. Use keyboard navigation for best experience.
        </p>
      </div>
    </div>
  );
};

export default Login;
