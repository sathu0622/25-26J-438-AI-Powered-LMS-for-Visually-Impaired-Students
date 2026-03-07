import React, { useState, useEffect } from 'react';
import { User, Lock, Eye, EyeOff, UserPlus, BookOpen, AlertCircle, CheckCircle, HelpCircle } from 'lucide-react';
import { useTTS } from '../contexts/TTSContext';

interface RegisterProps {
  onRegistered: (username: string) => void;
  onSwitchToLogin?: () => void;
}

const Register: React.FC<RegisterProps> = ({ onRegistered, onSwitchToLogin }) => {
  const { speak, cancel } = useTTS();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [touched, setTouched] = useState({ username: false, password: false, confirmPassword: false });

  // Announce page on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      speak('Welcome to the History Learning System registration page. Need help? Get assistance from a guardian or teacher to register. Please create a username and password to get started.', { interrupt: true });
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
    if (!/^[a-zA-Z0-9_]+$/.test(value)) return 'Username can only contain letters, numbers, and underscores';
    return '';
  };

  const validatePassword = (value: string) => {
    if (!value) return 'Password is required';
    if (value.length < 6) return 'Password must be at least 6 characters';
    if (!/[A-Z]/.test(value)) return 'Password must contain at least one uppercase letter';
    if (!/[a-z]/.test(value)) return 'Password must contain at least one lowercase letter';
    if (!/[0-9]/.test(value)) return 'Password must contain at least one number';
    return '';
  };

  const validateConfirmPassword = (value: string) => {
    if (!value) return 'Please confirm your password';
    if (value !== password) return 'Passwords do not match';
    return '';
  };

  // Password strength indicator
  const getPasswordStrength = (value: string) => {
    if (!value) return { strength: 0, label: '', color: '' };
    let strength = 0;
    if (value.length >= 6) strength++;
    if (value.length >= 8) strength++;
    if (/[A-Z]/.test(value)) strength++;
    if (/[a-z]/.test(value)) strength++;
    if (/[0-9]/.test(value)) strength++;
    if (/[^a-zA-Z0-9]/.test(value)) strength++;

    if (strength <= 2) return { strength: 1, label: 'Weak', color: 'bg-red-500' };
    if (strength <= 4) return { strength: 2, label: 'Medium', color: 'bg-yellow-500' };
    return { strength: 3, label: 'Strong', color: 'bg-green-500' };
  };

  const usernameError = touched.username ? validateUsername(username) : '';
  const passwordError = touched.password ? validatePassword(password) : '';
  const confirmPasswordError = touched.confirmPassword ? validateConfirmPassword(confirmPassword) : '';
  const passwordStrength = getPasswordStrength(password);
  const isFormValid = !validateUsername(username) && !validatePassword(password) && !validateConfirmPassword(confirmPassword);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouched({ username: true, password: true, confirmPassword: true });
    
    if (!isFormValid) return;
    
    setError('');
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Registration failed');
      onRegistered(username.trim());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-indigo-50 to-blue-50 p-4">
      <div className="w-full max-w-md">
        {/* Logo/Brand Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 mb-4 shadow-lg">
            <BookOpen className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-800">History Learning System</h1>
          <p className="text-gray-500 mt-1">Start your learning journey today</p>
        </div>

        {/* Register Card */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8">
          <h2 className="text-xl font-semibold text-center text-gray-800 mb-6">Create Your Account</h2>
          
          {/* Guardian Assistance Notice */}
          <div 
            className="flex items-start gap-3 bg-purple-50 border border-purple-200 rounded-xl p-4 mb-6 cursor-pointer hover:bg-purple-100 transition-colors"
            tabIndex={0}
            role="button"
            aria-label="Help notice: Get assistance from a guardian or teacher to register"
            onFocus={() => {
              cancel();
              speak('Need help? Get assistance from a guardian or teacher to register.', { interrupt: true });
            }}
            onClick={() => {
              cancel();
              speak('Need help? Get assistance from a guardian or teacher to register.', { interrupt: true });
            }}
          >
            <HelpCircle className="h-5 w-5 text-purple-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-purple-700">
              <strong>Need help?</strong> Get assistance from a guardian or teacher to register.
            </p>
          </div>
          
          <form onSubmit={handleRegister} className="space-y-5">
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
                  placeholder="Choose a username"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  onBlur={() => setTouched(t => ({ ...t, username: true }))}
                  className={`w-full pl-10 pr-4 py-3 border rounded-xl focus:ring-2 focus:border-transparent transition-all outline-none text-gray-800 placeholder-gray-400 ${
                    usernameError 
                      ? 'border-red-300 focus:ring-red-500 bg-red-50' 
                      : 'border-gray-200 focus:ring-purple-500'
                  }`}
                  aria-label="Username input field"
                  aria-invalid={!!usernameError}
                  aria-describedby={usernameError ? 'username-error' : 'username-hint'}
                />
              </div>
              {usernameError ? (
                <div id="username-error" className="flex items-center gap-1 mt-2 text-sm text-red-600" role="alert">
                  <AlertCircle className="h-4 w-4" />
                  {usernameError}
                </div>
              ) : (
                <p id="username-hint" className="mt-2 text-xs text-gray-500">
                  3-20 characters. Letters, numbers, and underscores only.
                </p>
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
                  placeholder="Create a password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  onBlur={() => setTouched(t => ({ ...t, password: true }))}
                  className={`w-full pl-10 pr-12 py-3 border rounded-xl focus:ring-2 focus:border-transparent transition-all outline-none text-gray-800 placeholder-gray-400 ${
                    passwordError 
                      ? 'border-red-300 focus:ring-red-500 bg-red-50' 
                      : 'border-gray-200 focus:ring-purple-500'
                  }`}
                  aria-label="Password input field"
                  aria-invalid={!!passwordError}
                  aria-describedby={passwordError ? 'password-error' : 'password-requirements'}
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
              {passwordError ? (
                <div id="password-error" className="flex items-center gap-1 mt-2 text-sm text-red-600" role="alert">
                  <AlertCircle className="h-4 w-4" />
                  {passwordError}
                </div>
              ) : password && (
                <div className="mt-2">
                  {/* Password Strength Bar */}
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-300 ${passwordStrength.color}`}
                        style={{ width: `${(passwordStrength.strength / 3) * 100}%` }}
                      />
                    </div>
                    <span className={`text-xs font-medium ${
                      passwordStrength.strength === 1 ? 'text-red-600' : 
                      passwordStrength.strength === 2 ? 'text-yellow-600' : 'text-green-600'
                    }`}>
                      {passwordStrength.label}
                    </span>
                  </div>
                  {/* Password Requirements */}
                  <div id="password-requirements" className="mt-2 grid grid-cols-2 gap-1">
                    <div className={`flex items-center gap-1 text-xs ${password.length >= 6 ? 'text-green-600' : 'text-gray-400'}`}>
                      <CheckCircle className="h-3 w-3" /> 6+ characters
                    </div>
                    <div className={`flex items-center gap-1 text-xs ${/[A-Z]/.test(password) ? 'text-green-600' : 'text-gray-400'}`}>
                      <CheckCircle className="h-3 w-3" /> Uppercase
                    </div>
                    <div className={`flex items-center gap-1 text-xs ${/[a-z]/.test(password) ? 'text-green-600' : 'text-gray-400'}`}>
                      <CheckCircle className="h-3 w-3" /> Lowercase
                    </div>
                    <div className={`flex items-center gap-1 text-xs ${/[0-9]/.test(password) ? 'text-green-600' : 'text-gray-400'}`}>
                      <CheckCircle className="h-3 w-3" /> Number
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password Field */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className={`h-5 w-5 ${confirmPasswordError ? 'text-red-400' : 'text-gray-400'}`} />
                </div>
                <input
                  id="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  placeholder="Re-enter your password"
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  onBlur={() => setTouched(t => ({ ...t, confirmPassword: true }))}
                  className={`w-full pl-10 pr-12 py-3 border rounded-xl focus:ring-2 focus:border-transparent transition-all outline-none text-gray-800 placeholder-gray-400 ${
                    confirmPasswordError 
                      ? 'border-red-300 focus:ring-red-500 bg-red-50' 
                      : confirmPassword && !confirmPasswordError
                        ? 'border-green-300 focus:ring-green-500 bg-green-50'
                        : 'border-gray-200 focus:ring-purple-500'
                  }`}
                  aria-label="Confirm password input field"
                  aria-invalid={!!confirmPasswordError}
                  aria-describedby={confirmPasswordError ? 'confirm-password-error' : undefined}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(v => !v)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 transition-colors"
                  aria-label={showConfirmPassword ? 'Hide confirm password' : 'Show confirm password'}
                >
                  {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {confirmPasswordError ? (
                <div id="confirm-password-error" className="flex items-center gap-1 mt-2 text-sm text-red-600" role="alert">
                  <AlertCircle className="h-4 w-4" />
                  {confirmPasswordError}
                </div>
              ) : confirmPassword && !validateConfirmPassword(confirmPassword) && (
                <div className="flex items-center gap-1 mt-2 text-sm text-green-600">
                  <CheckCircle className="h-4 w-4" />
                  Passwords match
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
              className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Creating account...
                </>
              ) : (
                <>
                  <UserPlus className="h-5 w-5" />
                  Create Account
                </>
              )}
            </button>
          </form>

          {/* Login Link */}
          <div className="mt-6 text-center">
            <span className="text-gray-500">Already have an account? </span>
            <button
              type="button"
              onClick={onSwitchToLogin}
              className="text-purple-600 hover:text-purple-700 font-semibold hover:underline transition-colors"
            >
              Sign In
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

export default Register;
