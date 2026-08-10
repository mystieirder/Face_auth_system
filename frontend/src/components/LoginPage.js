import React, { useState } from 'react';
import { loginUser } from '../services/authService';
import FaceLoginCapture from './FaceLoginCapture';

const styles = {
  page: {
    minHeight: '100vh',
    background: '#f3f4f6',
    padding: '40px 16px',
    fontFamily: "'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  },
  card: {
    maxWidth: 480,
    margin: '0 auto',
    background: '#ffffff',
    borderRadius: 16,
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
    padding: '36px 32px',
  },
  heading: {
    fontSize: 26,
    fontWeight: 700,
    marginBottom: 4,
    color: '#111827',
    textAlign: 'center',
  },
  subheading: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24,
  },
  tabRow: {
    display: 'flex',
    borderRadius: 10,
    background: '#f3f4f6',
    padding: 4,
    marginBottom: 24,
  },
  tabBtn: (active) => ({
    flex: 1,
    padding: '9px 0',
    border: 'none',
    borderRadius: 8,
    background: active ? '#ffffff' : 'transparent',
    color: active ? '#111827' : '#6b7280',
    fontWeight: 600,
    fontSize: 13,
    cursor: 'pointer',
    boxShadow: active ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
  }),
  section: { marginBottom: 18 },
  label: {
    display: 'block',
    marginBottom: 6,
    fontSize: 13,
    fontWeight: 600,
    color: '#374151',
  },
  input: {
    width: '100%',
    padding: '11px 12px',
    borderRadius: 10,
    border: '1px solid #d1d5db',
    boxSizing: 'border-box',
    fontSize: 14,
    outline: 'none',
  },
  loginButton: (ready) => ({
    width: '100%',
    padding: 13,
    background: ready ? '#2563eb' : '#d1d5db',
    color: '#fff',
    border: 'none',
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 600,
    cursor: ready ? 'pointer' : 'not-allowed',
  }),
  message: {
    marginTop: 16,
    textAlign: 'center',
    fontSize: 13,
    color: '#374151',
  },
  fallbackNote: {
    fontSize: 12,
    color: '#9a6b52',
    background: '#fdf6f2',
    border: '1px solid #eee5dc',
    borderRadius: 8,
    padding: '8px 10px',
    marginBottom: 16,
  },
};

/**
 * FR-2.1: Dual Authentication Paths (facial vs credential), selectable
 * via tabs — both equally available, neither is "primary".
 * FR-2.3: Automatic Fallback — if facial auth is unavailable (camera
 * denied, model load failure), the UI drops to the password tab on
 * its own, keeping whatever username/email the person already typed.
 */
export default function LoginPage({ onLoginSuccess, onBackToLanding }) {
  const [mode, setMode] = useState('password'); // 'password' | 'face'
  const [emailOrUsername, setEmailOrUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [fallbackTriggered, setFallbackTriggered] = useState(false);

  const handleCameraUnavailable = () => {
    setFallbackTriggered(true);
    setMode('password');
  };

  const handlePasswordLogin = async (e) => {
    e.preventDefault();
    if (!emailOrUsername || !password) {
      setMessage('Enter your username/email and password.');
      return;
    }
    setLoading(true);
    setMessage('');
    try {
      const data = await loginUser({ emailOrUsername, password });
      setMessage(data.message);
      onLoginSuccess && onLoginSuccess(data);
    } catch (err) {
      setMessage(err.response?.data?.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFaceVerified = async (faceEmbedding, livenessVerified) => {
    if (!emailOrUsername) {
      setMessage('Enter your username or email first, then verify your face.');
      return;
    }
    setLoading(true);
    setMessage('');
    try {
      const data = await loginUser({ emailOrUsername, faceEmbedding, livenessVerified });
      setMessage(data.message);
      onLoginSuccess && onLoginSuccess(data);
    } catch (err) {
      setMessage(err.response?.data?.message || 'Face login failed — try your password instead');
      setMode('password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2 style={styles.heading}>Welcome back</h2>
        <p style={styles.subheading}>Sign in with your face or your password</p>

        {fallbackTriggered && (
          <div style={styles.fallbackNote}>
            Camera unavailable — switched to password sign-in automatically.
          </div>
        )}

        <div style={styles.tabRow}>
          <button type="button" style={styles.tabBtn(mode === 'password')} onClick={() => setMode('password')}>
            Password
          </button>
          <button type="button" style={styles.tabBtn(mode === 'face')} onClick={() => setMode('face')}>
            Face login
          </button>
        </div>

        <div style={styles.section}>
          <label style={styles.label}>Username or email</label>
          <input
            type="text"
            value={emailOrUsername}
            onChange={(e) => setEmailOrUsername(e.target.value)}
            placeholder="Enter username or email…"
            style={styles.input}
          />
        </div>

        {mode === 'password' ? (
          <form onSubmit={handlePasswordLogin}>
            <div style={styles.section}>
              <label style={styles.label}>Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password…"
                style={styles.input}
              />
            </div>
            <button type="submit" disabled={loading} style={styles.loginButton(!loading)}>
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
        ) : (
          <div style={styles.section}>
            <FaceLoginCapture onVerified={handleFaceVerified} onCameraUnavailable={handleCameraUnavailable} />
          </div>
        )}

        {message && <p style={styles.message}>{message}</p>}

        {onBackToLanding && (
          <p style={{ ...styles.message, marginTop: 20 }}>
            <button type="button" onClick={onBackToLanding} style={{ background: 'none', border: 'none', color: '#2563eb', cursor: 'pointer', fontWeight: 600 }}>
              ← Back
            </button>
          </p>
        )}
      </div>
    </div>
  );
}
