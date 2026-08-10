import React, { useState } from 'react';
import RegisterPage from './components/RegisterPage';
import LoginPage from './components/LoginPage';
import ProfilePage from './components/ProfilePage';

function LandingPage({ onGetStarted, onLogin }) {
  return (
    <div className="landing">
      <header className="landing-header">
        <div className="brand">
          <span className="brand-icon">🛡️</span>
          <div>
            <div className="brand-name">Inclusive Profile</div>
            <div className="brand-tagline">Accessible & multimodal</div>
          </div>
        </div>
        <nav className="landing-nav">
          <button className="link-btn" onClick={onLogin}>Log in</button>
          <button className="primary-btn small" onClick={onGetStarted}>Create account</button>
        </nav>
      </header>

      <main className="hero">
        <div className="hero-copy">
          <div className="eyebrow">● ACCESSIBILITY-FIRST PWA</div>
          <h1>
            A profile that works<br />
            <span className="accent">the way you do.</span>
          </h1>
          <p className="hero-sub">
            Register with a real account, complete your profile by voice or
            keyboard, and try our simulated local face-match login — all in a
            fully accessible, installable web app.
          </p>
          <div className="hero-actions">
            <button className="primary-btn" onClick={onGetStarted}>
              Create your profile →
            </button>
            <button className="secondary-btn" onClick={onLogin}>
              I already have an account
            </button>
          </div>
          <p className="fine-print">
            Works fully with keyboard, screen readers, and reduced-motion settings. Voice
            and camera are optional.
          </p>
        </div>

        <div className="hero-card">
          <h3>What you can do</h3>
          <ul>
            <li>Sign in with a password or a demo face-match</li>
            <li>Fill fields by voice: "next field", "clear field", "submit profile"</li>
            <li>Track completion with clear labels — not color alone</li>
            <li>Capture a profile photo when — and only when — you choose</li>
            <li>Turn on high-contrast, large text, or reduced-motion any time</li>
          </ul>
        </div>
      </main>

      <section className="features">
        <h2>Built for every way you interact</h2>
        <div className="feature-grid">
          <div className="feature-card">
            <div className="feature-icon">🛡️</div>
            <h4>Real accounts</h4>
            <p>Email + password with secure hashing and session cookies. No third-party lock-in.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🎙️</div>
            <h4>Voice-first entry</h4>
            <p>Web Speech API commands you can use instead of, or alongside, typing.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📷</div>
            <h4>Local face-match demo</h4>
            <p>Simulated client-side descriptor. Password is always available as fallback.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⌨️</div>
            <h4>Fully keyboard-friendly</h4>
            <p>Visible focus rings, semantic landmarks, and no color-only status cues.</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState('landing'); // 'landing' | 'register' | 'login' | 'profile'

  if (view === 'register') {
    return (
      <div className="app-container">
        <RegisterPage />
      </div>
    );
  }

  if (view === 'login') {
    return (
      <div className="app-container">
        <LoginPage
          onBackToLanding={() => setView('landing')}
          onLoginSuccess={(data) => {
            localStorage.setItem('authToken', data.token);
            localStorage.setItem('authUser', JSON.stringify(data.user));
            setView('profile');
          }}
        />
      </div>
    );
  }

  if (view === 'profile') {
    const user = JSON.parse(localStorage.getItem('authUser'));
    return (
      <div className="app-container">
        <ProfilePage
          user={user}
          onLogout={() => {
            localStorage.removeItem('authToken');
            localStorage.removeItem('authUser');
            setView('landing');
          }}
        />
      </div>
    );
  }

  return (
    <LandingPage
      onGetStarted={() => setView('register')}
      onLogin={() => setView('login')}
    />
  );
}