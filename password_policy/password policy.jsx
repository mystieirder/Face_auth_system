import React, { useState } from 'react';

const PasswordInput = () => {
  const [password, setPassword] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);

  // Validation rules according to FR-1.4
  const checks = {
    length: password.length >= 8,
    hasLetter: /[a-zA-Z]/.test(password),
    hasNumber: /\d/.test(password),
    hasSpecialChar: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>/?]/.test(password),
  };

  const isFormValid = Object.values(checks).every(Boolean);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitted(true);
    if (isFormValid) {
      alert('Password successfully set!');
    }
  };

  return (
    <div style={{ maxWidth: '400px', margin: '20px auto', fontFamily: 'sans-serif' }}>
      <h2>Password Registration</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="password" style={{ display: 'block', marginBottom: '5px' }}>
            New Password:
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{
              width: '100%',
              padding: '8px',
              fontSize: '16px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              boxSizing: 'border-box'
            }}
            placeholder="Enter password..."
          />
        </div>

        {/* Validation Checklist UI */}
        <div style={{ background: '#f9f9f9', padding: '12px', borderRadius: '6px', marginBottom: '15px' }}>
          <p style={{ margin: '0 0 8px 0', fontWeight: 'bold', fontSize: '14px' }}>
            Password Requirements:
          </p>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '14px' }}>
            <li style={{ color: checks.length ? 'green' : 'red' }}>
              {checks.length ? '✓' : '✗'} At least 8 characters
            </li>
            <li style={{ color: checks.hasLetter ? 'green' : 'red' }}>
              {checks.hasLetter ? '✓' : '✗'} At least one letter (a-z, A-Z)
            </li>
            <li style={{ color: checks.hasNumber ? 'green' : 'red' }}>
              {checks.hasNumber ? '✓' : '✗'} At least one number (0-9)
            </li>
            <li style={{ color: checks.hasSpecialChar ? 'green' : 'red' }}>
              {checks.hasSpecialChar ? '✓' : '✗'} At least one special character
            </li>
          </ul>
        </div>

        <button
          type="submit"
          disabled={!isFormValid}
          style={{
            width: '100%',
            padding: '10px',
            backgroundColor: isFormValid ? '#007bff' : '#ccc',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: isFormValid ? 'pointer' : 'not-allowed',
            fontSize: '16px'
          }}
        >
          Set Password
        </button>

        {isSubmitted && !isFormValid && (
          <p style={{ color: 'red', fontSize: '14px', marginTop: '10px' }}>
            Please meet all password requirements before submitting.
          </p>
        )}
      </form>
    </div>
  );
};

export default PasswordInput;