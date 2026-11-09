import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import WebcamCapture from './WebcamCapture';

const PLANS = [
  {
    id: 'plan1',
    name: 'Plan 1',
    amount: 100,
    tagline: 'For getting started',
    features: [
      'Biometric face enrollment',
      'Single device access',
      'Standard email support',
    ],
  },
  {
    id: 'plan2',
    name: 'Plan 2',
    amount: 200,
    tagline: 'Most popular',
    highlighted: true,
    features: [
      'Everything in Plan 1',
      'Priority face-match processing',
      'Extended session validity',
      'Priority support',
    ],
  },
  {
    id: 'plan3',
    name: 'Plan 3',
    amount: 300,
    tagline: 'For advanced use',
    features: [
      'Everything in Plan 2',
      'Advanced security checks',
      'Dedicated account support',
    ],
  },
];

const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// Shared styles
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
  wideCard: {
    maxWidth: 760,
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
    marginBottom: 28,
  },
  section: {
    marginBottom: 22,
  },
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
    transition: 'border-color 0.15s ease',
  },
  inputError: {
    border: '1px solid #ef4444',
  },
  errorText: {
    color: '#ef4444',
    fontSize: 12,
    marginTop: 6,
  },
  successText: {
    color: '#16a34a',
    fontSize: 12,
    marginTop: 6,
  },
  passwordBox: {
    background: '#f9fafb',
    padding: 14,
    borderRadius: 10,
    marginTop: 10,
    border: '1px solid #eef0f3',
  },
  passwordBoxTitle: {
    margin: '0 0 8px 0',
    fontWeight: 600,
    fontSize: 12,
    color: '#374151',
  },
  checklist: {
    listStyle: 'none',
    padding: 0,
    margin: 0,
    fontSize: 13,
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  webcamOuter: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    background: '#f9fafb',
    border: '1px solid #eef0f3',
    borderRadius: 12,
    padding: 16,
  },
  faceEnrolledBox: {
    width: '100%',
    padding: 14,
    background: '#16a34a',
    color: '#fff',
    borderRadius: 10,
    textAlign: 'center',
    fontWeight: 600,
    fontSize: 14,
  },
  showHideBtn: {
    position: 'absolute',
    right: 10,
    top: '50%',
    transform: 'translateY(-50%)',
    background: 'none',
    border: 'none',
    color: '#2563eb',
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 600,
    padding: 0,
  },
  registerButton: (ready) => ({
    width: '100%',
    padding: 13,
    background: ready ? '#2563eb' : '#d1d5db',
    color: '#fff',
    border: 'none',
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 600,
    cursor: ready ? 'pointer' : 'not-allowed',
    transition: 'background 0.15s ease',
  }),
  message: {
    marginTop: 16,
    textAlign: 'center',
    fontSize: 13,
    color: '#374151',
  },

  // ---- Claude-style plan cards ----
  plansGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: 16,
  },
  planCard: (selected, highlighted) => ({
    position: 'relative',
    border: selected ? '2px solid #d97757' : highlighted ? '1px solid #e8ddd3' : '1px solid #e5e7eb',
    background: selected ? '#fdf6f2' : highlighted ? '#fdfaf7' : '#ffffff',
    borderRadius: 16,
    padding: '22px 18px 20px',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
    boxShadow: selected ? '0 0 0 3px rgba(217,119,87,0.12)' : 'none',
  }),
  planBadge: {
    position: 'absolute',
    top: -11,
    left: 18,
    background: '#d97757',
    color: '#fff',
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 0.4,
    textTransform: 'uppercase',
    padding: '4px 10px',
    borderRadius: 999,
  },
  planTagline: {
    fontSize: 11,
    fontWeight: 600,
    color: '#9a6b52',
    textTransform: 'uppercase',
    letterSpacing: 0.4,
    marginBottom: 6,
  },
  planName: {
    fontWeight: 700,
    fontSize: 17,
    color: '#111827',
    marginBottom: 6,
  },
  planPriceRow: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 4,
    marginBottom: 16,
  },
  planPrice: {
    fontSize: 26,
    fontWeight: 700,
    color: '#111827',
  },
  planPriceUnit: {
    fontSize: 12,
    color: '#6b7280',
  },
  planDivider: {
    borderTop: '1px solid #eee5dc',
    marginBottom: 14,
  },
  planFeatureList: {
    listStyle: 'none',
    padding: 0,
    margin: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    flexGrow: 1,
  },
  planFeatureItem: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 8,
    fontSize: 12.5,
    color: '#374151',
    lineHeight: 1.4,
  },
  planFeatureCheck: {
    color: '#d97757',
    fontWeight: 700,
    flexShrink: 0,
  },
  planSelectBtn: (selected) => ({
    marginTop: 18,
    width: '100%',
    padding: '9px 0',
    borderRadius: 8,
    border: selected ? 'none' : '1px solid #d1d5db',
    background: selected ? '#d97757' : '#fff',
    color: selected ? '#fff' : '#374151',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
  }),
};

export default function RegisterPage() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [faceCaptured, setFaceCaptured] = useState(false);
  const [faceEmbedding, setFaceEmbedding] = useState(null);
  const [environmentCheck, setEnvironmentCheck] = useState(null);
  const [selectedPlanId, setSelectedPlanId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [registrationComplete, setRegistrationComplete] = useState(false);
  const registerButtonRef = useRef(null);

  const isEmailValid = EMAIL_REGEX.test(email.trim());

  const checks = {
    length: password.length >= 8,
    hasLetter: /[a-zA-Z]/.test(password),
    hasNumber: /\d/.test(password),
    hasSpecialChar: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>/?]/.test(password),
  };
  const isPasswordValid = Object.values(checks).every(Boolean);
  const passwordsMatch = confirmPassword.length > 0 && password === confirmPassword;

  const handleFaceEnrolled = (embedding, envStatus) => {
    setFaceEmbedding(embedding);
    setEnvironmentCheck(envStatus);
    setFaceCaptured(true);
  };

  const isFormReady =
    username.trim().length >= 3 &&
    isEmailValid &&
    isPasswordValid &&
    passwordsMatch &&
    faceCaptured &&
    selectedPlanId;

  useEffect(() => {
    if (isFormReady && registerButtonRef.current) {
      registerButtonRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
      registerButtonRef.current.focus({ preventScroll: true });
    }
  }, [isFormReady]);

  const handleRegisterClick = async () => {
    if (registrationComplete || loading) return;

    if (!isFormReady) {
      setMessage('Please complete all fields, enroll your face, and select a plan.');
      return;
    }

    setLoading(true);
    setMessage('');

    try {
      const { data } = await axios.post('http://localhost:5000/api/payment/create-order', {
        planId: selectedPlanId,
      });

      const options = {
        key: data.key,
        amount: data.amount,
        currency: data.currency,
        order_id: data.orderId,
        name: 'Face Auth System',
        description: `${data.planName} Registration`,
        handler: async function (response) {
          const verifyRes = await axios.post('http://localhost:5000/api/payment/verify', {
            razorpay_order_id: response.razorpay_order_id,
            razorpay_payment_id: response.razorpay_payment_id,
            razorpay_signature: response.razorpay_signature,
          });

          try {
            const registerRes = await axios.post('http://localhost:5000/api/auth/register', {
              username,
              email,
              password,
              faceEmbedding,
              environmentCheck,
              paymentOrderId: verifyRes.data.orderId,
            });
            setMessage(registerRes.data.message);
            setRegistrationComplete(true);
          } catch (err) {
            setMessage(err.response?.data?.message || 'Registration failed after payment');
          } finally {
            setLoading(false);
          }
        },
        modal: {
          ondismiss: function () {
            setLoading(false);
            setMessage('Payment cancelled.');
          },
        },
        theme: { color: '#d97757' },
      };

      const rzp = new window.Razorpay(options);
      rzp.open();
    } catch (err) {
      console.error(err);
      setMessage('Could not start payment');
      setLoading(false);
    }
  };

  // ---- Success screen ----
  if (registrationComplete) {
    return (
      <div style={styles.page}>
        <div style={{ ...styles.card, textAlign: 'center' }}>
          <div style={{ fontSize: 48, marginBottom: 10 }}>✅</div>
          <h2 style={styles.heading}>Registration Successful</h2>
          <p style={{ color: '#6b7280', fontSize: 14, marginTop: 8 }}>
            Your account for <strong>{username}</strong> has been created and your payment was confirmed.
          </p>
          <button
            onClick={() => (window.location.href = '/login')}
            style={{ ...styles.registerButton(true), marginTop: 24, background: '#d97757' }}
          >
            Go to Login
          </button>
        </div>
      </div>
    );
  }

  // ---- Registration form ----
  return (
    <div style={styles.page}>
      <div style={styles.wideCard}>
        <h2 style={styles.heading}>Create Your Account</h2>
        <p style={styles.subheading}>Fill in your details to get started</p>

        {/* Username */}
        <div style={styles.section}>
          <label style={styles.label}>Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter username..."
            style={styles.input}
          />
        </div>

        {/* Email */}
        <div style={styles.section}>
          <label style={styles.label}>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter email..."
            style={{
              ...styles.input,
              ...(email.length > 0 && !isEmailValid ? styles.inputError : {}),
            }}
          />
          {email.length > 0 && !isEmailValid && (
            <p style={styles.errorText}>Enter a valid email address (e.g. name@example.com)</p>
          )}
        </div>

        {/* Password */}
        <div style={styles.section}>
          <label style={styles.label}>Password</label>
          <div style={{ position: 'relative' }}>
            <input
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password..."
              style={{ ...styles.input, paddingRight: 60 }}
            />
            <button
              type="button"
              onClick={() => setShowPassword((prev) => !prev)}
              style={styles.showHideBtn}
            >
              {showPassword ? 'Hide' : 'Show'}
            </button>
          </div>
          <div style={styles.passwordBox}>
            <p style={styles.passwordBoxTitle}>Password Requirements</p>
            <ul style={styles.checklist}>
              <li style={{ color: checks.length ? '#16a34a' : '#ef4444' }}>
                {checks.length ? '✓' : '✗'} At least 8 characters
              </li>
              <li style={{ color: checks.hasLetter ? '#16a34a' : '#ef4444' }}>
                {checks.hasLetter ? '✓' : '✗'} At least one letter
              </li>
              <li style={{ color: checks.hasNumber ? '#16a34a' : '#ef4444' }}>
                {checks.hasNumber ? '✓' : '✗'} At least one number
              </li>
              <li style={{ color: checks.hasSpecialChar ? '#16a34a' : '#ef4444' }}>
                {checks.hasSpecialChar ? '✓' : '✗'} At least one special character
              </li>
            </ul>
          </div>
        </div>

        {/* Confirm Password */}
        <div style={styles.section}>
          <label style={styles.label}>Confirm Password</label>
          <input
            type={showPassword ? 'text' : 'password'}
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Re-enter password..."
            style={{
              ...styles.input,
              ...(confirmPassword.length > 0 && !passwordsMatch ? styles.inputError : {}),
            }}
          />
          {confirmPassword.length > 0 && !passwordsMatch && (
            <p style={styles.errorText}>Passwords do not match</p>
          )}
          {confirmPassword.length > 0 && passwordsMatch && (
            <p style={styles.successText}>✓ Passwords match</p>
          )}
        </div>

        {/* Biometric enrollment */}
        <div style={styles.section}>
          <label style={styles.label}>Biometric Enrollment</label>
          {!faceCaptured ? (
            <div style={styles.webcamOuter}>
              <WebcamCapture onEnrolled={handleFaceEnrolled} />
            </div>
          ) : (
            <div style={styles.faceEnrolledBox}>✓ Face Enrolled</div>
          )}
        </div>

        {/* Plan selection — Claude-style subscription cards */}
        <div style={styles.section}>
          <label style={styles.label}>Choose a Plan</label>
          <div style={styles.plansGrid}>
            {PLANS.map((plan) => {
              const selected = selectedPlanId === plan.id;
              return (
                <div
                  key={plan.id}
                  onClick={() => setSelectedPlanId(plan.id)}
                  style={styles.planCard(selected, plan.highlighted)}
                >
                  {plan.highlighted && <div style={styles.planBadge}>Most Popular</div>}
                  <div style={styles.planTagline}>{plan.tagline}</div>
                  <div style={styles.planName}>{plan.name}</div>
                  <div style={styles.planPriceRow}>
                    <span style={styles.planPrice}>₹{plan.amount}</span>
                    <span style={styles.planPriceUnit}>/ one-time</span>
                  </div>
                  <div style={styles.planDivider} />
                  <ul style={styles.planFeatureList}>
                    {plan.features.map((feature, i) => (
                      <li key={i} style={styles.planFeatureItem}>
                        <span style={styles.planFeatureCheck}>✓</span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                  <button type="button" style={styles.planSelectBtn(selected)}>
                    {selected ? 'Selected' : 'Choose plan'}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Register button */}
        <button
          ref={registerButtonRef}
          onClick={handleRegisterClick}
          disabled={loading}
          style={styles.registerButton(isFormReady)}
        >
          {loading ? 'Processing...' : 'Register'}
        </button>

        {message && <p style={styles.message}>{message}</p>}
      </div>
    </div>
  );
}
