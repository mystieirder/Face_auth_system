import React, { useState, useEffect, useRef, useCallback } from 'react';

const SESSION_KEY_PREFIX = 'profileForm:';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
const GESTURE_WS_URL = process.env.REACT_APP_GESTURE_WS_URL || 'ws://localhost:8765';

const DEFAULT_FORM = {
  fullName: '',
  contactPhone: '',
  emailAddress: '',
  dateOfBirth: '',
  physicalAddress: '',
  accessibilityMode: 'default',
};

const styles = {
  page: { minHeight: '100vh', background: '#f3f4f6', padding: '40px 16px', fontFamily: "'Segoe UI', Roboto, sans-serif" },
  card: { maxWidth: 560, margin: '0 auto', background: '#fff', borderRadius: 16, boxShadow: '0 4px 20px rgba(0,0,0,0.08)', padding: '32px' },
  headerBar: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
  heading: { fontSize: 22, fontWeight: 700, color: '#111827', margin: 0 },
  logoutBtn: { background: 'transparent', color: '#ef4444', border: '1px solid #ef4444', padding: '6px 14px', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13 },
  captureSection: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, marginBottom: 24, padding: 16, background: '#f9fafb', borderRadius: 10 },
  viewportWrap: { position: 'relative', width: 320, height: 240, background: '#000', borderRadius: 8, overflow: 'hidden' },
  mediaFill: { position: 'absolute', top: 0, left: 0, width: 320, height: 240, objectFit: 'cover' },
  captureControls: { display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' },
  btn: { border: 'none', padding: '9px 16px', borderRadius: 6, fontWeight: 600, cursor: 'pointer', fontSize: 13 },
  btnPrimary: { background: '#2563eb', color: '#fff' },
  btnSecondary: { background: 'transparent', color: '#2563eb', border: '1px solid #2563eb' },
  btnSuccess: { background: '#16a34a', color: '#fff' },
  btnDisabled: { opacity: 0.45, cursor: 'not-allowed' },
  captureStatus: { fontSize: 13, color: '#6b7280', minHeight: '1.2em', textAlign: 'center' },
  speechControls: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, padding: 12, background: '#f9fafb', borderRadius: 8 },
  micBtn: { background: 'none', border: '2px solid #d1d5db', borderRadius: '50%', width: 44, height: 44, cursor: 'pointer', fontSize: 18 },
  micBtnListening: { background: '#ef4444', borderColor: '#ef4444', color: '#fff' },
  tooltip: { fontStyle: 'italic', fontSize: 13, opacity: 0.8, flexGrow: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  modesRow: { display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' },
  modeBox: { flex: '1 1 220px', background: '#f9fafb', border: '1px solid #eef0f3', borderRadius: 10, padding: 14, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, minWidth: 200 },
  modeBoxHeader: { width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13, fontWeight: 700, color: '#374151' },
  modeStatusDot: { width: 9, height: 9, borderRadius: '50%', flexShrink: 0 },
  modeStatusText: { fontSize: 12, color: '#6b7280', textAlign: 'center', minHeight: '1.5em', margin: 0 },
  gestureFeed: { width: '100%', maxWidth: 220, aspectRatio: '4 / 3', objectFit: 'cover', borderRadius: 8, background: '#000' },
  gestureFeedPlaceholder: { width: '100%', maxWidth: 220, aspectRatio: '4 / 3', borderRadius: 8, border: '1px dashed #d1d5db', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', fontSize: 11, color: '#9ca3af', padding: 10, boxSizing: 'border-box' },
  section: { marginBottom: 16 },
  label: { display: 'block', marginBottom: 6, fontSize: 13, fontWeight: 600, color: '#374151' },
  input: { width: '100%', padding: '11px 12px', borderRadius: 8, border: '1px solid #d1d5db', boxSizing: 'border-box', fontSize: 14 },
  saveButton: { width: '100%', padding: 13, background: '#2563eb', color: '#fff', border: 'none', borderRadius: 8, fontSize: 15, fontWeight: 600, cursor: 'pointer', marginTop: 8 },
  saveIndicator: { fontSize: 12, color: '#059669', textAlign: 'center', marginTop: 10 },
  commandGuide: { marginTop: 20, fontSize: 12, borderTop: '1px solid #e5e7eb', paddingTop: 14, color: '#6b7280' },
};

const FIELD_ID_MAP = {
  fullName: 'profile-fullName',
  contactPhone: 'profile-contactPhone',
  emailAddress: 'profile-emailAddress',
  dateOfBirth: 'profile-dateOfBirth',
  physicalAddress: 'profile-physicalAddress',
  accessibilityMode: 'profile-accessibilityMode',
};

export default function ProfilePage({ user, onLogout }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [saveMsg, setSaveMsg] = useState('');
  const [loading, setLoading] = useState(true);

  // ---------------- FR-3: load existing profile, restore session draft ----------------
  useEffect(() => {
    async function loadProfile() {
      const token = localStorage.getItem('authToken');
      try {
        const res = await fetch(`${API_URL}/profile`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        if (data.success && data.profile) {
          setForm({
            fullName: data.profile.FullName || '',
            contactPhone: data.profile.ContactPhone || '',
            emailAddress: data.profile.EmailAddress || user?.email || '',
            dateOfBirth: data.profile.DateOfBirth ? data.profile.DateOfBirth.split('T')[0] : '',
            physicalAddress: data.profile.PhysicalAddress || '',
            accessibilityMode: data.profile.AccessibilityMode || 'default',
          });
        } else {
          restoreSessionDraft();
        }
      } catch (err) {
        restoreSessionDraft();
      } finally {
        setLoading(false);
      }
    }

    function restoreSessionDraft() {
      const restored = { ...DEFAULT_FORM, emailAddress: user?.email || '' };
      Object.keys(DEFAULT_FORM).forEach((key) => {
        const saved = sessionStorage.getItem(SESSION_KEY_PREFIX + key);
        if (saved !== null) restored[key] = saved;
      });
      setForm(restored);
    }

    loadProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // FR-3.3: auto-save each field change to sessionStorage
  const handleChange = (field) => (e) => {
    const value = e.target.value;
    setForm((f) => ({ ...f, [field]: value }));
    sessionStorage.setItem(SESSION_KEY_PREFIX + field, value);
    setSaveMsg('Saved ' + new Date().toLocaleTimeString());
  };

  const handleSubmit = useCallback(async (e) => {
    e && e.preventDefault();
    const token = localStorage.getItem('authToken');
    try {
      const res = await fetch(`${API_URL}/profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (data.success) {
        setSaveMsg('Profile submitted successfully!');
      } else {
        setSaveMsg('Submission failed: ' + (data.message || 'unknown error'));
      }
    } catch (err) {
      setSaveMsg('Error: ' + err.message);
    }
  }, [form]);

  // =====================================================================
  // FR-5: Webcam Profile Picture Capture
  // =====================================================================
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [captureStage, setCaptureStage] = useState('live'); // live | captured | confirmed
  const [captureStatus, setCaptureStatus] = useState('Align your head and shoulders inside the guide.');
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    async function startWebcam() {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
        setStream(s);
        if (videoRef.current) videoRef.current.srcObject = s;
        drawFramingGuide();
      } catch (err) {
        setCaptureStatus('Camera unavailable: ' + err.message);
      }
    }
    startWebcam();
    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function drawFramingGuide() {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(160, 95, 55, 75, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(160, 210, 100, 45, 0, Math.PI, Math.PI * 2);
    ctx.stroke();
  }

  useEffect(() => {
    if (captureStage === 'live') drawFramingGuide();
  }, [captureStage, stream]);

  const handleSnap = useCallback(() => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    setCaptureStage('captured');
    setCaptureStatus('Photo captured. Retake or Crop & Confirm.');
  }, []);

  const handleRetake = useCallback(() => {
    setCaptureStage('live');
    setPreviewUrl(null);
    setCaptureStatus('Align your head and shoulders inside the guide.');
  }, []);

  function compressUnder2MB(sourceCanvas) {
    const MAX_BYTES = 2 * 1024 * 1024;
    const mimeCandidates = ['image/webp', 'image/jpeg'];
    return new Promise((resolve) => {
      (function tryMime(mimeIndex) {
        if (mimeIndex >= mimeCandidates.length) return resolve(null);
        const mime = mimeCandidates[mimeIndex];
        (function tryQuality(quality) {
          sourceCanvas.toBlob(
            (blob) => {
              if (!blob) return tryMime(mimeIndex + 1);
              if (blob.size <= MAX_BYTES) return resolve(blob);
              if (quality <= 0.2) return tryMime(mimeIndex + 1);
              tryQuality(quality - 0.1);
            },
            mime,
            quality
          );
        })(0.9);
      })(0);
    });
  }

  const handleConfirm = useCallback(async () => {
    setCaptureStatus('Processing photo...');
    const canvas = captureCanvasRef.current;
    const cropSize = Math.min(canvas.width, canvas.height);
    const sx = (canvas.width - cropSize) / 2;
    const sy = (canvas.height - cropSize) / 2;

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropSize;
    cropCanvas.height = cropSize;
    cropCanvas.getContext('2d').drawImage(canvas, sx, sy, cropSize, cropSize, 0, 0, cropSize, cropSize);

    const blob = await compressUnder2MB(cropCanvas);
    if (!blob) {
      setCaptureStatus('Could not compress photo under 2MB. Try better lighting / retake.');
      return;
    }

    setPreviewUrl(URL.createObjectURL(blob));
    setCaptureStage('confirmed');

    const token = localStorage.getItem('authToken');
    const formData = new FormData();
    formData.append('photo', blob, 'profile.webp');
    setCaptureStatus('Uploading photo...');
    try {
      const res = await fetch(`${API_URL}/profile/photo`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setCaptureStatus(`Photo saved (${(blob.size / 1024).toFixed(1)} KB).`);
      } else {
        setCaptureStatus('Upload failed: ' + (data.message || 'unknown error'));
      }
    } catch (err) {
      setCaptureStatus('Upload error: ' + err.message);
    }
  }, []);

  // =====================================================================
  // FR-4: Voice-Guided Data Entry
  // =====================================================================
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('Click mic or speak to start…');
  const recognitionRef = useRef(null);
  const formRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setTranscript('Speech API not supported in this browser.');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setListening(true);
      setTranscript('Listening…');
    };
    recognition.onend = () => {
      setListening(false);
      setTranscript('Microphone paused. Click to listen.');
    };
    recognition.onresult = (event) => {
      const r = event.results[event.resultIndex];
      const text = r[0].transcript.trim();
      setTranscript(`"${text}"`);
      if (r.isFinal) processVoiceCommand(text.toLowerCase());
    };

    recognitionRef.current = recognition;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleMic = () => {
    if (!recognitionRef.current) return;
    listening ? recognitionRef.current.stop() : recognitionRef.current.start();
  };

  function isInputField(el) {
    return el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName);
  }

  function triggerChange(el, value) {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function focusNextField(current) {
    const els = Array.from(formRef.current.querySelectorAll('input, select, textarea'));
    const i = els.indexOf(current);
    els[i > -1 && i < els.length - 1 ? i + 1 : 0].focus();
  }

  function focusPrevField(current) {
    const els = Array.from(formRef.current.querySelectorAll('input, select, textarea'));
    const i = els.indexOf(current);
    els[i > 0 ? i - 1 : els.length - 1].focus();
  }

  function processVoiceCommand(command) {
    const active = document.activeElement;

    if (command === 'logout' || command === 'log out') return onLogout && onLogout();
    if (command === 'submit profile') return handleSubmit();
    if (command === 'take photo' || command === 'snap photo') return handleSnap();
    if (command === 'retake photo') return handleRetake();
    if (command === 'confirm photo' || command === 'crop and confirm') return handleConfirm();

    if (command === 'clear field') {
      if (isInputField(active)) triggerChange(active, '');
      return;
    }
    if (command === 'delete last word') {
      if (isInputField(active) && active.value) {
        const words = active.value.trim().split(' ');
        words.pop();
        triggerChange(active, words.join(' '));
      }
      return;
    }
    if (command === 'next field') return focusNextField(active);
    if (command === 'previous field' || command === 'prev field') return focusPrevField(active);

    const fieldMap = {
      'select full name': FIELD_ID_MAP.fullName,
      'select contact phone': FIELD_ID_MAP.contactPhone,
      'select phone': FIELD_ID_MAP.contactPhone,
      'select email address': FIELD_ID_MAP.emailAddress,
      'select email': FIELD_ID_MAP.emailAddress,
      'select date of birth': FIELD_ID_MAP.dateOfBirth,
      'select physical address': FIELD_ID_MAP.physicalAddress,
      'select address': FIELD_ID_MAP.physicalAddress,
      'select accessibility mode': FIELD_ID_MAP.accessibilityMode,
    };
    if (fieldMap[command]) {
      document.getElementById(fieldMap[command]).focus();
      return;
    }

    if (isInputField(active)) {
      if (active.tagName === 'SELECT') {
        Array.from(active.options).forEach((opt) => {
          if (opt.text.toLowerCase().includes(command)) triggerChange(active, opt.value);
        });
      } else {
        triggerChange(active, (active.value ? active.value + ' ' : '') + command);
      }
    }
  }

  // =====================================================================
  // Hand Gesture control — connects to the local web_bridge.py WebSocket
  // server on page load, shows the live annotated camera feed, and
  // executes the gesture command directly the moment it arrives.
  // =====================================================================
  const [gestureStatus, setGestureStatus] = useState('connecting'); // connecting | connected | disconnected
  const [gestureFrame, setGestureFrame] = useState(null);
  const [gestureLastAction, setGestureLastAction] = useState('');
  const gestureWsRef = useRef(null);
  const gestureRetryRef = useRef(null);
  const executeGestureCommandRef = useRef(() => {});

  // Always keep this pointed at the latest closures (form state, capture
  // stage, etc.) so the long-lived WebSocket connection never acts on
  // stale data, even though it's only opened once.
  useEffect(() => {
    executeGestureCommandRef.current = function executeGestureCommand(command) {
      const active = document.activeElement;
      switch (command) {
        case 'next_field':
          focusNextField(active);
          break;
        case 'prev_field':
          focusPrevField(active);
          break;
        case 'select': {
          if (active && active.tagName === 'SELECT') {
            active.selectedIndex = (active.selectedIndex + 1) % active.options.length;
            active.dispatchEvent(new Event('change', { bubbles: true }));
          } else if (active && (active.tagName === 'BUTTON' || active.type === 'submit')) {
            active.click();
          }
          break;
        }
        case 'confirm': {
          if (captureStage === 'captured') {
            handleConfirm();
          } else {
            const els = Array.from(formRef.current.querySelectorAll('input, select, textarea'));
            const isLastField = els.length > 0 && els.indexOf(active) === els.length - 1;
            if (isLastField) handleSubmit();
            else focusNextField(active);
          }
          break;
        }
        default:
          break;
      }
    };
  });

  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;
      setGestureStatus((prev) => (prev === 'connected' ? prev : 'connecting'));
      const ws = new WebSocket(GESTURE_WS_URL);
      gestureWsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setGestureStatus('connected');
      };
      ws.onmessage = (event) => {
        if (cancelled) return;
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch (err) {
          return;
        }
        if (msg.type === 'frame' && msg.data) {
          setGestureFrame(`data:image/jpeg;base64,${msg.data}`);
        } else if (msg.type === 'action' && msg.command) {
          setGestureLastAction(msg.command);
          executeGestureCommandRef.current(msg.command);
        }
      };
      ws.onclose = () => {
        if (cancelled) return;
        setGestureStatus('disconnected');
        gestureRetryRef.current = setTimeout(connect, 3000);
      };
      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (gestureRetryRef.current) clearTimeout(gestureRetryRef.current);
      if (gestureWsRef.current) gestureWsRef.current.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) {
    return (
      <div style={styles.page}>
        <div style={styles.card}>Loading profile…</div>
      </div>
    );
  }

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.headerBar}>
          <h2 style={styles.heading}>Profile Information</h2>
          <button type="button" style={styles.logoutBtn} onClick={onLogout}>Logout</button>
        </div>

        {/* FR-5: Webcam Profile Picture Capture */}
        <div style={styles.captureSection}>
          <div style={styles.viewportWrap}>
            {captureStage !== 'confirmed' ? (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  width="320"
                  height="240"
                  style={{ ...styles.mediaFill, display: captureStage === 'live' ? 'block' : 'none' }}
                />
                <canvas
                  ref={overlayRef}
                  width="320"
                  height="240"
                  style={{ ...styles.mediaFill, pointerEvents: 'none', display: captureStage === 'live' ? 'block' : 'none' }}
                />
                <canvas
                  ref={captureCanvasRef}
                  width="320"
                  height="240"
                  style={{ ...styles.mediaFill, display: captureStage === 'captured' ? 'block' : 'none' }}
                />
              </>
            ) : (
              <img src={previewUrl} alt="Profile preview" style={styles.mediaFill} />
            )}
          </div>
          <div style={styles.captureControls}>
            <button
              type="button"
              style={{ ...styles.btn, ...styles.btnPrimary, ...(captureStage !== 'live' ? styles.btnDisabled : {}) }}
              onClick={handleSnap}
              disabled={captureStage !== 'live'}
            >
              📸 Snap Photo
            </button>
            <button
              type="button"
              style={{ ...styles.btn, ...styles.btnSecondary, ...(captureStage !== 'captured' ? styles.btnDisabled : {}) }}
              onClick={handleRetake}
              disabled={captureStage !== 'captured'}
            >
              🔄 Retake
            </button>
            <button
              type="button"
              style={{ ...styles.btn, ...styles.btnSuccess, ...(captureStage !== 'captured' ? styles.btnDisabled : {}) }}
              onClick={handleConfirm}
              disabled={captureStage !== 'captured'}
            >
              ✂️ Crop & Confirm
            </button>
          </div>
          <p style={styles.captureStatus}>{captureStatus}</p>
        </div>

        {/* FR-4: Voice-Guided Data Entry + Hand Gesture control — side by side */}
        <div style={styles.modesRow}>
          <div style={styles.modeBox}>
            <div style={styles.modeBoxHeader}>
              <span>🎙️ Speech-to-Text</span>
              <span
                style={{
                  ...styles.modeStatusDot,
                  background: listening ? '#16a34a' : '#9ca3af',
                }}
              />
            </div>
            <button
              type="button"
              style={{ ...styles.micBtn, ...(listening ? styles.micBtnListening : {}) }}
              onClick={toggleMic}
              title="Toggle Voice Recognition"
            >
              🎙️
            </button>
            <p style={styles.modeStatusText}>{transcript}</p>
          </div>

          <div style={styles.modeBox}>
            <div style={styles.modeBoxHeader}>
              <span>✋ Hand Gesture</span>
              <span
                style={{
                  ...styles.modeStatusDot,
                  background:
                    gestureStatus === 'connected'
                      ? '#16a34a'
                      : gestureStatus === 'connecting'
                      ? '#f59e0b'
                      : '#ef4444',
                }}
              />
            </div>
            {gestureFrame ? (
              <img src={gestureFrame} alt="Hand gesture camera feed" style={styles.gestureFeed} />
            ) : (
              <div style={styles.gestureFeedPlaceholder}>
                {gestureStatus === 'connecting'
                  ? 'Connecting to gesture control…'
                  : 'Run python web_bridge.py to start hand gesture control'}
              </div>
            )}
            <p style={styles.modeStatusText}>
              {gestureStatus === 'connected'
                ? gestureLastAction
                  ? `Last command: ${gestureLastAction.replace('_', ' ')}`
                  : 'Connected — show a gesture'
                : gestureStatus === 'connecting'
                ? 'Connecting…'
                : 'Not connected — retrying…'}
            </p>
          </div>
        </div>

        {/* FR-3: Profile form */}
        <form ref={formRef} onSubmit={handleSubmit}>
          <div style={styles.section}>
            <label style={styles.label} htmlFor={FIELD_ID_MAP.fullName}>Full Legal Name</label>
            <input id={FIELD_ID_MAP.fullName} type="text" value={form.fullName} onChange={handleChange('fullName')} style={styles.input} />
          </div>
          <div style={styles.section}>
            <label style={styles.label} htmlFor={FIELD_ID_MAP.contactPhone}>Contact Phone</label>
            <input id={FIELD_ID_MAP.contactPhone} type="tel" value={form.contactPhone} onChange={handleChange('contactPhone')} style={styles.input} />
          </div>
          <div style={styles.section}>
            <label style={styles.label} htmlFor={FIELD_ID_MAP.emailAddress}>Email Address</label>
            <input id={FIELD_ID_MAP.emailAddress} type="email" value={form.emailAddress} onChange={handleChange('emailAddress')} style={styles.input} />
          </div>
          <div style={styles.section}>
            <label style={styles.label} htmlFor={FIELD_ID_MAP.dateOfBirth}>Date of Birth</label>
            <input id={FIELD_ID_MAP.dateOfBirth} type="date" value={form.dateOfBirth} onChange={handleChange('dateOfBirth')} style={styles.input} />
          </div>
          <div style={styles.section}>
            <label style={styles.label} htmlFor={FIELD_ID_MAP.physicalAddress}>Physical Address</label>
            <input id={FIELD_ID_MAP.physicalAddress} type="text" value={form.physicalAddress} onChange={handleChange('physicalAddress')} style={styles.input} />
          </div>
          <div style={styles.section}>
            <label style={styles.label} htmlFor={FIELD_ID_MAP.accessibilityMode}>Accessibility Mode Preferences</label>
            <select id={FIELD_ID_MAP.accessibilityMode} value={form.accessibilityMode} onChange={handleChange('accessibilityMode')} style={styles.input}>
              <option value="default">Default</option>
              <option value="high-contrast">High Contrast</option>
              <option value="large-text">Large Text</option>
            </select>
          </div>

          <button type="submit" style={styles.saveButton}>Submit Profile</button>
          {saveMsg && <p style={styles.saveIndicator}>{saveMsg}</p>}
        </form>

        <div style={styles.commandGuide}>
          <strong>Voice Commands:</strong>
          <p>"Select [Field Name]" • "Clear Field" • "Delete Last Word"</p>
          <p>"Next Field" • "Previous Field" • "Submit Profile" • "Logout"</p>
          <p>"Take Photo" • "Retake Photo" • "Confirm Photo"</p>
          <strong style={{ display: 'block', marginTop: 10 }}>Hand Gestures:</strong>
          <p>👍 Thumbs Up = Confirm &nbsp;•&nbsp; ✋ Open Palm = Next Field</p>
          <p>👋 Side Sweep = Previous Field &nbsp;•&nbsp; 🤏 Pinch = Select</p>
        </div>
      </div>
    </div>
  );
}