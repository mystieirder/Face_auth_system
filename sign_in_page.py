"""
Inclusive Multimodal User Onboarding & Profile Management System
------------------------------------------------------------------
Implements SRS-2026-MM-01 sections:
  3.3  Profile Information Management (FR-3.1 - FR-3.3)
  3.4  Voice-Guided Data Entry / Speech-to-Text        (FR-4.1 - FR-4.4)
  3.5  Webcam Profile Picture Capture                  (FR-5.1 - FR-5.4)

NOTE: This is a standalone Flask reference prototype. It is NOT wired to
the real app's SQL Server backend or face-api.js login flow — those are
already implemented and working in the actual Node/Express + React app.
This file intentionally uses an in-memory store so it runs on its own
without requiring the real DB connection or face-recognition service.

The webcam capture is implemented as an EMBEDDED browser viewport
(getUserMedia + <canvas>) directly on the Profile page, per FR-5.1,
rather than a separate desktop process — this matches the SRS's
"Web / Progressive Web App" target platform.
"""

import json
import os
import re
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import face_recognition
except ImportError:  # pragma: no cover - optional dependency
    face_recognition = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover - optional dependency
    Fernet = None

from flask import (
    Flask, render_template_string, request, redirect, url_for,
    jsonify, session, send_from_directory
)
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret-change-me")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "profile_photos"
UPLOAD_FOLDER.mkdir(exist_ok=True)
DATA_FILE = BASE_DIR / "users.json"
BIOMETRIC_DIR = BASE_DIR / "biometric_module"
ENROLLED_FACE_PATH = BIOMETRIC_DIR / "enrolled_face.enc"
FACE_KEY_PATH = BIOMETRIC_DIR / "secret.key"

MAX_PHOTO_BYTES = 2 * 1024 * 1024  # FR-5.4: 2MB cap, enforced server-side too
FACE_MATCH_THRESHOLD = 0.6

PROFILE_FIELDS = [
    "full_name", "contact_phone", "email_address", "dob",
    "physical_address", "accessibility_mode",
]
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_POLICY = "Use at least 8 characters, including an uppercase letter and a special character."


def load_users():
    if DATA_FILE.exists():
        try:
            with DATA_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_users(users):
    with DATA_FILE.open("w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2)


USERS = load_users()


def clean_username(username):
    return re.sub(r"[^A-Za-z0-9_-]", "", username or "").strip()


def validate_password(password):
    if len(password or "") < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain an uppercase letter."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must contain a special character."
    return True, ""


def password_matches(stored_password, password):
    if not stored_password:
        return False
    try:
        return check_password_hash(stored_password, password)
    except ValueError:
        return stored_password == password


def find_user_by_identifier(identifier):
    identifier = (identifier or "").strip().lower()
    for username, user_data in USERS.items():
        if username.lower() == identifier:
            return username
        if (user_data.get("email") or "").lower() == identifier:
            return username
    return None


def ensure_face_key():
    if FACE_KEY_PATH.exists():
        return FACE_KEY_PATH.read_bytes()
    key = Fernet.generate_key() if Fernet is not None else b""
    if not key:
        raise RuntimeError("Face login dependencies are not available")
    FACE_KEY_PATH.write_bytes(key)
    return key


def save_enrolled_face_embedding(embedding):
    if Fernet is None:
        raise RuntimeError("Face login dependencies are not available")
    key = ensure_face_key()
    fernet = Fernet(key)
    encrypted = fernet.encrypt(json.dumps(embedding.tolist()).encode("utf-8"))
    ENROLLED_FACE_PATH.write_bytes(encrypted)
    return encrypted


def load_enrolled_face_embedding():
    if cv2 is None or face_recognition is None or np is None or Fernet is None:
        raise RuntimeError("Face login dependencies are not available")
    if not ENROLLED_FACE_PATH.exists():
        raise FileNotFoundError("No enrolled face found")
    if not FACE_KEY_PATH.exists():
        raise FileNotFoundError("Face encryption key not found")

    fernet = Fernet(ensure_face_key())
    decrypted = fernet.decrypt(ENROLLED_FACE_PATH.read_bytes()).decode("utf-8")
    return np.array(json.loads(decrypted), dtype=np.float32)


def decode_photo_frame(photo_file):
    if photo_file is None:
        raise ValueError("No face photo was provided.")

    photo_file.seek(0)
    image_bytes = photo_file.read()
    if not image_bytes:
        raise ValueError("No face photo was provided.")

    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The uploaded image could not be read.")
    return frame


def extract_face_embedding(photo_file):
    if cv2 is None or face_recognition is None or np is None:
        raise RuntimeError("Face login dependencies are not available")

    frame = decode_photo_frame(photo_file)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if len(face_locations) == 0:
        raise ValueError("No face detected. Please position yourself in front of the camera.")
    if len(face_locations) > 1:
        raise ValueError("Multiple faces detected. Please ensure only one face is visible.")

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        raise ValueError("The face could not be encoded.")
    return face_encodings[0]


def detect_blink(frame):
    if cv2 is None or face_recognition is None or np is None:
        return False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    landmarks = face_recognition.face_landmarks(rgb_frame)
    if not landmarks:
        return False

    left_eye = landmarks[0]["left_eye"]
    right_eye = landmarks[0]["right_eye"]

    def eye_aspect_ratio(eye_points):
        a = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        b = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        c = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        return (a + b) / (2.0 * c) if c else 0.0

    avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    return avg_ear < 0.25


def detect_blink_from_photo(photo_file):
    frame = decode_photo_frame(photo_file)
    return detect_blink(frame)


def verify_face_login(photo_file, username=None):
    if cv2 is None or face_recognition is None or np is None or Fernet is None:
        return False, "Face login dependencies are not available in this environment."

    if photo_file is None:
        return False, "No face photo was provided."

    try:
        live_embedding = extract_face_embedding(photo_file)
    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - runtime protection
        return False, f"Unable to process the uploaded image: {exc}"

    user_key = clean_username(username) if username else None
    try:
        if user_key:
            user_data = USERS.get(user_key, {})
            stored_vector = user_data.get("face_embedding")
            if stored_vector:
                enrolled_embedding = np.array(stored_vector, dtype=np.float32)
            else:
                enrolled_embedding = load_enrolled_face_embedding()
        else:
            enrolled_embedding = load_enrolled_face_embedding()
    except FileNotFoundError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - runtime protection
        return False, f"Unable to load enrolled face: {exc}"

    distance = float(np.linalg.norm(enrolled_embedding - live_embedding))
    if distance < FACE_MATCH_THRESHOLD:
        return True, "Face login successful."
    return False, f"Face does not match the enrolled profile (distance {distance:.3f})."


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sign In - Multimodal Profile System</title>
<style>
  :root { --primary:#16a34a; --bg:#f8fafc; --card:#fff; --text:#1e293b; --border:#cbd5e1; }
  body { font-family: system-ui, sans-serif; background:var(--bg); color:var(--text);
         display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }
  .card { background:var(--card); border:1px solid var(--border); padding:35px; border-radius:8px;
          width:100%; max-width:380px; box-shadow:0 4px 12px rgba(0,0,0,.08); }
  h2 { margin:0 0 8px; text-align:center; }
  p.sub { color:#15803d; font-size:.9rem; text-align:center; margin-bottom:24px; }
  .fg { margin-bottom:18px; }
  label { display:block; margin-bottom:6px; font-weight:600; font-size:.9rem; }
  input { width:100%; padding:10px 12px; border:1px solid var(--border); border-radius:4px;
          font-size:.95rem; box-sizing:border-box; }
  button { background:var(--primary); color:#fff; border:none; padding:12px; width:100%;
           border-radius:4px; font-size:1rem; font-weight:bold; cursor:pointer; }
  .err { color:#dc2626; font-size:.85rem; margin-bottom:12px; text-align:center; }
</style>
</head>
<body>
<div class="card">
  <h2>Welcome</h2>
  <p class="sub">Sign in or create an account to continue.</p>
  {% if error %}<p class="err">{{ error }}</p>{% endif %}
  <form method="POST" action="{{ url_for('login_page') }}">
    <div class="fg"><label for="username">Username</label>
      <input type="text" id="username" name="username" placeholder="Enter your username" required></div>
    <div class="fg">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" placeholder="Enter your password" required>
      <div style="margin-top:6px; display:flex; align-items:center; gap:6px;">
        <input type="checkbox" id="showPassword" style="width:auto; cursor:pointer;">
        <label for="showPassword" style="margin:0; font-weight:normal; font-size:.85rem; cursor:pointer;">Show password</label>
      </div>
    </div>
    <button type="submit">Sign In</button>
    <button type="button" onclick="window.location.href='{{ url_for('register_page') }}'"
            style="background:#64748b; margin-top:10px;">Register New Account</button>
    <div style="margin-top:16px; padding-top:12px; border-top:1px solid var(--border);">
      <button type="button" id="faceLoginBtn" style="background:#0f766e; margin-top:0;">Face Login</button>
      <div id="facePreviewWrap" style="display:none; margin-top:10px;">
        <video id="faceVideo" autoplay playsinline muted style="width:100%; border-radius:6px; background:#0f172a;"></video>
        <canvas id="faceCanvas" style="display:none;"></canvas>
      </div>
      <p id="faceStatus" style="margin:8px 0 0; font-size:.85rem; color:#64748b; text-align:center;">Use the camera to sign in with your face.</p>
    </div>
  </form>
</div>

<script>
  const passwordInput = document.getElementById('password');
  const showPasswordCheckbox = document.getElementById('showPassword');
  const faceLoginBtn = document.getElementById('faceLoginBtn');
  const facePreviewWrap = document.getElementById('facePreviewWrap');
  const faceVideo = document.getElementById('faceVideo');
  const faceCanvas = document.getElementById('faceCanvas');
  const faceStatus = document.getElementById('faceStatus');

  let faceStream = null;
  let faceLoginStarted = false;

  if (showPasswordCheckbox) {
    showPasswordCheckbox.addEventListener('change', () => {
      passwordInput.type = showPasswordCheckbox.checked ? 'text' : 'password';
    });
  }

  async function toggleFaceLogin() {
    if (!faceStream) {
      try {
        faceStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        faceVideo.srcObject = faceStream;
        facePreviewWrap.style.display = 'block';
        faceStatus.textContent = 'Camera ready. Click again to capture and sign in.';
        faceLoginBtn.textContent = 'Capture & Sign In';
        faceLoginStarted = true;
      } catch (err) {
        faceStatus.textContent = 'Camera access was denied or unavailable.';
      }
      return;
    }

    if (!faceLoginStarted) {
      faceStatus.textContent = 'Camera ready. Click again to capture and sign in.';
      return;
    }

    faceLoginBtn.disabled = true;
    faceStatus.textContent = 'Checking face…';

    faceCanvas.width = faceVideo.videoWidth || 320;
    faceCanvas.height = faceVideo.videoHeight || 240;
    const ctx = faceCanvas.getContext('2d');
    ctx.drawImage(faceVideo, 0, 0, faceCanvas.width, faceCanvas.height);

    try {
      const blob = await new Promise((resolve, reject) => {
        faceCanvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not capture image')), 'image/jpeg', 0.9);
      });

      const formData = new FormData();
      formData.append('photo', blob, 'face-login.jpg');
      const response = await fetch('/api/face-login', { method: 'POST', body: formData });
      const data = await response.json();
      if (data.status === 'success') {
        window.location.href = '/profile';
      } else {
        faceStatus.textContent = data.message || 'Face login failed.';
      }
    } catch (err) {
      faceStatus.textContent = 'Face login failed: ' + (err.message || err);
    } finally {
      faceLoginBtn.disabled = false;
    }
  }

  if (faceLoginBtn) {
    faceLoginBtn.addEventListener('click', toggleFaceLogin);
  }
</script>
</body>
</html>
"""

REGISTRATION_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Create Account - Multimodal Profile System</title>
<style>
  :root { --primary:#16a34a; --bg:#f8fafc; --card:#fff; --text:#1e293b; --border:#cbd5e1; --danger:#dc2626; --success:#15803d; }
  body { font-family: system-ui, sans-serif; background:var(--bg); color:var(--text);
         display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; padding:20px; }
  .card { background:var(--card); border:1px solid var(--border); padding:35px; border-radius:8px;
          width:100%; max-width:420px; box-shadow:0 4px 12px rgba(0,0,0,.08); }
  h2 { margin:0 0 8px; text-align:center; }
  p.sub { color:#15803d; font-size:.9rem; text-align:center; margin-bottom:24px; }
  .fg { margin-bottom:18px; }
  label { display:block; margin-bottom:6px; font-weight:600; font-size:.9rem; }
  input { width:100%; padding:10px 12px; border:1px solid var(--border); border-radius:4px;
          font-size:.95rem; box-sizing:border-box; }
  button { background:var(--primary); color:#fff; border:none; padding:12px; width:100%;
           border-radius:4px; font-size:1rem; font-weight:bold; cursor:pointer; }
  button:disabled { opacity:0.6; cursor:not-allowed; }
  .link-row { margin-top:12px; text-align:center; font-size:.9rem; }
  .err { color:var(--danger); font-size:.85rem; margin-bottom:12px; text-align:center; }
  .policy-box { font-size:.85rem; color:#64748b; line-height:1.45; margin:10px 0 18px; }
  .policy-list { margin:8px 0 0 16px; padding:0; list-style:disc; }
  .policy-item { color:#64748b; }
  .policy-item.ok { color:var(--success); }
  .face-box { margin-top:16px; padding:12px; border:1px solid var(--border); border-radius:6px; background:rgba(0,0,0,.03); }
  .face-video { width:100%; border-radius:6px; background:#0f172a; display:block; }
</style>
</head>
<body>
<div class="card">
  <h2>Create Account</h2>
  <p class="sub">Fill in your details to create a new account.</p>
  {% if error %}<p class="err">{{ error }}</p>{% endif %}
  <form id="registerForm" method="POST" action="{{ url_for('register_page') }}">
    <div class="fg">
      <label for="username">Username</label>
      <input type="text" id="username" name="username" required>
    </div>
    <div class="fg">
      <label for="email">Email</label>
      <input type="email" id="email" name="email" required>
    </div>
    <div class="fg">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" required minlength="8">
      <div class="policy-box">
        <div><b>Password policy:</b> {{ PASSWORD_POLICY }}</div>
        <ul class="policy-list">
          <li id="passwordRuleLength" class="policy-item">○ At least 8 characters</li>
          <li id="passwordRuleUppercase" class="policy-item">○ At least one uppercase letter</li>
          <li id="passwordRuleSpecial" class="policy-item">○ At least one special character</li>
          <li id="passwordRuleMatch" class="policy-item">○ Passwords match</li>
        </ul>
      </div>
    </div>
    <div class="fg">
      <label for="confirm_password">Confirm Password</label>
      <input type="password" id="confirm_password" name="confirm_password" required minlength="8">
    </div>
    <div class="face-box">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px;">
        <strong>Face Enrollment</strong>
        <button id="faceEnrollBtn" type="button" style="width:auto; padding:8px 12px;">Enable Camera</button>
      </div>
      <p id="faceEnrollHint" style="margin:0 0 8px; font-size:.8rem; color:#64748b;">Blink once when the camera is active to capture your face automatically.</p>
      <video id="faceEnrollVideo" class="face-video" autoplay playsinline muted></video>
      <canvas id="faceEnrollCanvas" style="display:none;"></canvas>
      <p id="faceEnrollStatus" style="margin:8px 0 0; font-size:.85rem; color:#64748b;">Capture a face during registration so you can sign in with Face Login later.</p>
    </div>
    <button id="createAccountBtn" type="submit">Create Account</button>
  </form>
  <div class="link-row">
    <a href="{{ url_for('login_page') }}">Back to sign in</a>
  </div>
</div>
<script>
  const passwordInput = document.getElementById('password');
  const confirmPasswordInput = document.getElementById('confirm_password');
  const submitButton = document.getElementById('createAccountBtn');
  const faceEnrollBtn = document.getElementById('faceEnrollBtn');
  const faceEnrollVideo = document.getElementById('faceEnrollVideo');
  const faceEnrollCanvas = document.getElementById('faceEnrollCanvas');
  const faceEnrollStatus = document.getElementById('faceEnrollStatus');
  const faceEnrollHint = document.getElementById('faceEnrollHint');
  const passwordRuleLength = document.getElementById('passwordRuleLength');
  const passwordRuleUppercase = document.getElementById('passwordRuleUppercase');
  const passwordRuleSpecial = document.getElementById('passwordRuleSpecial');
  const passwordRuleMatch = document.getElementById('passwordRuleMatch');

  function setPasswordRule(element, ok) {
    if (!element) return;
    element.textContent = (ok ? '✓ ' : '○ ') + element.textContent.replace(/^[✓○]\s/, '');
    element.classList.toggle('ok', ok);
  }

  function validateRegistrationForm() {
    const password = passwordInput.value;
    const confirmPassword = confirmPasswordInput.value;
    const isLengthOk = password.length >= 8;
    const hasUppercase = /[A-Z]/.test(password);
    const hasSpecial = /[^A-Za-z0-9]/.test(password);
    const passwordsMatch = password !== '' && password === confirmPassword;

    setPasswordRule(passwordRuleLength, isLengthOk);
    setPasswordRule(passwordRuleUppercase, hasUppercase);
    setPasswordRule(passwordRuleSpecial, hasSpecial);
    setPasswordRule(passwordRuleMatch, passwordsMatch);

    submitButton.disabled = !(isLengthOk && hasUppercase && hasSpecial && passwordsMatch);
    return isLengthOk && hasUppercase && hasSpecial && passwordsMatch;
  }

  [passwordInput, confirmPasswordInput].forEach(input => {
    input.addEventListener('input', validateRegistrationForm);
  });

  let faceEnrollStream = null;
  let faceEnrollCaptured = false;

  async function captureCurrentFrameBlob() {
    const video = faceEnrollVideo;
    if (!video || !video.videoWidth || !video.videoHeight) {
      throw new Error('Camera is not ready yet.');
    }

    faceEnrollCanvas.width = video.videoWidth;
    faceEnrollCanvas.height = video.videoHeight;
    const ctx = faceEnrollCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, faceEnrollCanvas.width, faceEnrollCanvas.height);

    return await new Promise((resolve, reject) => {
      faceEnrollCanvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('Could not capture image')), 'image/jpeg', 0.9);
    });
  }

  async function enrollCapturedPhoto(blob) {
    faceEnrollBtn.disabled = true;
    faceEnrollStatus.textContent = 'Blink detected. Capturing face…';
    try {
      const formData = new FormData(document.getElementById('registerForm'));
      formData.append('face_photo', blob, 'enroll.jpg');
      const response = await fetch('/api/enroll-face', { method: 'POST', body: formData });
      const data = await response.json();
      if (data.status === 'success') {
        faceEnrollCaptured = true;
        faceEnrollStatus.textContent = 'Face enrolled successfully.';
        faceEnrollHint.textContent = 'Face enrollment completed.';
      } else {
        faceEnrollStatus.textContent = data.message || 'Face enrollment failed.';
        faceEnrollHint.textContent = 'Blink once again if you want to try again.';
      }
    } catch (err) {
      faceEnrollStatus.textContent = 'Face enrollment failed: ' + (err.message || err);
      faceEnrollHint.textContent = 'Blink once again if you want to try again.';
    } finally {
      faceEnrollBtn.disabled = false;
    }
  }

  async function enableFaceEnrollment() {
    if (!faceEnrollStream) {
      try {
        faceEnrollStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        faceEnrollVideo.srcObject = faceEnrollStream;
        faceEnrollStatus.textContent = 'Camera ready. Blink once to capture your face.';
        faceEnrollHint.textContent = 'Blink once when the camera is active to capture your face automatically.';
        faceEnrollBtn.textContent = 'Start Enrollment';
      } catch (err) {
        faceEnrollStatus.textContent = 'Camera access was denied or unavailable.';
      }
      return;
    }

    if (faceEnrollCaptured) {
      faceEnrollStatus.textContent = 'Face already captured for this registration.';
      return;
    }

    faceEnrollStatus.textContent = 'Waiting for blink…';
    faceEnrollHint.textContent = 'Blink once to capture your face automatically.';

    for (let attempt = 0; attempt < 5 && !faceEnrollCaptured; attempt += 1) {
      try {
        const blob = await captureCurrentFrameBlob();
        const response = await fetch('/api/check-blink', { method: 'POST', body: (() => {
          const formData = new FormData();
          formData.append('photo', blob, 'blink-check.jpg');
          return formData;
        })() });
        const data = await response.json();

        if (data.status === 'success' && data.blink_detected) {
          await enrollCapturedPhoto(blob);
          return;
        }
      } catch (err) {
        faceEnrollStatus.textContent = 'Blink detection failed: ' + (err.message || err);
        faceEnrollHint.textContent = 'Blink once again if you want to try again.';
        return;
      }

      if (attempt < 4) {
        faceEnrollStatus.textContent = `No blink detected yet (${attempt + 1}/5). Blink once and hold still.`;
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
    }

    faceEnrollStatus.textContent = 'No blink detected. Please try again.';
    faceEnrollHint.textContent = 'Blink once again if you want to try again.';
    faceEnrollBtn.disabled = false;
  }

  if (faceEnrollBtn) {
    faceEnrollBtn.addEventListener('click', enableFaceEnrollment);
  }

  validateRegistrationForm();
</script>
</body>
</html>
"""

PROFILE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Profile Information Management</title>
<style>
  :root {
  --primary:#16a34a; --bg:#f8fafc; --card:#fff; --text:#1e293b;
  --border:#cbd5e1; --mic-active:#ef4444; --focus-ring:#16a34a;
}
  body.high-contrast {
    --primary:#facc15; --bg:#000; --card:#121212; --text:#fff; --border:#fff;
  }
  body { font-family: system-ui, sans-serif; background:var(--bg); color:var(--text);
         margin:0; padding:20px; display:flex; justify-content:center; }
  .container { background:var(--card); border:2px solid var(--border); border-radius:8px;
               padding:30px; width:100%; max-width:620px; box-shadow:0 4px 6px -1px rgba(0,0,0,.1); }
  .header-bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; }
  h2 { margin:0; }
  .btn-logout { background:transparent; color:#ef4444; border:1px solid #ef4444; padding:6px 14px;
                border-radius:4px; cursor:pointer; font-weight:600; font-size:.9rem; }
  .btn-logout:hover { background:#ef4444; color:#fff; }

  /* ---- FR-5: Webcam Profile Picture Capture ---- */
  .capture-section { display:flex; flex-direction:column; align-items:center; gap:10px;
                      margin-bottom:24px; padding:16px; background:rgba(0,0,0,.03); border-radius:8px; }
  .viewport-wrap { position:relative; width:320px; height:240px; background:#000; border-radius:6px;
                    overflow:hidden; }
  #videoFeed, #captureCanvas, #overlayCanvas, #previewImg {
    position:absolute; top:0; left:0; width:320px; height:240px; object-fit:cover;
  }
  #captureCanvas, #previewImg { display:none; }
  #overlayCanvas { pointer-events:none; }
  .capture-controls { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }
  .btn { border:none; padding:9px 16px; border-radius:5px; font-weight:600; cursor:pointer; font-size:.85rem; }
  .btn:disabled { opacity:.45; cursor:not-allowed; }
  .btn-primary { background:var(--primary); color:#fff; }
  .btn-secondary { background:transparent; color:var(--primary); border:1px solid var(--primary); }
  .btn-success { background:#16a34a; color:#fff; }
  .capture-status { font-size:.85rem; color:#64748b; min-height:1.2em; text-align:center; }
  .profile-photo-thumb { width:64px; height:64px; border-radius:50%; object-fit:cover;
                          border:2px solid var(--border); vertical-align:middle; margin-right:10px; }

  /* ---- FR-4: Voice-Guided Data Entry ---- */
  .speech-controls { display:flex; align-items:center; gap:12px; margin-bottom:20px; padding:12px;
                      background:rgba(0,0,0,.03); border-radius:6px; }
  .mic-btn { background:none; border:2px solid var(--border); border-radius:50%; width:44px; height:44px;
             cursor:pointer; display:flex; align-items:center; justify-content:center; font-size:1.2rem; }
  .mic-btn.listening { background:var(--mic-active); border-color:var(--mic-active); color:#fff;
                        animation:pulse 1.5s infinite; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(239,68,68,.7);} 70%{box-shadow:0 0 0 12px rgba(239,68,68,0);}
                      100%{box-shadow:0 0 0 0 rgba(239,68,68,0);} }
  .transcription-tooltip { font-style:italic; font-size:.9rem; opacity:.8; flex-grow:1;
                            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  /* ---- FR-3: Profile form ---- */
  .fg { margin-bottom:18px; }
  label { display:block; margin-bottom:6px; font-weight:600; }
  input, select, textarea { width:100%; padding:10px; border:1px solid var(--border); border-radius:4px;
                             background:var(--card); color:var(--text); box-sizing:border-box; font-size:1rem; }
  input:focus, select:focus, textarea:focus { outline:3px solid var(--focus-ring); }
  .btn-submit { background:var(--primary); color:#fff; border:none; padding:12px 20px; font-size:1rem;
                font-weight:bold; border-radius:4px; cursor:pointer; width:100%; margin-top:10px; }
  .save-indicator { font-size:.8rem; color:#16a34a; text-align:right; min-height:1.1em; margin-top:-10px; margin-bottom:14px; }
  .command-guide { margin-top:25px; font-size:.85rem; border-top:1px solid var(--border); padding-top:15px; }
</style>
</head>
<body>
<div class="container">
  <div class="header-bar">
    <h2>Profile Information</h2>
    <button id="logoutBtn" type="button" class="btn-logout">Logout</button>
  </div>

  <!-- =================================================================
       FR-5: Webcam Profile Picture Capture
       FR-5.1 Embedded video viewport
       FR-5.2 Framing assist overlay (oval head/shoulder guide)
       FR-5.3 Snap / Retake / Crop & Confirm lifecycle
       FR-5.4 Client-side compression, max 2MB, JPEG/WebP
       ================================================================= -->
  <div class="capture-section">
    <div class="viewport-wrap">
      <video id="videoFeed" autoplay playsinline muted></video>
      <canvas id="overlayCanvas" width="320" height="240"></canvas>
      <canvas id="captureCanvas" width="320" height="240"></canvas>
      <img id="previewImg" alt="Captured profile preview">
    </div>
    <div class="capture-controls">
      <button type="button" id="btnSnap" class="btn btn-primary">📸 Snap Photo</button>
      <button type="button" id="btnRetake" class="btn btn-secondary" disabled>🔄 Retake</button>
      <button type="button" id="btnConfirm" class="btn btn-success" disabled>✂️ Crop &amp; Confirm</button>
    </div>
    <p id="captureStatus" class="capture-status">Align your head and shoulders inside the guide.</p>
    {% if photo_url %}
      <img class="profile-photo-thumb" id="currentPhotoThumb" src="{{ photo_url }}" alt="Current profile photo">
    {% endif %}
  </div>

  <!-- =================================================================
       FR-4: Voice-Guided Data Entry (Speech-to-Text)
       FR-4.1 Continuous speech engine
       FR-4.2 Verbal field activation + dictation
       FR-4.3 Pulsing mic indicator + live transcription tooltip
       FR-4.4 "Clear Field" / "Delete Last Word" / "Next Field" / "Submit Profile"
       ================================================================= -->
  <div class="speech-controls">
    <button type="button" id="micToggle" class="mic-btn" title="Toggle Voice Recognition">🎙️</button>
    <span id="transcriptionTooltip" class="transcription-tooltip">Click mic or speak to start...</span>
  </div>

  <!-- =================================================================
       FR-3: Profile Information Management
       FR-3.2 Structured fields
       FR-3.3 Auto-save to session storage after each field update
       ================================================================= -->
  <form id="profileForm" method="POST" action="/submit-profile">
    <div class="fg">
      <label for="fullName">Enter Full Name</label>
      <input type="text" id="fullName" name="full_name" value="{{ profile.full_name or '' }}" required>
    </div>
    <div class="fg">
      <label for="contactPhone">Contact Phone</label>
      <input type="tel" id="contactPhone" name="contact_phone" value="{{ profile.contact_phone or '' }}" required>
    </div>
    <div class="fg">
      <label for="emailAddress">Email Address</label>
      <input type="email" id="emailAddress" name="email_address" value="{{ profile.email_address or '' }}" required>
    </div>
    <div class="fg">
      <label for="dob">Date of Birth</label>
      <input type="date" id="dob" name="dob" value="{{ profile.dob or '' }}" required>
    </div>
    <div class="fg">
      <label for="physicalAddress">Physical Address</label>
      <textarea id="physicalAddress" name="physical_address" rows="3" required>{{ profile.physical_address or '' }}</textarea>
    </div>
    <div class="fg">
      <label for="accessibilityMode">Accessibility Mode Preferences</label>
      <select id="accessibilityMode" name="accessibility_mode">
        <option value="default" {{ 'selected' if profile.accessibility_mode == 'default' else '' }}>Default</option>
        <option value="high-contrast" {{ 'selected' if profile.accessibility_mode == 'high-contrast' else '' }}>High Contrast</option>
        <option value="large-text" {{ 'selected' if profile.accessibility_mode == 'large-text' else '' }}>Large Text</option>
      </select>
    </div>
    <p id="saveIndicator" class="save-indicator"></p>
    <button type="submit" class="btn-submit">Submit Profile</button>
  </form>

  <div class="command-guide">
    <strong>Voice Commands:</strong>
    <p>&bull; <em>"Select [Field Name]"</em> | <em>"Clear Field"</em> | <em>"Delete Last Word"</em></p>
    <p>&bull; <em>"Next Field"</em> | <em>"Submit Profile"</em> | <em>"Logout"</em></p>
    <p>&bull; <em>"Take Photo"</em> | <em>"Retake Photo"</em> | <em>"Confirm Photo"</em></p>
  </div>
</div>

<script>
/* =====================================================================
   FR-3.3  Auto-Save State
   Form progress is written to sessionStorage on every field change and
   restored on load, so an accidental refresh/navigation doesn't lose data.
   ===================================================================== */
const form = document.getElementById('profileForm');
const inputs = form.querySelectorAll('input, select, textarea');
const saveIndicator = document.getElementById('saveIndicator');
const SESSION_KEY_PREFIX = 'profileForm:';

function restoreFormState() {
  inputs.forEach(input => {
    const saved = sessionStorage.getItem(SESSION_KEY_PREFIX + input.name);
    if (saved !== null) {
      input.value = saved;
      if (input.id === 'accessibilityMode') applyAccessibilityMode(saved);
    }
  });
}

function persistField(input) {
  sessionStorage.setItem(SESSION_KEY_PREFIX + input.name, input.value);
  saveIndicator.innerText = 'Saved ' + new Date().toLocaleTimeString();
  if (input.id === 'accessibilityMode') applyAccessibilityMode(input.value);
}

window.addEventListener('DOMContentLoaded', restoreFormState);
inputs.forEach(input => input.addEventListener('input', () => persistField(input)));

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(form);
  try {
    const resp = await fetch('/submit-profile', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.status === 'success') {
      saveIndicator.innerText = 'Profile submitted successfully!';
      saveIndicator.style.color = '#16a34a';
    } else {
      saveIndicator.innerText = 'Submission failed: ' + (data.message || 'unknown error');
      saveIndicator.style.color = '#dc2626';
    }
  } catch (err) {
    saveIndicator.innerText = 'Error: ' + err;
    saveIndicator.style.color = '#dc2626';
  }
});

function applyAccessibilityMode(mode) {
  document.body.classList.toggle('high-contrast', mode === 'high-contrast');
  document.body.style.fontSize = (mode === 'large-text') ? '1.2rem' : '';
}

function performLogout() {
  sessionStorage.clear();
  window.location.href = '/logout';
}
document.getElementById('logoutBtn').addEventListener('click', performLogout);

/* =====================================================================
   FR-5  Webcam Profile Picture Capture
   ===================================================================== */
const video = document.getElementById('videoFeed');
const overlay = document.getElementById('overlayCanvas');
const overlayCtx = overlay.getContext('2d');
const captureCanvas = document.getElementById('captureCanvas');
const captureCtx = captureCanvas.getContext('2d');
const previewImg = document.getElementById('previewImg');
const btnSnap = document.getElementById('btnSnap');
const btnRetake = document.getElementById('btnRetake');
const btnConfirm = document.getElementById('btnConfirm');
const captureStatus = document.getElementById('captureStatus');

let mediaStream = null;
let capturedDataUrl = null;

// FR-5.1: Embedded live video feed
async function startWebcam() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
    video.srcObject = mediaStream;
    video.style.display = 'block';
    captureCanvas.style.display = 'none';
    previewImg.style.display = 'none';
    drawFramingGuide();
  } catch (err) {
    captureStatus.innerText = 'Camera unavailable: ' + err.message;
  }
}

// FR-5.2: Framing assist overlay — oval head-and-shoulders guide
function drawFramingGuide() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.strokeStyle = '#ffffff';
  overlayCtx.lineWidth = 2;

  // Head guide
  overlayCtx.beginPath();
  overlayCtx.ellipse(160, 95, 55, 75, 0, 0, Math.PI * 2);
  overlayCtx.stroke();

  // Shoulder guide (lower arc)
  overlayCtx.beginPath();
  overlayCtx.ellipse(160, 210, 100, 45, 0, Math.PI, Math.PI * 2);
  overlayCtx.stroke();
}

// FR-5.3: Snap Photo — freeze current frame to canvas
btnSnap.addEventListener('click', () => {
  if (!mediaStream) return;
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  video.style.display = 'none';
  overlay.style.display = 'none';
  captureCanvas.style.display = 'block';

  btnSnap.disabled = true;
  btnRetake.disabled = false;
  btnConfirm.disabled = false;
  captureStatus.innerText = 'Photo captured. Retake or Crop & Confirm.';
});

// FR-5.3: Retake — resume live feed
btnRetake.addEventListener('click', () => {
  captureCanvas.style.display = 'none';
  overlay.style.display = 'block';
  video.style.display = 'block';
  previewImg.style.display = 'none';
  capturedDataUrl = null;

  btnSnap.disabled = false;
  btnRetake.disabled = true;
  btnConfirm.disabled = true;
  captureStatus.innerText = 'Align your head and shoulders inside the guide.';
});

// FR-5.3 + FR-5.4: Crop (center square) & Confirm, then compress under 2MB
btnConfirm.addEventListener('click', async () => {
  captureStatus.innerText = 'Processing photo...';
  const cropSize = Math.min(captureCanvas.width, captureCanvas.height);
  const sx = (captureCanvas.width - cropSize) / 2;
  const sy = (captureCanvas.height - cropSize) / 2;

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropSize;
  cropCanvas.height = cropSize;
  cropCanvas.getContext('2d').drawImage(
    captureCanvas, sx, sy, cropSize, cropSize, 0, 0, cropSize, cropSize
  );

  const blob = await compressUnder2MB(cropCanvas);
  if (!blob) {
    captureStatus.innerText = 'Could not compress photo under 2MB. Try better lighting / retake.';
    return;
  }

  capturedDataUrl = URL.createObjectURL(blob);
  previewImg.src = capturedDataUrl;
  previewImg.style.display = 'block';
  captureCanvas.style.display = 'none';

  await uploadPhoto(blob);
});

// FR-5.4: Automated image optimization — WebP (fallback JPEG), max 2MB, client-side
function compressUnder2MB(sourceCanvas) {
  const MAX_BYTES = 2 * 1024 * 1024;
  const mimeCandidates = ['image/webp', 'image/jpeg'];

  return new Promise(resolve => {
    (function tryMime(mimeIndex) {
      if (mimeIndex >= mimeCandidates.length) return resolve(null);
      const mime = mimeCandidates[mimeIndex];

      (function tryQuality(quality) {
        sourceCanvas.toBlob(blob => {
          if (!blob) return tryMime(mimeIndex + 1);
          if (blob.size <= MAX_BYTES || quality <= 0.2) {
            if (blob.size <= MAX_BYTES) return resolve(blob);
            return tryMime(mimeIndex + 1);
          }
          tryQuality(quality - 0.1);
        }, mime, quality);
      })(0.9);
    })(0);
  });
}

async function uploadPhoto(blob) {
  captureStatus.innerText = 'Uploading photo...';
  const formData = new FormData();
  formData.append('photo', blob, 'profile.webp');
  try {
    const resp = await fetch('/api/upload-photo', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.status === 'success') {
      captureStatus.innerText = `Photo saved (${(blob.size / 1024).toFixed(1)} KB).`;
    } else {
      captureStatus.innerText = 'Upload failed: ' + (data.message || 'unknown error');
    }
  } catch (err) {
    captureStatus.innerText = 'Upload error: ' + err;
  }
}

startWebcam();

/* =====================================================================
   FR-4  Voice-Guided Data Entry (Speech-to-Text)
   ===================================================================== */
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn = document.getElementById('micToggle');
const tooltip = document.getElementById('transcriptionTooltip');

if (!SpeechRecognition) {
  tooltip.innerText = 'Speech API not supported in this browser.';
} else {
  const recognition = new SpeechRecognition();
  recognition.continuous = true;       // FR-4.1
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  let isListening = false;

  micBtn.addEventListener('click', () => isListening ? recognition.stop() : recognition.start());

  recognition.onstart = () => {           // FR-4.3
    isListening = true;
    micBtn.classList.add('listening');
    tooltip.innerText = 'Listening...';
  };
  recognition.onend = () => {
    isListening = false;
    micBtn.classList.remove('listening');
    tooltip.innerText = 'Microphone paused. Click to listen.';
  };

  recognition.onresult = (event) => {
    const r = event.results[event.resultIndex];
    const transcript = r[0].transcript.trim();
    tooltip.innerText = `"${transcript}"`;
    if (r.isFinal) processVoiceCommand(transcript.toLowerCase());
  };

  // FR-4.2 / FR-4.4: verbal field activation + dictation commands
  function processVoiceCommand(command) {
    const activeElem = document.activeElement;

    if (command === 'logout' || command === 'log out') return performLogout();
    if (command === 'submit profile') return form.requestSubmit();

    if (command === 'take photo' || command === 'snap photo') return btnSnap.click();
    if (command === 'retake photo') return btnRetake.click();
    if (command === 'confirm photo' || command === 'crop and confirm') return btnConfirm.click();

    if (command === 'clear field') {
      if (isInputField(activeElem)) { activeElem.value = ''; triggerInputEvent(activeElem); }
      return;
    }
    if (command === 'delete last word') {
      if (isInputField(activeElem) && activeElem.value) {
        const words = activeElem.value.trim().split(' ');
        words.pop();
        activeElem.value = words.join(' ');
        triggerInputEvent(activeElem);
      }
      return;
    }
    if (command === 'next field') return focusNextField(activeElem);

    const fieldMap = {
      'select full name': 'fullName', 'select Enter full name': 'fullName',
      'select contact phone': 'contactPhone', 'select phone': 'contactPhone',
      'select email address': 'emailAddress', 'select email': 'emailAddress',
      'select date of birth': 'dob',
      'select physical address': 'physicalAddress', 'select address': 'physicalAddress',
      'select accessibility mode': 'accessibilityMode',
    };
    if (fieldMap[command]) return document.getElementById(fieldMap[command]).focus();

    if (isInputField(activeElem)) {
      if (activeElem.type === 'date') {
        activeElem.value = command.replace(/\\s+/g, '');
      } else if (activeElem.tagName === 'SELECT') {
        Array.from(activeElem.options).forEach(opt => {
          if (opt.text.toLowerCase().includes(command)) activeElem.value = opt.value;
        });
      } else {
        activeElem.value = (activeElem.value ? activeElem.value + ' ' : '') + command;
      }
      triggerInputEvent(activeElem);
    }
  }

  function isInputField(el) { return el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName); }
  function triggerInputEvent(el) { el.dispatchEvent(new Event('input', { bubbles: true })); }
  function focusNextField(current) {
    const els = Array.from(inputs);
    const i = els.indexOf(current);
    els[i > -1 && i < els.length - 1 ? i + 1 : 0].focus();
  }
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if session.get("authenticated"):
        return redirect(url_for("profile_page"))
    return redirect(url_for("login_page"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    """FR-3.1 groundwork: standard credential auth, then redirect to profile."""
    error = None
    if request.method == "POST":
        identifier = (request.form.get("username") or request.form.get("identifier") or "").strip()
        password = request.form.get("password") or ""

        if not identifier or not password:
            error = "Username/email and password are required."
        else:
            username = find_user_by_identifier(identifier)
            existing = USERS.get(username, {}) if username else None
            if existing is None:
                error = "Account not found. Please register first."
            elif not password_matches(existing.get("password", ""), password):
                error = "Incorrect password."

            if error is None:
                session["authenticated"] = True
                session["username"] = username
                return redirect(url_for("profile_page"))  # FR-3.1

    if session.get("authenticated"):
        return redirect(url_for("profile_page"))
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/register", methods=["GET", "POST"])
def register_page():
    """Open a dedicated account-creation screen and create a new account on submit."""
    error = None
    if request.method == "POST":
        username = clean_username(request.form.get("username") or "")
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        email = (request.form.get("email") or "").strip()

        if not username or not password or not confirm_password:
            error = "Username, password and confirmation are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif email and not EMAIL_PATTERN.match(email):
            error = "Please enter a valid email address."
        elif USERS.get(username) is not None:
            error = "Username already exists. Please sign in instead."
        else:
            valid, password_error = validate_password(password)
            if not valid:
                error = password_error
            else:
                profile = {}
                if email:
                    profile["email_address"] = email
                USERS[username] = {
                    "password": generate_password_hash(password),
                    "email": email,
                    "profile": profile,
                }
                save_users(USERS)
                session["authenticated"] = True
                session["username"] = username
                return redirect(url_for("profile_page"))

    if session.get("authenticated"):
        return redirect(url_for("profile_page"))
    return render_template_string(
        REGISTRATION_HTML,
        error=error,
        PASSWORD_POLICY=PASSWORD_POLICY,
    )


@app.route("/api/register", methods=["POST"])
def api_register():
    payload = request.get_json(silent=True) or {}
    username = clean_username(payload.get("username") or "")
    password = payload.get("password") or ""
    confirm_password = payload.get("confirm_password") or ""
    email = (payload.get("email") or "").strip()

    if not username or not password or not confirm_password:
        return jsonify({"status": "error", "message": "Username, password and confirmation are required."}), 400
    if password != confirm_password:
        return jsonify({"status": "error", "message": "Passwords do not match."}), 400
    if email and not EMAIL_PATTERN.match(email):
        return jsonify({"status": "error", "message": "Please enter a valid email address."}), 400
    if USERS.get(username) is not None:
        return jsonify({"status": "error", "message": "Username already exists."}), 409

    valid, password_error = validate_password(password)
    if not valid:
        return jsonify({"status": "error", "message": password_error}), 400

    profile = {}
    if email:
        profile["email_address"] = email
    USERS[username] = {
        "password": generate_password_hash(password),
        "email": email,
        "profile": profile,
    }
    save_users(USERS)
    session["authenticated"] = True
    session["username"] = username
    return jsonify({"status": "success", "message": "Account created successfully."})


@app.route("/api/login", methods=["POST"])
def api_login():
    payload = request.get_json(silent=True) or {}
    identifier = (payload.get("identifier") or payload.get("username") or "").strip()
    password = payload.get("password") or ""

    if not identifier or not password:
        return jsonify({"status": "error", "message": "Identifier and password are required."}), 400

    username = find_user_by_identifier(identifier)
    existing = USERS.get(username, {}) if username else None
    if existing is None:
        return jsonify({"status": "error", "message": "Account not found."}), 404
    if not password_matches(existing.get("password", ""), password):
        return jsonify({"status": "error", "message": "Incorrect password."}), 401

    session["authenticated"] = True
    session["username"] = username
    return jsonify({"status": "success", "message": "Login successful."})


@app.route("/api/face-login", methods=["POST"])
def api_face_login():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"status": "error", "message": "No face photo provided."}), 400

    username = (request.form.get("username") or "").strip()
    success, message = verify_face_login(photo, username=username or None)
    if success:
        session["authenticated"] = True
        session["username"] = clean_username(username) or "face-user"
        return jsonify({"status": "success", "message": message})
    return jsonify({"status": "error", "message": message}), 401


@app.route("/api/check-blink", methods=["POST"])
def api_check_blink():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"status": "error", "message": "No face photo provided."}), 400

    try:
        blink_detected = detect_blink_from_photo(photo)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Unable to detect blink: {exc}"}), 400

    return jsonify({"status": "success", "blink_detected": blink_detected})


@app.route("/api/enroll-face", methods=["POST"])
def api_enroll_face():
    username = clean_username(request.form.get("username") or "")
    photo = request.files.get("face_photo")
    if not username:
        return jsonify({"status": "error", "message": "Username is required for face enrollment."}), 400
    if not photo:
        return jsonify({"status": "error", "message": "No face photo provided."}), 400

    try:
        embedding = extract_face_embedding(photo)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Unable to process the uploaded image: {exc}"}), 400

    try:
        save_enrolled_face_embedding(embedding)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Face enrollment failed: {exc}"}), 500

    USERS[username] = USERS.get(username, {"password": "", "email": "", "profile": {}})
    USERS[username]["face_embedding"] = embedding.tolist()
    save_users(USERS)
    return jsonify({"status": "success", "message": "Face enrolled successfully."})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


def _photo_path(username):
    safe = re.sub(r"[^A-Za-z0-9_-]", "", username) or "user"
    return str(UPLOAD_FOLDER / f"{safe}.webp")


def _photo_url(username):
    return url_for("get_photo", username=username) if os.path.exists(_photo_path(username)) else None


@app.route("/profile")
def profile_page():
    """FR-3.1: post-login redirection target. Renders the multimodal hub."""
    if not session.get("authenticated"):
        return redirect(url_for("login_page"))
    username = session["username"]
    user_data = USERS.get(username, {})
    profile = dict(user_data.get("profile", {}) or {})
    email_from_account = (user_data.get("email") or "").strip()
    if email_from_account and not profile.get("email_address"):
        profile["email_address"] = email_from_account
    return render_template_string(PROFILE_HTML, profile=profile, photo_url=_photo_url(username))


@app.route("/submit-profile", methods=["POST"])
def submit_profile():
    """FR-3.2: persist the structured profile fields."""
    if not session.get("authenticated"):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    username = session["username"]
    profile = {field: request.form.get(field, "") for field in PROFILE_FIELDS}
    user_data = USERS.setdefault(username, {"password": "", "email": "", "profile": {}})
    user_data["profile"] = profile
    save_users(USERS)
    return jsonify({"status": "success", "message": "Profile submitted successfully", "data": profile})


@app.route("/api/upload-photo", methods=["POST"])
def upload_photo():
    """FR-5.4: receives the client-compressed (<=2MB) profile photo."""
    if not session.get("authenticated"):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    photo = request.files.get("photo")
    if not photo:
        return jsonify({"status": "error", "message": "No photo included"}), 400

    photo.seek(0, os.SEEK_END)
    size = photo.tell()
    photo.seek(0)
    if size > MAX_PHOTO_BYTES:
        return jsonify({"status": "error", "message": "Photo exceeds 2MB limit"}), 413

    photo.save(_photo_path(session["username"]))
    return jsonify({"status": "success"})


@app.route("/photo/<username>")
def get_photo(username):
    if not session.get("authenticated"):
        return redirect(url_for("login_page"))
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(_photo_path(username)))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
