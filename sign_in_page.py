"""
Inclusive Multimodal User Onboarding & Profile Management System
------------------------------------------------------------------
Implements SRS-2026-MM-01 sections:
  3.3  Profile Information Management (FR-3.1 - FR-3.3)
  3.4  Voice-Guided Data Entry / Speech-to-Text        (FR-4.1 - FR-4.4)
  3.5  Webcam Profile Picture Capture                  (FR-5.1 - FR-5.4)

The webcam capture is implemented as an EMBEDDED browser viewport
(getUserMedia + <canvas>) directly on the Profile page, per FR-5.1,
rather than a separate desktop process — this matches the SRS's
"Web / Progressive Web App" target platform.
"""

import os
import re
import uuid

from flask import (
    Flask, render_template_string, request, redirect, url_for,
    jsonify, session, send_from_directory
)

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "profile_photos")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_PHOTO_BYTES = 2 * 1024 * 1024  # FR-5.4: 2MB cap, enforced server-side too

# In-memory "database" for this reference implementation.
# FR-3.2: structured profile fields per user.
USERS = {}          # username -> {"password": ..., "profile": {...}}
PROFILE_FIELDS = [
    "full_name", "contact_phone", "email_address", "dob",
    "physical_address", "accessibility_mode",
]


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
  <form method="POST">
    <div class="fg"><label for="username">Username</label>
      <input type="text" id="username" name="username" required></div>
    <div class="fg"><label for="password">Password</label>
      <input type="password" id="password" name="password" required></div>
    <button type="submit">Sign In / Register</button>
  </form>
</div>
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
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            error = "Username and password are required."
        else:
            user = USERS.get(username)
            if user is None:
                USERS[username] = {"password": password, "profile": {}}
            elif user["password"] != password:
                error = "Incorrect password."

            if error is None:
                session["authenticated"] = True
                session["username"] = username
                return redirect(url_for("profile_page"))  # FR-3.1

    if session.get("authenticated"):
        return redirect(url_for("profile_page"))
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


def _photo_path(username):
    safe = re.sub(r"[^A-Za-z0-9_-]", "", username) or "user"
    return os.path.join(UPLOAD_FOLDER, f"{safe}.webp")


def _photo_url(username):
    return url_for("get_photo", username=username) if os.path.exists(_photo_path(username)) else None


@app.route("/profile")
def profile_page():
    """FR-3.1: post-login redirection target. Renders the multimodal hub."""
    if not session.get("authenticated"):
        return redirect(url_for("login_page"))
    username = session["username"]
    profile = USERS.get(username, {}).get("profile", {})
    return render_template_string(PROFILE_HTML, profile=profile, photo_url=_photo_url(username))


@app.route("/submit-profile", methods=["POST"])
def submit_profile():
    """FR-3.2: persist the structured profile fields."""
    if not session.get("authenticated"):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    username = session["username"]
    profile = {field: request.form.get(field, "") for field in PROFILE_FIELDS}
    USERS.setdefault(username, {"password": "", "profile": {}})
    USERS[username]["profile"] = profile
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
    app.run(debug=True, port=5000)