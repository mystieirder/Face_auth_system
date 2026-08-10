# Face Auth — User Registration & Biometric Enrollment

Implements:
- **FR-1.1** Standard Registration (email/username + password)
- **FR-1.2** Biometric Enrollment (webcam captures a baseline **facial landmark embedding vector** — raw images are never stored)
- **FR-1.3** Environmental Validation (live face presence / lighting / frame-positioning checks, re-validated server-side)
- **FR-1.4** Password Policy (min 8 chars, letters + numbers + special character)

## Project Structure

```
face-auth-system/
├── backend/
│   ├── config/
│   │   └── db.js                # MongoDB connection
│   ├── controllers/
│   │   └── authController.js    # register/login logic
│   ├── middleware/
│   │   └── validatePassword.js  # FR-1.4 password policy
│   ├── models/
│   │   └── User.js              # stores embedding vector, not images
│   ├── routes/
│   │   └── authRoutes.js
│   ├── utils/
│   │   └── faceUtils.js         # embedding distance + env re-validation
│   ├── .env.example
│   ├── package.json
│   └── server.js
├── frontend/
│   ├── public/
│   │   ├── models/               # place face-api.js model weights here
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── EnvironmentValidator.js  # FR-1.3 live UI feedback
│   │   │   ├── RegisterForm.js          # FR-1.1 + FR-1.4
│   │   │   └── WebcamCapture.js         # FR-1.2 + FR-1.3
│   │   ├── services/
│   │   │   ├── authService.js
│   │   │   └── faceApiService.js        # loads models, extracts embedding
│   │   ├── styles/App.css
│   │   ├── App.js
│   │   └── index.js
│   ├── .env.example
│   └── package.json
├── .gitignore
└── README.md
```

## How Biometric Enrollment Works (FR-1.2 / FR-1.3)

1. The browser loads `face-api.js` models (tiny face detector, 68-point
   landmark model, recognition model) from `frontend/public/models`.
2. While the webcam is live, `checkEnvironment()` runs every 400ms to
   verify: a face is present, lighting is sufficient (brightness sample),
   and the face is centered/sized correctly in frame.
3. Only once all three checks pass can the user click **Capture Baseline
   Face**, which runs `extractFaceEmbedding()` to produce a 128-d vector.
4. That vector — **never the image itself** — is sent to the backend and
   stored on the `User` document. The server re-validates the reported
   environment metrics before accepting the registration (defense in
   depth against a tampered client).

## Setup

### 1. Download face-api.js model weights
Place these files in `frontend/public/models/` (see the README there):
`tiny_face_detector_model-*`, `face_landmark_68_model-*`, `face_recognition_model-*`
from https://github.com/justadudewhohacks/face-api.js/tree/master/weights

### 2. Backend
```bash
cd backend
cp .env.example .env      # then edit MONGO_URI / JWT_SECRET
npm install
npm run dev                # requires MongoDB running locally or via URI
```

### 3. Frontend
```bash
cd frontend
cp .env.example .env
npm install
npm start
```

The app opens at `http://localhost:3000` and talks to the API at
`http://localhost:5000/api` (configurable via `REACT_APP_API_URL`).

## API

### `POST /api/auth/register`
```json
{
  "username": "jdoe",
  "email": "jdoe@example.com",
  "password": "Str0ng!Pass",
  "faceEmbedding": [0.123, -0.045, ...],
  "environmentCheck": { "facePresent": true, "lightingScore": 82, "framePositioned": true }
}
```

### `POST /api/auth/login`
```json
{ "emailOrUsername": "jdoe", "password": "Str0ng!Pass", "faceEmbedding": [...] }
```
`faceEmbedding` is optional here and, if supplied, is used as a second
biometric factor compared against the stored baseline using Euclidean
distance (`FACE_MATCH_THRESHOLD` in `.env`).

## Security Notes
- Passwords are hashed with bcrypt (12 salt rounds) — never stored in plaintext.
- Only facial **embedding vectors** are persisted; raw webcam frames never
  leave the browser.
- Environmental validation is enforced both client-side (real-time UX) and
  server-side (integrity check).
