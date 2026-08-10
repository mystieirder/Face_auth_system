import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

/**
 * FR-1.1 + FR-1.2 + FR-1.3: sends standard credentials plus the facial
 * embedding vector (never a raw image) and environmental check results.
 */
export async function registerUser({ username, email, password, faceEmbedding, environmentCheck }) {
  const response = await axios.post(`${API_BASE_URL}/auth/register`, {
    username,
    email,
    password,
    faceEmbedding,
    environmentCheck,
  });
  return response.data;
}

/**
 * FR-2.1 + FR-2.2: pass EITHER `password` (credential login) OR
 * `faceEmbedding` + `livenessVerified` (facial login) — not both.
 */
export async function loginUser({ emailOrUsername, password, faceEmbedding, livenessVerified }) {
  const response = await axios.post(`${API_BASE_URL}/auth/login`, {
    emailOrUsername,
    password,
    faceEmbedding,
    livenessVerified,
  });
  return response.data;
}