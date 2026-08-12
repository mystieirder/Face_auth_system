const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { findByEmailOrUsername, createUser } = require('../models/User');
const { validateEnvironment, evaluateFaceMatch } = require('../utils/faceUtils');
const connectDB = require('../config/db');
const { getPool, sql } = connectDB;

const SALT_ROUNDS = 12;

async function register(req, res) {
  try {
    const { username, email, password, faceEmbedding, environmentCheck, paymentOrderId } = req.body;

    if (!paymentOrderId) {
      return res.status(402).json({ success: false, message: 'Payment is required before registration' });
    }

    const pool = getPool();
    const paymentCheck = await pool
      .request()
      .input('orderId', sql.NVarChar, paymentOrderId)
      .query("SELECT * FROM Payments WHERE RazorpayOrderId = @orderId AND Status = 'paid' AND UserId IS NULL");

    if (paymentCheck.recordset.length === 0) {
      return res.status(402).json({ success: false, message: 'No valid payment found for this order' });
    }

    if (!username || !email) {
      return res.status(400).json({ success: false, message: 'Username and email are required' });
    }

    if (!Array.isArray(faceEmbedding) || faceEmbedding.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Facial embedding vector is required (raw images are not accepted)',
      });
    }

    const minLighting = Number(process.env.MIN_LIGHTING_SCORE) || 60;
    const envResult = validateEnvironment(environmentCheck, minLighting);
    if (!envResult.valid) {
      return res.status(422).json({
        success: false,
        message: 'Environmental validation failed',
        errors: envResult.errors,
      });
    }

    const existingUser = await findByEmailOrUsername(email);
    if (existingUser) {
      return res.status(409).json({ success: false, message: 'Username or email already registered' });
    }

    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
    const user = await createUser({ username, email, passwordHash, faceEmbedding, environmentCheck });

    await pool
      .request()
      .input('userId', sql.Int, user.Id)
      .input('orderId', sql.NVarChar, paymentOrderId)
      .query('UPDATE Payments SET UserId = @userId WHERE RazorpayOrderId = @orderId');

    await pool
      .request()
      .input('userId', sql.Int, user.Id)
      .query('UPDATE Users SET HasPaid = 1 WHERE Id = @userId');

    return res.status(201).json({
      success: true,
      message: 'Registration successful',
      user: { id: user.Id, username: user.Username, email: user.Email },
    });
  } catch (err) {
    console.error('Register error:', err);
    return res.status(500).json({ success: false, message: 'Server error during registration' });
  }
}

/**
 * FR-2.1: Dual Authentication Paths — the client selects EITHER facial
 * authentication OR credential authentication; this endpoint accepts
 * either shape:
 *   - Credential login:  { emailOrUsername, password }
 *   - Facial login:      { emailOrUsername, faceEmbedding, livenessVerified }
 * FR-2.2: Facial logins MUST include a client-side-confirmed liveness
 * check (livenessVerified === true) — this endpoint refuses to accept
 * a raw embedding without it, since that is exactly what a static
 * photo/screen spoofing attack would send.
 * FR-2.3: Automatic Fallback is a client-side concern (see LoginPage) —
 * this endpoint simply supports both paths so the client can fall back
 * without a separate round trip or losing session context.
 */
async function login(req, res) {
  try {
    const { emailOrUsername, password, faceEmbedding, livenessVerified } = req.body;

    if (!emailOrUsername) {
      return res.status(400).json({ success: false, message: 'Username or email is required' });
    }

    const user = await findByEmailOrUsername(emailOrUsername);
    if (!user) {
      return res.status(401).json({ success: false, message: 'Invalid credentials' });
    }

    const usingFaceLogin = Array.isArray(faceEmbedding) && faceEmbedding.length > 0;
    const faceEmbeddingBurst = Array.isArray(faceEmbedding) && faceEmbedding[0] && Array.isArray(faceEmbedding[0]) ? faceEmbedding : null;

    if (usingFaceLogin) {
      if (!livenessVerified) {
        return res.status(401).json({
          success: false,
          message: 'Liveness check did not complete — please try again (blink or move slightly)',
        });
      }

      const threshold = Number(process.env.FACE_MATCH_THRESHOLD) || 0.5;
      const storedEmbedding = JSON.parse(user.FaceEmbedding);
      const burstToCompare = faceEmbeddingBurst || [faceEmbedding];
      const matchResult = evaluateFaceMatch(storedEmbedding, burstToCompare, threshold);
      if (!matchResult.matched) {
        return res.status(401).json({ success: false, message: 'Face did not match — please try again or use your password' });
      }
    } else {
      if (!password) {
        return res.status(400).json({ success: false, message: 'Password is required' });
      }
      const passwordMatches = await bcrypt.compare(password, user.PasswordHash);
      if (!passwordMatches) {
        return res.status(401).json({ success: false, message: 'Invalid credentials' });
      }
    }

    if (!user.HasPaid) {
      return res.status(402).json({
        success: false,
        message: 'Payment required before login',
        userId: user.Id,
      });
    }

    const token = jwt.sign(
      { userId: user.Id, username: user.Username },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
    );

    return res.status(200).json({
      success: true,
      message: 'Login successful',
      authMethod: usingFaceLogin ? 'face' : 'password',
      token,
      user: { id: user.Id, username: user.Username, email: user.Email },
    });
  } catch (err) {
    console.error('Login error:', err);
    return res.status(500).json({ success: false, message: 'Server error during login' });
  }
}

module.exports = { register, login };