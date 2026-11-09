const connectDB = require('../config/db');
const { getPool } = connectDB;

const USER_COLUMNS = `
  id AS "Id",
  username AS "Username",
  email AS "Email",
  password_hash AS "PasswordHash",
  face_embedding AS "FaceEmbedding",
  face_present AS "FacePresent",
  lighting_score AS "LightingScore",
  frame_positioned AS "FramePositioned",
  validated_at AS "ValidatedAt",
  has_paid AS "HasPaid"
`;

async function findByEmailOrUsername(emailOrUsername) {
  const pool = getPool();
  const result = await pool.query(
    `SELECT ${USER_COLUMNS} FROM users WHERE email = $1 OR username = $1`,
    [emailOrUsername]
  );
  return result.rows[0] || null;
}

async function createUser({ username, email, passwordHash, faceEmbedding, environmentCheck }) {
  const pool = getPool();
  const result = await pool.query(
    `INSERT INTO users
       (username, email, password_hash, face_embedding, face_present, lighting_score, frame_positioned, validated_at)
     VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
     RETURNING ${USER_COLUMNS}`,
    [
      username,
      email,
      passwordHash,
      JSON.stringify(faceEmbedding),
      environmentCheck.facePresent,
      environmentCheck.lightingScore,
      environmentCheck.framePositioned,
    ]
  );
  return result.rows[0];
}

module.exports = { findByEmailOrUsername, createUser };