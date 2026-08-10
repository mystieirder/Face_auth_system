const connectDB = require('../config/db');
const { getPool, sql } = connectDB;

async function findByEmailOrUsername(emailOrUsername) {
  const pool = getPool();
  const result = await pool
    .request()
    .input('value', sql.NVarChar, emailOrUsername)
    .query('SELECT * FROM Users WHERE Email = @value OR Username = @value');
  return result.recordset[0] || null;
}

async function createUser({ username, email, passwordHash, faceEmbedding, environmentCheck }) {
  const pool = getPool();
  const result = await pool
    .request()
    .input('username', sql.NVarChar, username)
    .input('email', sql.NVarChar, email)
    .input('passwordHash', sql.NVarChar, passwordHash)
    .input('faceEmbedding', sql.NVarChar, JSON.stringify(faceEmbedding))
    .input('facePresent', sql.Bit, environmentCheck.facePresent)
    .input('lightingScore', sql.Float, environmentCheck.lightingScore)
    .input('framePositioned', sql.Bit, environmentCheck.framePositioned)
    .query(`
      INSERT INTO Users
        (Username, Email, PasswordHash, FaceEmbedding, FacePresent, LightingScore, FramePositioned, ValidatedAt)
      OUTPUT INSERTED.*
      VALUES
        (@username, @email, @passwordHash, @faceEmbedding, @facePresent, @lightingScore, @framePositioned, GETDATE())
    `);
  return result.recordset[0];
}

module.exports = { findByEmailOrUsername, createUser };