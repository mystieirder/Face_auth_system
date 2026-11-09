const { Pool } = require('pg');

let pool;

async function connectDB() {
  try {
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      ssl: { rejectUnauthorized: false },
    });
    await pool.query('SELECT 1');
    console.log('Connected to PostgreSQL');
    return pool;
  } catch (err) {
    console.error('Database Connection Failed! Bad Config: ', err);
    process.exit(1);
  }
}

function getPool() {
  if (!pool) {
    throw new Error('Database not connected yet. Call connectDB() first.');
  }
  return pool;
}

module.exports = connectDB;
module.exports.getPool = getPool;