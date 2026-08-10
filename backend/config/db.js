const sql = require('mssql');

const config = {
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  server: process.env.DB_SERVER,
  database: process.env.DB_DATABASE,
  port: parseInt(process.env.DB_PORT),
  options: {
    encrypt: false,
    trustServerCertificate: true,
  },
};

let pool;

async function connectDB() {
  try {
    pool = await new sql.ConnectionPool(config).connect();
    console.log('Connected to SQL Server');
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
module.exports.sql = sql;