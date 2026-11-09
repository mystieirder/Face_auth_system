require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const connectDB = require('./config/db');
const authRoutes = require('./routes/authRoutes');
const paymentRoutes = require('./routes/paymentRoutes');
const profileRoutes = require('./routes/profileRoutes');

const app = express();

// Only your frontend may call this API (CLIENT_URL is set on Render later)
const allowedOrigins = [process.env.CLIENT_URL, 'http://localhost:3000'].filter(Boolean);
app.use(cors({ origin: allowedOrigins }));
app.use(express.json({ limit: '5mb' })); // face embeddings + base64 photos can be large

// ------------------------------------------------------------
// Routes
// ------------------------------------------------------------
app.use('/api/auth', authRoutes);
app.use('/api/payment', paymentRoutes);
app.use('/api/profile', profileRoutes);
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// ------------------------------------------------------------
// Startup: connect to DB, then start listening
// ------------------------------------------------------------
const PORT = process.env.PORT || 5000;

connectDB().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
  });
});