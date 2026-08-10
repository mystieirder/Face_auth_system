require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const connectDB = require('./config/db');
const authRoutes = require('./routes/authRoutes');
const paymentRoutes = require('./routes/paymentRoutes');
const profileRoutes = require('./routes/profileRoutes');

const app = express();

app.use(cors());
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
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
});