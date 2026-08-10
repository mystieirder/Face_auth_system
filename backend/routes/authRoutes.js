const express = require('express');
const { body } = require('express-validator');
const { register, login } = require('../controllers/authController');
const { validatePasswordPolicy } = require('../middleware/validatePassword');

const router = express.Router();

router.post(
  '/register',
  [
    body('username').trim().isLength({ min: 3 }).withMessage('Username must be at least 3 characters'),
    body('email').isEmail().withMessage('A valid email is required').normalizeEmail(),
  ],
  validatePasswordPolicy,
  register
);

router.post('/login', login);

module.exports = router;