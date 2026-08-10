function validatePassword(password) {
  if (!password || password.length < 8) {
    return { valid: false, message: 'Password must be at least 8 characters long.' };
  }
  if (!/[a-zA-Z]/.test(password)) {
    return { valid: false, message: 'Password must contain at least one letter.' };
  }
  if (!/\d/.test(password)) {
    return { valid: false, message: 'Password must contain at least one number.' };
  }
  if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>/?]/.test(password)) {
    return { valid: false, message: 'Password must contain at least one special character.' };
  }
  return { valid: true, message: 'Password meets all policy requirements.' };
}

function validatePasswordPolicy(req, res, next) {
  const { password } = req.body;
  const result = validatePassword(password);

  if (!result.valid) {
    return res.status(422).json({ success: false, message: result.message });
  }

  next();
}

module.exports = { validatePasswordPolicy, validatePassword };