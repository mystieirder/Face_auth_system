const express = require('express');
const multer = require('multer');
const router = express.Router();
const requireAuth = require('../middleware/auth');
const { getProfile, saveProfile, uploadPhoto } = require('../controllers/profileController');

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 2 * 1024 * 1024 } });

router.get('/', requireAuth, getProfile);
router.post('/', requireAuth, saveProfile);
router.post('/photo', requireAuth, upload.single('photo'), uploadPhoto);

module.exports = router;