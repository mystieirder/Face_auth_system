const connectDB = require('../config/db');
const { getPool, sql } = connectDB;
const path = require('path');
const fs = require('fs');

const UPLOAD_DIR = path.join(__dirname, '..', 'uploads', 'profile_photos');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

async function getProfile(req, res) {
  try {
    const pool = getPool();
    const result = await pool
      .request()
      .input('userId', sql.Int, req.userId)
      .query('SELECT * FROM Profiles WHERE UserId = @userId');

    res.json({ success: true, profile: result.recordset[0] || null });
  } catch (err) {
    console.error('Get profile error:', err);
    res.status(500).json({ success: false, message: 'Could not load profile' });
  }
}

async function saveProfile(req, res) {
  try {
    const { fullName, contactPhone, emailAddress, dateOfBirth, physicalAddress, accessibilityMode } = req.body;
    const pool = getPool();

    const existing = await pool
      .request()
      .input('userId', sql.Int, req.userId)
      .query('SELECT Id FROM Profiles WHERE UserId = @userId');

    if (existing.recordset.length > 0) {
      await pool
        .request()
        .input('userId', sql.Int, req.userId)
        .input('fullName', sql.NVarChar, fullName)
        .input('contactPhone', sql.NVarChar, contactPhone)
        .input('emailAddress', sql.NVarChar, emailAddress)
        .input('dob', sql.Date, dateOfBirth || null)
        .input('address', sql.NVarChar, physicalAddress)
        .input('accessibilityMode', sql.NVarChar, accessibilityMode)
        .query(`UPDATE Profiles SET FullName=@fullName, ContactPhone=@contactPhone,
                EmailAddress=@emailAddress, DateOfBirth=@dob, PhysicalAddress=@address,
                AccessibilityMode=@accessibilityMode, UpdatedAt=GETDATE() WHERE UserId=@userId`);
    } else {
      await pool
        .request()
        .input('userId', sql.Int, req.userId)
        .input('fullName', sql.NVarChar, fullName)
        .input('contactPhone', sql.NVarChar, contactPhone)
        .input('emailAddress', sql.NVarChar, emailAddress)
        .input('dob', sql.Date, dateOfBirth || null)
        .input('address', sql.NVarChar, physicalAddress)
        .input('accessibilityMode', sql.NVarChar, accessibilityMode)
        .query(`INSERT INTO Profiles (UserId, FullName, ContactPhone, EmailAddress, DateOfBirth, PhysicalAddress, AccessibilityMode)
                VALUES (@userId, @fullName, @contactPhone, @emailAddress, @dob, @address, @accessibilityMode)`);
    }

    res.json({ success: true, message: 'Profile saved' });
  } catch (err) {
    console.error('Save profile error:', err);
    res.status(500).json({ success: false, message: 'Could not save profile' });
  }
}

async function uploadPhoto(req, res) {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No photo included' });
    }
    const filename = `user_${req.userId}.webp`;
    const filepath = path.join(UPLOAD_DIR, filename);
    fs.writeFileSync(filepath, req.file.buffer);

    const pool = getPool();
    await pool
      .request()
      .input('userId', sql.Int, req.userId)
      .input('photoPath', sql.NVarChar, `/uploads/profile_photos/${filename}`)
      .query(`UPDATE Profiles SET PhotoPath=@photoPath WHERE UserId=@userId`);

    res.json({ success: true, photoUrl: `/uploads/profile_photos/${filename}` });
  } catch (err) {
    console.error('Upload photo error:', err);
    res.status(500).json({ success: false, message: 'Upload failed' });
  }
}

module.exports = { getProfile, saveProfile, uploadPhoto };