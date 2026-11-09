DROP TABLE IF EXISTS payments, profiles, users CASCADE;

CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(100) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  face_embedding TEXT,
  face_present BOOLEAN,
  lighting_score DOUBLE PRECISION,
  frame_positioned BOOLEAN,
  validated_at TIMESTAMPTZ,
  has_paid BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE profiles (
  id SERIAL PRIMARY KEY,
  user_id INT UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  full_name VARCHAR(255),
  contact_phone VARCHAR(50),
  email_address VARCHAR(255),
  date_of_birth DATE,
  physical_address TEXT,
  accessibility_mode VARCHAR(50),
  photo_path TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE payments (
  id SERIAL PRIMARY KEY,
  user_id INT REFERENCES users(id) ON DELETE SET NULL,
  username VARCHAR(100),
  plan_name VARCHAR(100),
  razorpay_order_id VARCHAR(100) UNIQUE NOT NULL,
  razorpay_payment_id VARCHAR(100),
  amount DOUBLE PRECISION,
  status VARCHAR(30) NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT NOW()
);