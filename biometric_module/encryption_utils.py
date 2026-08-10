from cryptography.fernet import Fernet
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(BASE_DIR, "secret.key")

def load_or_create_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key

def encrypt_data(data_str):
    key = load_or_create_key()
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data_str.encode())
    return encrypted

def decrypt_data(encrypted_bytes):
    key = load_or_create_key()
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_bytes)
    return decrypted.decode()