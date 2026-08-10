import os
import numpy as np
import json
from encryption_utils import decrypt_data
from liveness import check_liveness
import cv2
import face_recognition
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENROLLED_FACE_PATH = os.path.join(BASE_DIR, "enrolled_face.enc")

def attempt_face_login():
    """
    Returns (success: bool, message: str)
    Designed to be called from a Flask route.
    """
    try:
        with open(ENROLLED_FACE_PATH, "rb") as f:
            encrypted = f.read()
        decrypted_str = decrypt_data(encrypted)
        enrolled_embedding = np.array(json.loads(decrypted_str))
    except FileNotFoundError:
        return False, "No enrolled face found — please register first"

    live, live_message = check_liveness()
    if not live:
        return False, live_message

    video = cv2.VideoCapture(0)
    for _ in range(15):
        ret, frame = video.read()
        time.sleep(0.1)
    video.release()

    if not ret:
        return False, "Webcam not detected"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    if len(face_locations) == 0:
        return False, "No face detected — please reposition and try again"

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    live_embedding = face_encodings[0]

    distance = np.linalg.norm(enrolled_embedding - live_embedding)
    threshold = 0.6
    
    print(f"[DEBUG] Distance: {distance}")


    if distance < threshold:
        return True, "Face matched and liveness confirmed!"
    else:
        return False, "Face does not match"