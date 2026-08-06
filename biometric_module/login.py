import cv2
import face_recognition
import numpy as np
import time
import json
from liveness import check_liveness
from encryption_utils import decrypt_data
import time as timing_module
import time as timing_module

# ---------- Facial Authentication ----------

def login_face():
    try:
        with open("enrolled_face.enc", "rb") as f:
            encrypted = f.read()
        decrypted_str = decrypt_data(encrypted)
        enrolled_embedding = np.array(json.loads(decrypted_str))
    except FileNotFoundError:
         return False, "No enrolled face found — please register first"


    live, live_message = check_liveness()
    if not live:
        return False, live_message
    start_time = timing_module.time()

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
    end_time = timing_module.time()
    latency = end_time - start_time
    print(f"Facial authentication latency: {latency:.3f} seconds (benchmark: ≤1.2s)")

    print("Distance:", distance)

    if distance < threshold:
        return True, "Login successful — face matched and liveness confirmed!"
    else:
        return False, "Login failed — face does not match"
# ---------- Credential Authentication ----------
def login_credentials():
    username = input("Username: ")
    password = input("Password: ")

    try:
        with open(f"user_{username}.json", "r") as f:
            user_record = json.load(f)
    except FileNotFoundError:
        return False, "User not found"

    if user_record.get("password") == password:
        return True, "Login successful — credentials matched!"
    else:
        return False, "Login failed — incorrect password"

# ---------- FR-2.1 & FR-2.3: Login Menu with Automatic Fallback ----------
def login_menu():
    print("=== Login ===")
    print("1. Facial Authentication")
    print("2. Credential Authentication")
    choice = input("Choose login method (1 or 2): ")

    if choice == "1":
        success, message = login_face()
        print(message)

        if not success:
            print("\nFacial authentication unsuccessful. Falling back to password login...")
            success, message = login_credentials()
            print(message)

    elif choice == "2":
        success, message = login_credentials()
        print(message)

    else:
        print("Invalid choice")
        return False

    return success

if __name__ == "__main__":
    login_menu()