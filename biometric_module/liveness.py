import cv2
import face_recognition
import numpy as np
import time
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye_points):
    # eye_points = 6 (x, y) coordinates around one eye
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def check_liveness(duration=8, ear_threshold=0.25):
    print("Liveness check: please blink naturally...")
    video = cv2.VideoCapture(0)

    blink_detected = False
    eyes_were_closed = False
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = video.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        if len(face_landmarks_list) == 0:
            continue

        landmarks = face_landmarks_list[0]
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

       
        print(f"EAR: {avg_ear:.3f}")  # ADD THIS LINE

        if avg_ear < ear_threshold:
            eyes_were_closed = True
        elif eyes_were_closed and avg_ear >= ear_threshold:
            blink_detected = True
            break

    video.release()

    if blink_detected:
        return True, "Liveness confirmed — blink detected"
    else:
        return False, "Liveness check failed — no blink detected, possible spoofing attempt"

if __name__ == "__main__":
    success, message = check_liveness()
    print(message)