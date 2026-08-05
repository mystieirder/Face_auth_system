import cv2
import face_recognition
import numpy as np
import time
import json

def capture_face_embedding():
    video = cv2.VideoCapture(0)
    for _ in range(15):
        ret, frame = video.read()
        time.sleep(0.1)
    video.release()

    if not ret:
        return None, "Webcam not detected"

    # Brightness check (basic environmental validation)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < 50:
        return None, "Lighting too low — please improve lighting and try again"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if len(face_locations) == 0:
        return None, "No face detected — please reposition and try again"
    elif len(face_locations) > 1:
        return None, "Multiple faces detected — please ensure only one person is in frame"

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    embedding = face_encodings[0]
    return embedding, "Success"

if __name__ == "__main__":
    embedding, message = capture_face_embedding()
    if embedding is not None:
        print(message)
        print("Embedding vector length:", len(embedding))

        # Save embedding to a JSON file (acts as our "enrolled" template)
        with open("enrolled_face.json", "w") as f:
            json.dump(embedding.tolist(), f)
        print("Embedding saved to enrolled_face.json")
    else:
        print(message)