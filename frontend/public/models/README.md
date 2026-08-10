# face-api.js Models

Download the following model weight files from the face-api.js weights
repository and place them in this folder before running the app:

- tiny_face_detector_model-*
- face_landmark_68_model-*
- face_recognition_model-*

Source: https://github.com/justadudewhohacks/face-api.js/tree/master/weights

These models run entirely in the browser (client-side), which is what
allows the app to extract a facial embedding vector locally and send
only that vector to the server — never a raw image (see FR-1.2).