# Hand Gesture Recognition System

A real-time, webcam-based hand gesture recognition application built with
OpenCV and MediaPipe. The system tracks a single hand, classifies five
predefined gestures, drives a smoothed virtual cursor, and confirms every
gesture with a dwell-time mechanism before triggering its mapped action.

## Project Overview

The application captures live webcam video, detects a single hand using
MediaPipe Hands, extracts all 21 hand landmarks, and classifies the hand
pose into one of five supported gestures. Each gesture is held for a fixed
dwell period before it is confirmed and its mapped action is triggered,
preventing accidental activations from momentary or transitional poses.
A virtual, exponentially-smoothed cursor follows the index fingertip while
pointing, and a professional on-screen information panel reports FPS,
gesture, mapped action, confirmation status, dwell progress, and cursor
position in real time.

## Features

- Real-time single-hand tracking with all 21 MediaPipe landmarks drawn
- Live FPS counter and hand bounding box
- Five predefined gestures with clear, non-overlapping classification rules
- Dwell-time confirmation (1.5s) to prevent accidental triggers
- Each mapped action fires exactly once per gesture hold
- Smooth, jitter-free virtual cursor using exponential smoothing
- Horizontal side-sweep detection built on top of the open-palm pose
- Color-coded status UI (green = confirmed, yellow = detecting, red = none)
- Clean two-module architecture with full type hints and docstrings

## Folder Structure

```
hand-gesture-recognition/
├── main.py             # Webcam loop, frame capture, UI orchestration
├── gesture_utils.py     # MediaPipe logic, gesture classification, drawing
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Installation

1. Ensure Python 3.12+ is installed.
2. (Recommended) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.12+
- opencv-python
- mediapipe
- numpy
- A connected webcam

## Usage

Run the application from the project directory:

```
python main.py
```

- Present one hand to the webcam to begin tracking.
- Hold a supported gesture steady for 1.5 seconds to confirm it.
- Press **ESC** at any time to exit the application.

## Supported Gestures

| Gesture               | Conditions                                              | Mapped Action                              |
|------------------------|----------------------------------------------------------|---------------------------------------------|
| Thumbs Up              | Thumb extended upward; other fingers folded              | Confirm Selection / Save Field               |
| Open Palm Up            | All five fingers extended, palm facing the camera        | Move to Next Field                           |
| Open Palm Side Sweep    | Open palm moved horizontally past a displacement threshold | Previous Field / Cancel Action              |
| Index Pointing          | Only the index finger extended                           | Virtual Cursor Positioning                   |
| Pinch                    | Thumb tip touching index fingertip                        | Select Checkbox / Radio Button / Dropdown Item |

Every gesture must remain stable for **1.5 seconds** before it is confirmed
(status changes from `Detecting...` to `Confirmed`), and its mapped action
fires exactly once until the gesture changes.

## Future Improvements

- Two-hand support with independent gesture streams
- Configurable dwell time and gesture thresholds via a settings file
- OS-level integration for real cursor control and keyboard/mouse events
- Gesture customization and calibration UI for different hand sizes
- On-screen gesture history log and session analytics