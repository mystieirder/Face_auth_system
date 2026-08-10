"""Entry point for the real-time Hand Gesture Recognition System.

This module is responsible only for webcam initialization, frame
capture, invoking the gesture-recognition pipeline defined in
``gesture_utils``, drawing the resulting UI elements, and handling
keyboard-based exit. All gesture-recognition and drawing logic lives
in ``gesture_utils``.
"""

from __future__ import annotations

import sys
import time

import cv2

import gesture as gu

ESC_KEY: int = 27
WINDOW_NAME: str = "Hand Gesture Recognition System"
FRAME_WIDTH: int = 960
FRAME_HEIGHT: int = 720


def main() -> None:
    """Run the real-time hand gesture recognition application.

    Opens the default webcam, continuously reads frames, feeds each
    frame through the gesture-recognition pipeline, draws the
    resulting UI, and exits cleanly when the user presses ESC.
    """
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Unable to access the webcam.", file=sys.stderr)
        sys.exit(1)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    processor = gu.HandGestureProcessor()
    previous_time = time.time()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Warning: Failed to read frame from webcam.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            result = processor.process(frame)

            current_time = time.time()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            gu.draw_hand_landmarks(frame, result.hand_landmarks)

            if result.bounding_box is not None:
                status_color = gu.get_status_color(result.confirmed, result.gesture)
                gu.draw_bounding_box(frame, result.bounding_box, status_color)
                gu.draw_gesture_label(frame, result.bounding_box, result.gesture, status_color)

            if result.cursor_position is not None:
                gu.draw_cursor(frame, result.cursor_position)

            gu.draw_info_panel(frame, fps, result)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY:
                break
    finally:
        processor.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()