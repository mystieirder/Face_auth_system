"""Bridges the existing hand-gesture-recognition system to the web app.

This file is NEW and does not modify gesture.py or main.py in any way —
it imports gesture.py exactly as-is and reuses its HandGestureProcessor,
GESTURE_ACTIONS, and dwell-confirmed GestureResult objects untouched.

What it does:
    - Opens the webcam and runs the same recognition loop as main.py,
      but with NO desktop window — the annotated frame is JPEG-encoded
      and streamed to the browser over the WebSocket instead of being
      shown via cv2.imshow. This is meant to run headless in a terminal
      while the browser (ProfilePage's Hand Gesture box) displays the
      feed and reacts to commands.
    - Starts a local WebSocket server on ws://localhost:8765.
    - Every time gesture.py reports `action_triggered=True` (i.e. a
      gesture has completed its dwell period and is confirmed), this
      broadcasts a small JSON message describing it to any connected
      browser tab.
    - Every frame where INDEX POINTING is active, it also streams the
      live cursor_position (normalized 0..1) so the browser can draw a
      matching on-screen pointer.

Run this INSTEAD of main.py when you want the browser bridge active:
    python web_bridge.py
(main.py itself is untouched and still works exactly as before, with
its own desktop window, if you run it directly.)

Press Ctrl+C in the terminal to stop (there is no window to close).

Requires the `websockets` package in addition to what's already in
requirements.txt:
    pip install websockets
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import time

import cv2

import gesture as gu  # the existing, unmodified module

FRAME_WIDTH: int = 960
FRAME_HEIGHT: int = 720
WS_HOST: str = "localhost"
WS_PORT: int = 8765

# The annotated frame is downscaled before sending to the browser to keep
# each WebSocket message small — the full-res frame is still used for
# gesture detection, this only affects what's streamed for display.
STREAM_WIDTH: int = 480
STREAM_HEIGHT: int = 360
STREAM_JPEG_QUALITY: int = 60

# Maps gesture.py's own GESTURE_ACTIONS vocabulary to short command
# strings the browser side understands. Left-hand column is exactly
# what gesture.py already reports; nothing here changes gesture.py.
GESTURE_TO_COMMAND: dict[str, str] = {
    gu.GESTURE_THUMBS_UP: "confirm",       # Confirm Selection / Save Field
    gu.GESTURE_OPEN_PALM: "next_field",    # Move to Next Field
    gu.GESTURE_SIDE_SWEEP: "prev_field",   # Previous Field / Cancel Action
    gu.GESTURE_PINCH: "select",            # Select Checkbox / Radio / Dropdown
}

_clients: set = set()
_loop: asyncio.AbstractEventLoop | None = None


async def _register(websocket) -> None:
    _clients.add(websocket)
    try:
        async for _ in websocket:
            pass  # this bridge only sends; it ignores anything the page sends back
    finally:
        _clients.discard(websocket)


async def _broadcast(message: dict) -> None:
    if not _clients:
        return
    data = json.dumps(message)
    dead = []
    for ws in _clients:
        try:
            await ws.send(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def _push(message: dict) -> None:
    """Thread-safe hand-off from the OpenCV loop (sync) to the WS server (async)."""
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(message), _loop)


def _run_ws_server() -> None:
    global _loop
    import websockets

    async def _main() -> None:
        async with websockets.serve(_register, WS_HOST, WS_PORT):
            print(f"[web_bridge] WebSocket server listening on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()  # run forever

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(_main())


def main() -> None:
    threading.Thread(target=_run_ws_server, daemon=True).start()
    time.sleep(0.3)  # give the WS server a moment to bind before the first frame

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Unable to access the webcam.", file=sys.stderr)
        sys.exit(1)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    processor = gu.HandGestureProcessor()
    previous_time = time.time()
    print("[web_bridge] Running headless — no desktop window. Press Ctrl+C here to stop.")

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
                # Stream live cursor position (normalized) every frame while pointing.
                nx = result.cursor_position[0] / FRAME_WIDTH
                ny = result.cursor_position[1] / FRAME_HEIGHT
                _push({"type": "cursor", "x": nx, "y": ny})

            gu.draw_info_panel(frame, fps, result)

            # Stream the annotated frame to the browser instead of showing a
            # desktop window. Downscale first to keep messages small.
            stream_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
            encode_ok, buffer = cv2.imencode(
                ".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
            )
            if encode_ok:
                b64_frame = base64.b64encode(buffer).decode("ascii")
                _push({"type": "frame", "data": b64_frame})

            if result.action_triggered:
                command = GESTURE_TO_COMMAND.get(result.gesture)
                if command:
                    _push({
                        "type": "action",
                        "gesture": result.gesture,
                        "action": result.action,
                        "command": command,
                    })
    except KeyboardInterrupt:
        print("\n[web_bridge] Stopping (Ctrl+C received).")
    finally:
        processor.close()
        capture.release()


if __name__ == "__main__":
    main()