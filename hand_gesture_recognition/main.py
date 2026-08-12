"""GestureAccess - hand gesture communication interface.

This module contains ONLY the hand-gesture communication UI.

It intentionally does NOT contain:
- Profile Entry
- Full Legal Name / Email / Phone fields
- ProfileForm
- Login / registration logic
- Field numbering or pending/completed status

The virtual keyboard is an input device.  The existing application/form that
owns the currently focused field remains responsible for storing the text.
"""
from __future__ import annotations

import time

import cv2
import mediapipe as mp

import gesture as gu
import fingerspelling as fs
from config import *
from keyboard import VirtualKeyboard

MODE_MENU = "MENU"
MODE_KEYBOARD = "KEYBOARD"
MODE_LETTERS = "LETTERS"
CHEAT_SHEET_KEY = ord("h")


def _map_active_region(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    """Remap a point from the smaller, comfortable 'active region' of
    the frame (see CURSOR_ACTIVE_MARGIN_LEFT/RIGHT/TOP/BOTTOM in
    config.py) onto the full frame coordinate range that the keyboard
    layout actually uses.

    Without this, a straight 1:1 mapping means reaching an edge key
    (leftmost/rightmost column, top row, bottom control row) requires
    pointing at the literal boundary of the webcam frame - right where
    MediaPipe hand tracking is least reliable, and further than a
    pointing hand can comfortably reach in some directions (crossing
    toward the left, or dropping the arm low). Margins are independent
    per edge rather than a single centered region, since that reach is
    not symmetric.
    """
    left = CURSOR_ACTIVE_MARGIN_LEFT * w
    right = CURSOR_ACTIVE_MARGIN_RIGHT * w
    top = CURSOR_ACTIVE_MARGIN_TOP * h
    bottom = CURSOR_ACTIVE_MARGIN_BOTTOM * h
    active_w = max(w - left - right, 1.0)
    active_h = max(h - top - bottom, 1.0)
    mapped_x = (x - left) / active_w * w
    mapped_y = (y - top) / active_h * h
    return mapped_x, mapped_y


def _raw_cursor(frame, hand_landmarks, gesture, smoother):
    """Track the index fingertip only while an actual pointing/pinch
    gesture is active. Previously this fed the smoother on every frame
    a hand was visible at all - including frames mid-transition to a
    different gesture (thumbs up, open palm, etc). Since the smoother
    is an exponential moving average, those irrelevant positions kept
    dragging the smoothed point around, so when the user pointed again
    the cursor had to "catch up" from wherever it had drifted to -
    felt like input lag and made it land on the wrong key. Resetting
    whenever we're not actively tracking means every new point gesture
    starts fresh from the real fingertip position instead of an average
    contaminated by the last few frames of a different hand shape.
    """
    if hand_landmarks is None or gesture not in (gu.GESTURE_INDEX_POINT, gu.GESTURE_PINCH):
        smoother.reset()
        return None
    points = gu._landmarks_to_array(hand_landmarks)
    h, w = frame.shape[:2]
    raw_x = points[gu.INDEX_TIP][0] * w
    raw_y = points[gu.INDEX_TIP][1] * h
    x, y = smoother.update(raw_x, raw_y)
    # Stretch the comfortable active region back out to the full frame
    # so every key, including ones flush against an edge, is reachable
    # (see _map_active_region docstring above).
    x, y = _map_active_region(x, y, w, h)
    # _map_active_region does floating-point arithmetic, but every
    # downstream consumer (cv2 drawing calls, the keyboard's key
    # hit-testing) expects integer pixel coordinates - cv2.circle in
    # particular raises "wrong type" if handed a float here.
    x, y = int(round(x)), int(round(y))
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def _draw_top_bar(frame, mode: str, fps: float) -> None:
    """Draw only the application header; never draw profile information."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, TOP_BAR_HEIGHT), PANEL, -1)
    cv2.line(frame, (0, TOP_BAR_HEIGHT), (w, TOP_BAR_HEIGHT), BORDER, 1)

    cv2.putText(frame, "GESTURE ACCESS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "Non-Verbal Communication", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, MUTED, 1, cv2.LINE_AA)

    fps_label = f"FPS {fps:.0f}"
    (fps_w, _), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    fps_x = w - 15 - fps_w
    cv2.putText(frame, fps_label, (fps_x, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, MUTED, 1, cv2.LINE_AA)

    (mode_w, _), _ = cv2.getTextSize(mode, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    mode_x = fps_x - 20 - mode_w
    cv2.putText(frame, mode, (mode_x, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, ACCENT, 2, cv2.LINE_AA)


def _draw_hand(frame, landmarks):
    if landmarks is None:
        return
    spec = mp.solutions.drawing_utils.DrawingSpec(
        color=WHITE, thickness=2, circle_radius=3
    )
    mp.solutions.drawing_utils.draw_landmarks(
        frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS, spec, spec
    )


def _draw_pointer(frame, cursor):
    """Small pointer so it never hides a keyboard key."""
    if cursor is None:
        return
    x, y = cursor
    cv2.circle(frame, (x, y), 5, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 9, PANEL, 2, cv2.LINE_AA)


def _blend_rect(frame, x1, y1, x2, y2, color, alpha):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def _draw_menu(frame, cursor, progress):
    h, w = frame.shape[:2]
    _blend_rect(frame, 0, TOP_BAR_HEIGHT, w, h, (18, 18, 18), MENU_ALPHA)

    cv2.putText(frame, "Choose how you want to communicate",
                (w // 2 - 265, TOP_BAR_HEIGHT + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "Point at an option and hold",
                (w // 2 - 170, TOP_BAR_HEIGHT + 77),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, MUTED, 1, cv2.LINE_AA)

    cards_top = TOP_BAR_HEIGHT + 135
    card_h, card_w = 210, 320
    cards = [
        (w // 2 - 360, "VIRTUAL KEYBOARD", "Letters - numbers - symbols", "Point + hold to type"),
        (w // 2 + 40, "HAND LETTERS", "A-Z fingerspelling", "Hold a sign to spell"),
    ]

    for x, title, subtitle, hint in cards:
        active = (cursor is not None and x <= cursor[0] <= x + card_w and
                  cards_top <= cursor[1] <= cards_top + card_h)
        _blend_rect(frame, x, cards_top, x + card_w, cards_top + card_h,
                    (42, 42, 42), MENU_ALPHA)
        cv2.rectangle(frame, (x, cards_top), (x + card_w, cards_top + card_h),
                      ACCENT if active else BORDER, 2 if active else 1)
        cv2.putText(frame, title, (x + 25, cards_top + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x + 25, cards_top + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, hint, (x + 25, cards_top + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    ACCENT if active else MUTED, 1, cv2.LINE_AA)
        if active:
            bar_w = int((card_w - 50) * progress)
            cv2.rectangle(frame, (x + 25, cards_top + 176),
                          (x + 25 + bar_w, cards_top + 180), ACCENT, -1)
            cv2.rectangle(frame, (x + 25, cards_top + 176),
                          (x + card_w - 25, cards_top + 180), BORDER, 1)

    _draw_pointer(frame, cursor)
    cv2.putText(frame,
                "1: Keyboard   2: Hand Letters   |   TAB: Menu   |   ESC: Exit",
                (w // 2 - 290, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                MUTED, 1, cv2.LINE_AA)


def _menu_selection(cursor, state):
    now = time.time()
    if cursor is None:
        state["key"] = None
        state["start"] = 0.0
        return None, 0.0

    x, y = cursor
    w = state["width"]
    cards_top = TOP_BAR_HEIGHT + 135
    card_h = 210

    if w // 2 - 360 <= x <= w // 2 - 40 and cards_top <= y <= cards_top + card_h:
        label = MODE_KEYBOARD
    elif w // 2 + 40 <= x <= w // 2 + 360 and cards_top <= y <= cards_top + card_h:
        label = MODE_LETTERS
    else:
        label = None

    if label != state["key"]:
        state["key"] = label
        state["start"] = now

    if label is None:
        return None, 0.0

    progress = min((now - state["start"]) / MODE_DWELL_SECONDS, 1.0)
    return (label if progress >= 1.0 else None), progress


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try CAMERA_INDEX = 1 in config.py.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # The webcam does not always honor the requested capture size above
    # (e.g. it may fall back to 640x480 instead of 1280x720). Every
    # layout that anchors itself to the bottom or edges of the frame
    # (the keyboard in particular) must be built from the size the
    # camera is *actually* delivering, not the configured FRAME_WIDTH/
    # FRAME_HEIGHT - otherwise a bottom-docked panel can be positioned
    # below the real frame's visible/reachable area entirely.
    ok, probe_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read an initial frame from the webcam.")
    actual_h, actual_w = probe_frame.shape[:2]
    if (actual_w, actual_h) != (FRAME_WIDTH, FRAME_HEIGHT):
        print(
            f"[GestureAccess] Webcam delivered {actual_w}x{actual_h} instead of "
            f"the requested {FRAME_WIDTH}x{FRAME_HEIGHT}; laying out UI for the "
            "actual size."
        )

    nav = gu.HandGestureProcessor(
        detection_confidence=0.65,
        tracking_confidence=0.65,
        # Raised from 0.9: gestures were confirming too quickly for
        # comfortable, deliberate use.
        dwell_seconds=1.3,
    )
    keyboard = VirtualKeyboard(actual_w, actual_h)
    spelling = fs.SpellingSession(dwell_seconds=0.72)
    cursor_smoother = gu.CursorSmoother(alpha=CURSOR_SMOOTHING_ALPHA)

    mode = MODE_MENU
    menu_state = {"key": None, "start": 0.0, "width": FRAME_WIDTH}
    show_cheat_sheet = False
    fps = 0.0
    previous = time.perf_counter()

    try:
        consecutive_failures = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    raise RuntimeError("Webcam stopped returning frames.")
                time.sleep(0.01)
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            menu_state["width"] = w

            result = nav.process(
                frame,
                process_size=(PROCESS_WIDTH, PROCESS_HEIGHT),
            )

            raw_cursor = _raw_cursor(
                frame, result.hand_landmarks, result.gesture, cursor_smoother
            )
            cursor = raw_cursor

            if mode == MODE_MENU:
                chosen, progress = _menu_selection(cursor, menu_state)
                _draw_menu(frame, cursor, progress)

                if chosen == MODE_KEYBOARD:
                    mode = MODE_KEYBOARD
                    keyboard.reset_enter()
                    menu_state["key"] = None
                    menu_state["start"] = 0.0
                elif chosen == MODE_LETTERS:
                    mode = MODE_LETTERS
                    spelling.reset()
                    menu_state["key"] = None
                    menu_state["start"] = 0.0

            elif mode == MODE_KEYBOARD:
                # Point + hold selects keys. Pinch is also accepted as an
                # intentional UI-selection gesture.
                keyboard.update(
                    cursor
                    if result.gesture in (
                        gu.GESTURE_INDEX_POINT,
                        gu.GESTURE_PINCH,
                    )
                    else None
                )

                # ENTER only reports that keyboard input is finished.
                # There is deliberately no profile/field submission here.
                if keyboard.enter_pressed:
                    keyboard.reset_enter()

                keyboard.draw(frame)
                _draw_pointer(frame, cursor)
                _draw_top_bar(frame, "VIRTUAL KEYBOARD", fps)

            elif mode == MODE_LETTERS:
                if result.hand_landmarks is not None:
                    points = gu._landmarks_to_array(result.hand_landmarks)
                    states = gu.compute_finger_states(points)
                    bbox = result.bounding_box
                else:
                    points = states = bbox = None

                spelling_result = spelling.process(
                    points, states, bbox, result.hand_landmarks
                )

                fs.draw_spelling_panel(
                    frame,
                    spelling_result,
                    typing_enabled=False,
                    top_offset=TOP_BAR_HEIGHT,
                    field_label="",
                )
                _draw_hand(frame, result.hand_landmarks)
                _draw_top_bar(frame, "HAND LETTERS", fps)
                cv2.putText(
                    frame,
                    "A-Z fingerspelling  |  Hold a stable sign  |  H = letter guide",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    MUTED, 1, cv2.LINE_AA,
                )
                if show_cheat_sheet:
                    fs.draw_cheat_sheet(frame)

            now = time.perf_counter()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps = instant if fps == 0 else fps * 0.88 + instant * 0.12

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == 9:
                mode = MODE_MENU
                spelling.reset()
                keyboard.reset_enter()
            elif key == ord("1"):
                mode = MODE_KEYBOARD
                keyboard.reset_enter()
            elif key == ord("2"):
                mode = MODE_LETTERS
                spelling.reset()
            elif key == CHEAT_SHEET_KEY and mode == MODE_LETTERS:
                show_cheat_sheet = not show_cheat_sheet

    finally:
        nav.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
