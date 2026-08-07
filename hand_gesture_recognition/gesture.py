"""Utility module for the Hand Gesture Recognition System.

This module contains all logic required to detect and classify hand
gestures using MediaPipe Hands, including landmark extraction, finger
state detection, gesture classification, pinch detection, side-sweep
detection, virtual cursor smoothing, dwell-time confirmation, gesture
to action mapping, and all OpenCV drawing helpers used to render the
application's user interface.

Everything in the application except webcam initialization and the
main capture loop lives in this module, per the project's Software
Requirements Specification.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DWELL_TIME_SECONDS: float = 1.5
"""Time a gesture must remain stable before it is confirmed."""

PINCH_DISTANCE_THRESHOLD: float = 0.055
"""Normalized (0-1) distance below which thumb and index tips are
considered to be touching."""

THUMB_UP_VERTICAL_MARGIN: float = 0.05
"""Minimum normalized vertical distance the thumb tip must be above
the thumb MCP joint for the thumb to be considered pointing upward."""

SWEEP_HISTORY_SIZE: int = 12
"""Number of recent palm x-positions tracked for side-sweep detection."""

SWEEP_MIN_DISPLACEMENT: float = 0.22
"""Minimum normalized horizontal displacement of the palm required to
classify an open-palm motion as a side sweep."""

SWEEP_LATCH_SECONDS: float = 2.0
"""Seconds the SIDE SWEEP classification is held after the swipe motion
is detected, so the gesture stays stable long enough for dwell-time
confirmation. Must be greater than DWELL_TIME_SECONDS."""

CURSOR_SMOOTHING_ALPHA: float = 0.35
"""Exponential smoothing factor applied to virtual cursor movement.
Lower values produce smoother, less jittery motion."""

BOUNDING_BOX_PADDING: int = 20
"""Pixel padding applied around the hand bounding box."""

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# ---------------------------------------------------------------------------
# Gesture identifiers and mapped actions
# ---------------------------------------------------------------------------

GESTURE_NONE: str = "NO GESTURE"
GESTURE_THUMBS_UP: str = "THUMBS UP"
GESTURE_OPEN_PALM: str = "OPEN PALM"
GESTURE_SIDE_SWEEP: str = "SIDE SWEEP"
GESTURE_INDEX_POINT: str = "INDEX POINTING"
GESTURE_PINCH: str = "PINCH"

GESTURE_ACTIONS: dict[str, str] = {
    GESTURE_THUMBS_UP: "Confirm Selection / Save Field",
    GESTURE_OPEN_PALM: "Move to Next Field",
    GESTURE_SIDE_SWEEP: "Previous Field / Cancel Action",
    GESTURE_INDEX_POINT: "Virtual Cursor Positioning",
    GESTURE_PINCH: "Select Checkbox / Radio / Dropdown Item",
    GESTURE_NONE: "No Action",
}

# ---------------------------------------------------------------------------
# Colors (BGR) and MediaPipe drawing objects
# ---------------------------------------------------------------------------

COLOR_CONFIRMED: tuple[int, int, int] = (0, 200, 0)
COLOR_DETECTING: tuple[int, int, int] = (0, 210, 255)
COLOR_NONE: tuple[int, int, int] = (0, 0, 220)
COLOR_PANEL_BG: tuple[int, int, int] = (30, 30, 30)
COLOR_TEXT: tuple[int, int, int] = (255, 255, 255)
COLOR_CURSOR: tuple[int, int, int] = (0, 255, 0)

_MP_HANDS = mp.solutions.hands
_MP_DRAWING = mp.solutions.drawing_utils
_MP_STYLES = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GestureResult:
    """Container for a single frame's gesture-recognition output.

    Attributes:
        gesture: Name of the currently classified gesture.
        action: Human-readable action mapped to the gesture.
        confirmed: Whether the gesture has completed its dwell period.
        dwell_progress: Fraction (0.0-1.0) of dwell time elapsed.
        cursor_position: Smoothed pixel coordinates of the virtual
            cursor, or ``None`` if no cursor should be shown.
        bounding_box: Pixel bounding box ``(x_min, y_min, x_max,
            y_max)`` of the detected hand, or ``None`` if no hand is
            detected.
        hand_landmarks: Raw MediaPipe hand landmarks for the detected
            hand, used for drawing, or ``None`` if no hand is
            detected.
        action_triggered: ``True`` only on the single frame where the
            gesture transitions from detecting to confirmed.
    """

    gesture: str
    action: str
    confirmed: bool
    dwell_progress: float
    cursor_position: Optional[tuple[int, int]]
    bounding_box: Optional[tuple[int, int, int, int]]
    hand_landmarks: Optional[object]
    action_triggered: bool = False


# ---------------------------------------------------------------------------
# Helper: landmark extraction and finger-state detection
# ---------------------------------------------------------------------------


def _landmarks_to_array(hand_landmarks: object) -> np.ndarray:
    """Convert MediaPipe hand landmarks to a NumPy array.

    Args:
        hand_landmarks: A MediaPipe ``NormalizedLandmarkList`` for a
            single detected hand.

    Returns:
        An array of shape ``(21, 3)`` holding normalized ``(x, y, z)``
        coordinates for each landmark.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float64,
    )


def _is_finger_extended(points: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
    """Determine whether a non-thumb finger is extended.

    A finger is considered extended when its tip landmark is higher
    (smaller normalized y-value) than its PIP joint landmark.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        tip_idx: Index of the fingertip landmark.
        pip_idx: Index of the finger's PIP joint landmark.

    Returns:
        ``True`` if the finger is extended, ``False`` otherwise.
    """
    return bool(points[tip_idx][1] < points[pip_idx][1])


def _is_thumb_extended(points: np.ndarray) -> bool:
    """Determine whether the thumb is extended away from the palm.

    Uses a distance-based heuristic that is independent of hand
    orientation: the thumb tip must be farther from the pinky MCP
    joint than the thumb MCP joint is, indicating the thumb has
    spread away from the palm.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.

    Returns:
        ``True`` if the thumb is extended, ``False`` otherwise.
    """
    tip_to_pinky = float(np.linalg.norm(points[THUMB_TIP][:2] - points[PINKY_MCP][:2]))
    mcp_to_pinky = float(np.linalg.norm(points[THUMB_MCP][:2] - points[PINKY_MCP][:2]))
    return tip_to_pinky > mcp_to_pinky * 1.1


def compute_finger_states(points: np.ndarray) -> dict[str, bool]:
    """Compute the extended/folded state of all five fingers.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.

    Returns:
        A mapping of finger name to a boolean indicating whether it
        is extended.
    """
    return {
        "thumb": _is_thumb_extended(points),
        "index": _is_finger_extended(points, INDEX_TIP, INDEX_PIP),
        "middle": _is_finger_extended(points, MIDDLE_TIP, MIDDLE_PIP),
        "ring": _is_finger_extended(points, RING_TIP, RING_PIP),
        "pinky": _is_finger_extended(points, PINKY_TIP, PINKY_PIP),
    }


def is_pinch(points: np.ndarray) -> bool:
    """Detect whether the thumb tip and index fingertip are touching.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.

    Returns:
        ``True`` if the normalized distance between the thumb tip and
        index fingertip is below :data:`PINCH_DISTANCE_THRESHOLD`.
    """
    distance = float(np.linalg.norm(points[THUMB_TIP][:2] - points[INDEX_TIP][:2]))
    return distance < PINCH_DISTANCE_THRESHOLD


def _is_thumbs_up(points: np.ndarray, finger_states: dict[str, bool]) -> bool:
    """Detect the THUMBS UP gesture.

    Requires the thumb to be extended and pointing clearly upward
    while the remaining four fingers are folded.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        finger_states: Precomputed finger extension states.

    Returns:
        ``True`` if the THUMBS UP gesture is detected.
    """
    others_folded = not any(
        (finger_states["index"], finger_states["middle"], finger_states["ring"], finger_states["pinky"])
    )
    thumb_points_up = points[THUMB_TIP][1] < points[THUMB_MCP][1] - THUMB_UP_VERTICAL_MARGIN
    return finger_states["thumb"] and thumb_points_up and others_folded


def _is_index_pointing(finger_states: dict[str, bool]) -> bool:
    """Detect the INDEX POINTING gesture.

    Requires only the index finger to be extended, with all other
    fingers folded.

    Args:
        finger_states: Precomputed finger extension states.

    Returns:
        ``True`` if the INDEX POINTING gesture is detected.
    """
    others_folded = not any(
        (finger_states["thumb"], finger_states["middle"], finger_states["ring"], finger_states["pinky"])
    )
    return finger_states["index"] and others_folded


def _is_open_palm(finger_states: dict[str, bool]) -> bool:
    """Detect an OPEN PALM (all five fingers extended).

    Args:
        finger_states: Precomputed finger extension states.

    Returns:
        ``True`` if all five fingers are extended.
    """
    return all(finger_states.values())


def classify_gesture(points: np.ndarray, finger_states: dict[str, bool], sweep_detected: bool) -> str:
    """Classify the current hand pose into one of the supported gestures.

    Classification priority: PINCH, THUMBS UP, INDEX POINTING, then
    OPEN PALM (or SIDE SWEEP if horizontal motion was detected while
    the palm was open). Falls back to :data:`GESTURE_NONE`.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        finger_states: Precomputed finger extension states.
        sweep_detected: Whether a horizontal side-sweep motion was
            detected during an open-palm pose.

    Returns:
        The name of the classified gesture.
    """
    if is_pinch(points):
        return GESTURE_PINCH
    if _is_thumbs_up(points, finger_states):
        return GESTURE_THUMBS_UP
    if _is_index_pointing(finger_states):
        return GESTURE_INDEX_POINT
    if _is_open_palm(finger_states):
        return GESTURE_SIDE_SWEEP if sweep_detected else GESTURE_OPEN_PALM
    return GESTURE_NONE


def _compute_bounding_box(
    points: np.ndarray, frame_width: int, frame_height: int, padding: int = BOUNDING_BOX_PADDING
) -> tuple[int, int, int, int]:
    """Compute a padded pixel bounding box around all hand landmarks.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        frame_width: Width of the source frame in pixels.
        frame_height: Height of the source frame in pixels.
        padding: Extra pixels to pad around the tight bounding box.

    Returns:
        A tuple ``(x_min, y_min, x_max, y_max)`` in pixel coordinates,
        clamped to the frame boundaries.
    """
    xs = points[:, 0] * frame_width
    ys = points[:, 1] * frame_height
    x_min = max(int(xs.min()) - padding, 0)
    y_min = max(int(ys.min()) - padding, 0)
    x_max = min(int(xs.max()) + padding, frame_width - 1)
    y_max = min(int(ys.max()) + padding, frame_height - 1)
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# Side-sweep detection
# ---------------------------------------------------------------------------


class SideSweepDetector:
    """Detects horizontal side-sweep motion during an open-palm pose.

    Once a swipe is detected, the SIDE SWEEP state is latched for
    :data:`SWEEP_LATCH_SECONDS` (as long as the palm stays open) so the
    classification remains stable long enough for the dwell-time
    confirmation mechanism to complete. Without this latch, the
    gesture would only exist for a single frame and could never be
    confirmed.
    """

    def __init__(
        self,
        history_size: int = SWEEP_HISTORY_SIZE,
        min_displacement: float = SWEEP_MIN_DISPLACEMENT,
        latch_seconds: float = SWEEP_LATCH_SECONDS,
    ) -> None:
        """Initialize the detector.

        Args:
            history_size: Number of recent palm x-positions to retain.
            min_displacement: Minimum normalized horizontal spread
                across the retained history required to register a
                sweep.
            latch_seconds: Seconds to hold the SIDE SWEEP state after
                a swipe is detected.
        """
        self._history: deque[float] = deque(maxlen=history_size)
        self._min_displacement = min_displacement
        self._latch_seconds = latch_seconds
        self._latch_until: Optional[float] = None

    def update(self, palm_x: float, is_open_palm: bool) -> bool:
        """Update the motion history and check for a sweep.

        Args:
            palm_x: Normalized x-coordinate of the palm center.
            is_open_palm: Whether the hand currently forms an open
                palm. The history and latch reset whenever the palm
                closes.

        Returns:
            ``True`` if a sweep was just detected or is still latched
            from a recent detection.
        """
        if not is_open_palm:
            self._history.clear()
            self._latch_until = None
            return False

        now = time.time()

        if self._latch_until is not None:
            if now < self._latch_until:
                return True
            self._latch_until = None

        self._history.append(palm_x)
        if len(self._history) < (self._history.maxlen or 1):
            return False

        displacement = max(self._history) - min(self._history)
        if displacement >= self._min_displacement:
            self._history.clear()
            self._latch_until = now + self._latch_seconds
            return True
        return False

    def reset(self) -> None:
        """Clear the motion history and any active latch."""
        self._history.clear()
        self._latch_until = None


# ---------------------------------------------------------------------------
# Cursor smoothing
# ---------------------------------------------------------------------------


class CursorSmoother:
    """Applies exponential smoothing to virtual cursor coordinates."""

    def __init__(self, alpha: float = CURSOR_SMOOTHING_ALPHA) -> None:
        """Initialize the smoother.

        Args:
            alpha: Smoothing factor in ``(0, 1]``. Lower values yield
                smoother, less jittery motion at the cost of latency.
        """
        self._alpha = alpha
        self._smoothed: Optional[tuple[float, float]] = None

    def update(self, x: float, y: float) -> tuple[int, int]:
        """Feed a new raw coordinate and return the smoothed result.

        Args:
            x: Raw x-coordinate in pixels.
            y: Raw y-coordinate in pixels.

        Returns:
            The smoothed ``(x, y)`` coordinate as integer pixels.
        """
        if self._smoothed is None:
            self._smoothed = (x, y)
        else:
            prev_x, prev_y = self._smoothed
            self._smoothed = (
                self._alpha * x + (1.0 - self._alpha) * prev_x,
                self._alpha * y + (1.0 - self._alpha) * prev_y,
            )
        return int(self._smoothed[0]), int(self._smoothed[1])

    def reset(self) -> None:
        """Reset the smoother, discarding any prior state."""
        self._smoothed = None


# ---------------------------------------------------------------------------
# Dwell-time confirmation
# ---------------------------------------------------------------------------


class DwellTimer:
    """Confirms a gesture only after it remains stable for a fixed time."""

    def __init__(self, dwell_seconds: float = DWELL_TIME_SECONDS) -> None:
        """Initialize the dwell timer.

        Args:
            dwell_seconds: Seconds a gesture must remain unchanged
                before it is confirmed.
        """
        self._dwell_seconds = dwell_seconds
        self._current_gesture: str = GESTURE_NONE
        self._start_time: Optional[float] = None
        self._triggered: bool = False

    def update(self, gesture: str) -> tuple[bool, float, bool]:
        """Advance the dwell state machine with the latest gesture.

        Args:
            gesture: The gesture classified for the current frame.

        Returns:
            A tuple ``(confirmed, progress, just_triggered)`` where
            ``confirmed`` indicates the dwell period has elapsed,
            ``progress`` is the fraction (0.0-1.0) of dwell time
            elapsed, and ``just_triggered`` is ``True`` only on the
            single frame the gesture becomes confirmed.
        """
        now = time.time()

        if gesture != self._current_gesture:
            self._current_gesture = gesture
            self._start_time = now
            self._triggered = False

        if gesture == GESTURE_NONE or self._start_time is None:
            return False, 0.0, False

        elapsed = now - self._start_time
        progress = min(elapsed / self._dwell_seconds, 1.0)
        confirmed = progress >= 1.0

        just_triggered = confirmed and not self._triggered
        if confirmed:
            self._triggered = True

        return confirmed, progress, just_triggered


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------


class HandGestureProcessor:
    """Coordinates MediaPipe hand tracking and gesture recognition.

    This class owns the MediaPipe ``Hands`` solution instance along
    with the stateful helpers (dwell timer, cursor smoother, side-sweep
    detector) required to turn a raw BGR frame into a fully classified,
    confirmed, and mapped :class:`GestureResult`.
    """

    def __init__(
        self,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ) -> None:
        """Initialize MediaPipe Hands and internal state helpers.

        Args:
            detection_confidence: Minimum confidence for initial hand
                detection.
            tracking_confidence: Minimum confidence for landmark
                tracking across frames.
        """
        self._hands = _MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._dwell_timer = DwellTimer()
        self._sweep_detector = SideSweepDetector()
        self._cursor_smoother = CursorSmoother()

    def process(self, frame_bgr: np.ndarray) -> GestureResult:
        """Process a single BGR frame and produce a gesture result.

        Args:
            frame_bgr: The current webcam frame in BGR color order, as
                returned by ``cv2.VideoCapture.read``.

        Returns:
            A fully populated :class:`GestureResult` for this frame.
        """
        height, width = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        detection = self._hands.process(rgb_frame)

        if not detection.multi_hand_landmarks:
            self._cursor_smoother.reset()
            self._sweep_detector.reset()
            self._dwell_timer.update(GESTURE_NONE)
            return GestureResult(
                gesture=GESTURE_NONE,
                action=GESTURE_ACTIONS[GESTURE_NONE],
                confirmed=False,
                dwell_progress=0.0,
                cursor_position=None,
                bounding_box=None,
                hand_landmarks=None,
                action_triggered=False,
            )

        hand_landmarks = detection.multi_hand_landmarks[0]
        points = _landmarks_to_array(hand_landmarks)
        finger_states = compute_finger_states(points)

        palm_x = float(
            np.mean(
                [
                    points[INDEX_MCP][0],
                    points[MIDDLE_MCP][0],
                    points[RING_MCP][0],
                    points[PINKY_MCP][0],
                    points[WRIST][0],
                ]
            )
        )
        palm_open = _is_open_palm(finger_states)
        sweep_detected = self._sweep_detector.update(palm_x, palm_open)

        gesture = classify_gesture(points, finger_states, sweep_detected)
        confirmed, progress, action_triggered = self._dwell_timer.update(gesture)

        cursor_position: Optional[tuple[int, int]] = None
        if gesture == GESTURE_INDEX_POINT:
            raw_x = points[INDEX_TIP][0] * width
            raw_y = points[INDEX_TIP][1] * height
            cursor_position = self._cursor_smoother.update(raw_x, raw_y)
        else:
            self._cursor_smoother.reset()

        bounding_box = _compute_bounding_box(points, width, height)

        if action_triggered:
            print(f"[ACTION TRIGGERED] {gesture} -> {GESTURE_ACTIONS[gesture]}")

        return GestureResult(
            gesture=gesture,
            action=GESTURE_ACTIONS[gesture],
            confirmed=confirmed,
            dwell_progress=progress,
            cursor_position=cursor_position,
            bounding_box=bounding_box,
            hand_landmarks=hand_landmarks,
            action_triggered=action_triggered,
        )

    def close(self) -> None:
        """Release the underlying MediaPipe ``Hands`` resources."""
        self._hands.close()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def get_status_color(confirmed: bool, gesture: str) -> tuple[int, int, int]:
    """Select the status color for the current gesture state.

    Args:
        confirmed: Whether the current gesture has been confirmed.
        gesture: Name of the current gesture.

    Returns:
        A BGR color tuple: green if confirmed, yellow if detecting,
        red if no gesture is present.
    """
    if gesture == GESTURE_NONE:
        return COLOR_NONE
    if confirmed:
        return COLOR_CONFIRMED
    return COLOR_DETECTING


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks: Optional[object]) -> None:
    """Draw MediaPipe hand landmarks and connections onto a frame.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        hand_landmarks: Raw MediaPipe landmarks for the detected hand,
            or ``None`` if no hand is present.
    """
    if hand_landmarks is None:
        return
    _MP_DRAWING.draw_landmarks(
        frame,
        hand_landmarks,
        _MP_HANDS.HAND_CONNECTIONS,
        _MP_STYLES.get_default_hand_landmarks_style(),
        _MP_STYLES.get_default_hand_connections_style(),
    )


def draw_bounding_box(frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    """Draw the hand bounding box onto a frame.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        bbox: Bounding box as ``(x_min, y_min, x_max, y_max)``.
        color: BGR color used for the rectangle outline.
    """
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)


def draw_gesture_label(frame: np.ndarray, bbox: tuple[int, int, int, int], gesture: str, color: tuple[int, int, int]) -> None:
    """Draw the classified gesture name above the bounding box.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        bbox: Bounding box as ``(x_min, y_min, x_max, y_max)``.
        gesture: Name of the current gesture.
        color: BGR color used for the label text and background.
    """
    x_min, y_min, x_max, _ = bbox
    label = gesture
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(y_min - 10, text_h + 10)
    cv2.rectangle(frame, (x_min, label_y - text_h - 8), (x_min + text_w + 10, label_y + 4), color, -1)
    cv2.putText(frame, label, (x_min + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def draw_cursor(frame: np.ndarray, position: tuple[int, int]) -> None:
    """Draw the virtual cursor as a labeled green circle.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        position: Pixel coordinates ``(x, y)`` of the cursor.
    """
    x, y = position
    cv2.circle(frame, (x, y), 12, COLOR_CURSOR, -1)
    cv2.circle(frame, (x, y), 16, COLOR_CURSOR, 2)
    cv2.putText(
        frame,
        "CURSOR",
        (x + 20, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_CURSOR,
        2,
        cv2.LINE_AA,
    )


def draw_info_panel(frame: np.ndarray, fps: float, result: GestureResult) -> None:
    """Draw the top information panel summarizing recognition state.

    Displays FPS, current gesture, mapped action, confirmation status,
    dwell timer progress, and cursor position.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        fps: Current frames-per-second estimate.
        result: The gesture result for the current frame.
    """
    panel_height = 150
    frame_width = frame.shape[1]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, panel_height), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    status_color = get_status_color(result.confirmed, result.gesture)
    if result.gesture == GESTURE_NONE:
        status_text = "No Gesture"
    elif result.confirmed:
        status_text = "Confirmed"
    else:
        status_text = "Detecting..."

    cursor_text = f"{result.cursor_position}" if result.cursor_position is not None else "N/A"
    dwell_text = f"{result.dwell_progress * 100:.0f}%"

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"FPS: {fps:.1f}", COLOR_TEXT),
        (f"Gesture: {result.gesture}", COLOR_TEXT),
        (f"Action: {result.action}", COLOR_TEXT),
        (f"Status: {status_text}", status_color),
        (f"Dwell: {dwell_text}", status_color),
        (f"Cursor: {cursor_text}", COLOR_TEXT),
    ]

    col_x = [10, 10, 10, frame_width // 2, frame_width // 2, frame_width // 2]
    row_y = [25, 55, 85, 25, 55, 85]

    for (text, color), x, y in zip(lines, col_x, row_y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 10, 110, frame_width - 20, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    filled_w = int(bar_w * result.dwell_progress)
    if filled_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), status_color, -1)