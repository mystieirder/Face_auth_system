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
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DWELL_TIME_SECONDS: float = 1.5
"""Default time a gesture must remain stable before it is confirmed.
Per FR-6.4 this is a *default* — HandGestureProcessor accepts an
override so the value can be made configurable (e.g. from an
accessibility settings panel) without editing this module."""

PINCH_DISTANCE_RATIO: float = 0.25
"""Thumb tip and index tip are considered touching when their
distance is below this fraction of the hand's own scale (see
:func:`_hand_scale`), rather than a fixed normalized-image distance.
MediaPipe's landmark coordinates are normalized to the *frame*, not
to the hand, so a fixed absolute threshold (e.g. 0.055) makes pinch
detection scale-dependent: the same physical gesture produces a
smaller apparent distance as the hand moves farther from the camera,
causing false pinch detections during normal, non-pinching motion.
Scaling the threshold by the hand's own size keeps detection
consistent regardless of distance from the camera. Kept fairly tight
(0.25, meaning tips must be genuinely close) so that natural hand
poses where the thumb isn't fully spread away from the palm — e.g. a
relaxed open palm — aren't mistaken for a deliberate pinch."""

THUMB_UP_VERTICAL_RATIO: float = 0.20
"""The thumb tip must be above the thumb MCP joint by at least this
fraction of the hand's own scale (see :func:`_hand_scale`) for the
thumb to be considered pointing upward. Scale-invariant for the same
reason as :data:`PINCH_DISTANCE_RATIO`."""

SWEEP_HISTORY_SIZE: int = 12
"""Number of recent palm x-positions tracked for side-sweep detection."""

SWEEP_MIN_DISPLACEMENT: float = 0.22
"""Minimum normalized horizontal displacement of the palm required to
classify an open-palm motion as a side sweep."""

SWEEP_LATCH_SECONDS: float = 2.0
"""Seconds the SIDE SWEEP classification is held after the swipe motion
is detected, so the gesture stays stable long enough for dwell-time
confirmation. Must be greater than DWELL_TIME_SECONDS."""

SWEEP_OPEN_PALM_GRACE_FRAMES: int = 6
"""Number of consecutive non-open-palm frames tolerated before the
side-sweep motion history is reset. A lateral sweep naturally tilts
the hand slightly frame-to-frame, which can cause a single frame's
open-palm check to flicker false; without this grace period the
motion history would be cleared before enough displacement is ever
recorded, and a sweep could never be detected."""

FINGER_EXTENSION_RATIO: float = 1.4
"""A finger is considered extended when the wrist-to-fingertip
distance exceeds the wrist-to-MCP distance by this factor. This
distance-based test (rather than comparing raw y-coordinates) keeps
finger-extension detection accurate even when the hand is tilted or
rotated relative to the camera, instead of only when held perfectly
upright. Set high enough (1.4) that a finger must be decisively
straightened, not just slightly uncurled, to count as extended."""

GESTURE_STABILITY_FRAMES: int = 8
"""Number of consecutive frames a newly classified gesture must
persist before it replaces the previously reported gesture. Every
frame carries some landmark jitter (motion blur, tracking noise,
brief mid-transition hand shapes), and without this debounce a single
noisy frame changes the reported gesture and resets
:class:`DwellTimer`'s progress back to zero — so a gesture can flicker
indefinitely between two or three labels and never hold long enough
to confirm.

Raised from 4 to 8: gestures were being recognized/switched too
quickly for comfortable, deliberate use, so requiring roughly twice
as many stable frames slows down how fast the app reacts to a new
hand shape."""

CURSOR_SMOOTHING_ALPHA: float = 0.35
"""Exponential smoothing factor applied to virtual cursor movement.
Lower values produce smoother, less jittery motion."""

BOUNDING_BOX_PADDING: int = 20
"""Pixel padding applied around the hand bounding box."""

MIN_ACCEPTABLE_FPS: float = 24.0
"""Minimum gesture-tracking frame rate required by NFR-3 of the SRS
(\"Gesture Tracking Frame Rate >= 24 FPS, standard integrated GPU/CPU
hardware\"). Used to flag performance degradation in the info panel."""

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
COLOR_WARNING: tuple[int, int, int] = (0, 140, 255)

# The virtual cursor is drawn with a two-tone (white core / black ring)
# treatment rather than a single flat color. A solid single-color dot
# (e.g. pure green) can disappear against a same-colored background in
# the live webcam feed, which fails the AAA-target contrast requirement
# in NFR-2 (>= 7:1) for input-accessibility elements. White-on-black
# guarantees strong contrast against virtually any background.
COLOR_CURSOR_CORE: tuple[int, int, int] = (255, 255, 255)
COLOR_CURSOR_OUTLINE: tuple[int, int, int] = (0, 0, 0)

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
        diagnostics: Raw threshold-tuning values for the current frame
            (pinch ratio, extension ratios, sweep displacement, etc.),
            empty if no hand is detected. Intended for the optional
            debug overlay (:func:`draw_diagnostics_panel`), not for
            gesture logic itself.
    """

    gesture: str
    action: str
    confirmed: bool
    dwell_progress: float
    cursor_position: Optional[tuple[int, int]]
    bounding_box: Optional[tuple[int, int, int, int]]
    hand_landmarks: Optional[object]
    action_triggered: bool = False
    diagnostics: dict[str, float] = field(default_factory=dict)


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


def _hand_scale(points: np.ndarray) -> float:
    """Compute a distance-from-camera-invariant measure of hand size.

    Uses the distance between the wrist and the middle finger's MCP
    joint as a stable proxy for how large the hand appears in the
    current frame. Other thresholds (pinch distance, thumb-up margin)
    are expressed as a fraction of this value so they hold steady
    regardless of how close the hand is to the camera.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.

    Returns:
        The normalized wrist-to-middle-MCP distance. Never zero in
        practice since these are two distinct physical joints.
    """
    return float(np.linalg.norm(points[MIDDLE_MCP][:2] - points[WRIST][:2]))


def compute_diagnostics(
    points: np.ndarray, finger_states: dict[str, bool], sweep_spread: float
) -> dict[str, float]:
    """Compute raw threshold-tuning values for the current frame.

    These are the same underlying measurements the classification
    functions use, exposed directly so the live numbers can be
    compared against their thresholds on screen (via
    :func:`draw_diagnostics_panel`) instead of tuning constants blind.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        finger_states: Precomputed finger extension states.
        sweep_spread: Current tracked horizontal displacement from
            :meth:`SideSweepDetector.current_spread`.

    Returns:
        A mapping of diagnostic label to its current value, alongside
        the configured threshold it will be compared against.
    """
    scale = _hand_scale(points)
    pinch_distance = float(np.linalg.norm(points[THUMB_TIP][:2] - points[INDEX_TIP][:2]))
    pinch_ratio = (pinch_distance / scale) if scale > 0 else 0.0
    thumb_margin_ratio = ((points[THUMB_MCP][1] - points[THUMB_TIP][1]) / scale) if scale > 0 else 0.0

    extension_ratios: dict[str, float] = {}
    for name, tip_idx, mcp_idx in (
        ("index", INDEX_TIP, INDEX_MCP),
        ("middle", MIDDLE_TIP, MIDDLE_MCP),
        ("ring", RING_TIP, RING_MCP),
        ("pinky", PINKY_TIP, PINKY_MCP),
    ):
        wrist = points[WRIST][:2]
        tip_dist = float(np.linalg.norm(points[tip_idx][:2] - wrist))
        mcp_dist = float(np.linalg.norm(points[mcp_idx][:2] - wrist))
        extension_ratios[name] = (tip_dist / mcp_dist) if mcp_dist > 0 else 0.0

    return {
        "pinch_ratio": pinch_ratio,
        "pinch_ratio_threshold": PINCH_DISTANCE_RATIO,
        "thumb_margin_ratio": thumb_margin_ratio,
        "thumb_margin_threshold": THUMB_UP_VERTICAL_RATIO,
        "ext_index": extension_ratios["index"],
        "ext_middle": extension_ratios["middle"],
        "ext_ring": extension_ratios["ring"],
        "ext_pinky": extension_ratios["pinky"],
        "ext_threshold": FINGER_EXTENSION_RATIO,
        "sweep_spread": sweep_spread,
        "sweep_spread_threshold": SWEEP_MIN_DISPLACEMENT,
    }


def _is_finger_extended(points: np.ndarray, tip_idx: int, mcp_idx: int) -> bool:
    """Determine whether a non-thumb finger is extended.

    Uses a distance-based heuristic that is independent of hand
    orientation: the fingertip must be farther from the wrist than
    the finger's MCP (knuckle) joint is, by at least
    :data:`FINGER_EXTENSION_RATIO`. Comparing raw y-coordinates
    instead (tip higher than PIP on screen) only works when the hand
    is held perfectly upright facing the camera; it misclassifies
    fingers as soon as the hand tilts or rotates, which happens
    naturally during motions like a side sweep.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        tip_idx: Index of the fingertip landmark.
        mcp_idx: Index of the finger's MCP (knuckle) joint landmark.

    Returns:
        ``True`` if the finger is extended, ``False`` otherwise.
    """
    wrist = points[WRIST][:2]
    tip_dist = float(np.linalg.norm(points[tip_idx][:2] - wrist))
    mcp_dist = float(np.linalg.norm(points[mcp_idx][:2] - wrist))
    return tip_dist > mcp_dist * FINGER_EXTENSION_RATIO


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
        "index": _is_finger_extended(points, INDEX_TIP, INDEX_MCP),
        "middle": _is_finger_extended(points, MIDDLE_TIP, MIDDLE_MCP),
        "ring": _is_finger_extended(points, RING_TIP, RING_MCP),
        "pinky": _is_finger_extended(points, PINKY_TIP, PINKY_MCP),
    }


def is_pinch(points: np.ndarray) -> bool:
    """Detect whether the thumb tip and index fingertip are touching.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.

    Returns:
        ``True`` if the distance between the thumb tip and index
        fingertip, relative to the hand's own scale, is below
        :data:`PINCH_DISTANCE_RATIO`.
    """
    distance = float(np.linalg.norm(points[THUMB_TIP][:2] - points[INDEX_TIP][:2]))
    scale = _hand_scale(points)
    if scale <= 0:
        return False
    return (distance / scale) < PINCH_DISTANCE_RATIO


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
    scale = _hand_scale(points)
    margin = scale * THUMB_UP_VERTICAL_RATIO
    thumb_points_up = points[THUMB_TIP][1] < points[THUMB_MCP][1] - margin
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

    Classification priority: SIDE SWEEP (if a swipe is actively being
    tracked or latched) takes priority over everything else, then
    PINCH, THUMBS UP, INDEX POINTING, then OPEN PALM. Falls back to
    :data:`GESTURE_NONE`.

    SIDE SWEEP is checked first, unconditionally, because it is a
    deliberate, sustained, large-scale motion tracked over many
    frames by :class:`SideSweepDetector` — a much stronger signal than
    a single frame's static pinch/thumb/index check. Without this
    priority, a sweep that has already been detected internally could
    still be silently overridden and never reported, simply because
    the thumb and index tip happened to pass close together at that
    exact instant during the swipe motion.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        finger_states: Precomputed finger extension states.
        sweep_detected: Whether a horizontal side-sweep motion was
            detected during an open-palm pose.

    Returns:
        The name of the classified gesture.
    """
    if sweep_detected:
        return GESTURE_SIDE_SWEEP
    if is_pinch(points):
        return GESTURE_PINCH
    if _is_thumbs_up(points, finger_states):
        return GESTURE_THUMBS_UP
    if _is_index_pointing(finger_states):
        return GESTURE_INDEX_POINT
    if _is_open_palm(finger_states):
        return GESTURE_OPEN_PALM
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
        open_palm_grace_frames: int = SWEEP_OPEN_PALM_GRACE_FRAMES,
    ) -> None:
        """Initialize the detector.

        Args:
            history_size: Number of recent palm x-positions to retain.
            min_displacement: Minimum normalized horizontal spread
                across the retained history required to register a
                sweep.
            latch_seconds: Seconds to hold the SIDE SWEEP state after
                a swipe is detected.
            open_palm_grace_frames: Number of consecutive frames the
                open-palm check is allowed to read ``False`` before
                the motion history is reset. Tolerates the brief
                per-frame flicker caused by natural hand tilt during
                the sweep motion itself.
        """
        self._history: deque[float] = deque(maxlen=history_size)
        self._min_displacement = min_displacement
        self._latch_seconds = latch_seconds
        self._latch_until: Optional[float] = None
        self._open_palm_grace_frames = open_palm_grace_frames
        self._non_open_streak: int = 0

    def update(self, palm_x: float, is_open_palm: bool) -> bool:
        """Update the motion history and check for a sweep.

        Args:
            palm_x: Normalized x-coordinate of the palm center.
            is_open_palm: Whether the hand currently forms an open
                palm. The history and latch reset only after the palm
                reads closed for more than
                :data:`SWEEP_OPEN_PALM_GRACE_FRAMES` consecutive
                frames, tolerating brief flicker during the sweep
                motion itself.

        Returns:
            ``True`` if a sweep was just detected or is still latched
            from a recent detection.
        """
        if not is_open_palm:
            self._non_open_streak += 1
            if self._non_open_streak > self._open_palm_grace_frames:
                self._history.clear()
                self._latch_until = None
                self._non_open_streak = 0
            # Within the grace window: keep any accumulated history
            # and latch state, but don't record this frame's position.
            return self._latch_until is not None and time.time() < self._latch_until

        self._non_open_streak = 0
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
        self._non_open_streak = 0

    def current_spread(self) -> float:
        """Return the current tracked horizontal displacement.

        Exposed for the diagnostics overlay so the actual displacement
        being accumulated can be compared live against
        :data:`SWEEP_MIN_DISPLACEMENT` while tuning.

        Returns:
            ``max(history) - min(history)``, or ``0.0`` if fewer than
            two positions have been recorded yet.
        """
        if len(self._history) < 2:
            return 0.0
        return max(self._history) - min(self._history)


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


class GestureStabilizer:
    """Debounces raw per-frame gesture classifications.

    A newly classified gesture only replaces the previously reported
    gesture once it has appeared for :data:`GESTURE_STABILITY_FRAMES`
    consecutive frames. This absorbs single-frame landmark jitter
    (tracking noise, motion blur, brief mid-transition hand shapes)
    that would otherwise flip the reported gesture every frame and
    continually reset :class:`DwellTimer`'s progress, preventing any
    gesture from ever being held long enough to confirm.
    """

    def __init__(self, required_frames: int = GESTURE_STABILITY_FRAMES) -> None:
        """Initialize the stabilizer.

        Args:
            required_frames: Number of consecutive matching raw
                classifications required before the reported gesture
                changes.
        """
        self._required_frames = max(1, required_frames)
        self._candidate: str = GESTURE_NONE
        self._candidate_count: int = 0
        self._stable: str = GESTURE_NONE

    def update(self, raw_gesture: str) -> str:
        """Feed the latest raw classification and get the stable gesture.

        Args:
            raw_gesture: The gesture classified for the current frame,
                before debouncing.

        Returns:
            The current stable (debounced) gesture.
        """
        if raw_gesture == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw_gesture
            self._candidate_count = 1

        if self._candidate_count >= self._required_frames:
            self._stable = self._candidate

        return self._stable

    def reset(self) -> None:
        """Reset to the no-gesture state, discarding any candidate."""
        self._candidate = GESTURE_NONE
        self._candidate_count = 0
        self._stable = GESTURE_NONE


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
        dwell_seconds: float = DWELL_TIME_SECONDS,
        on_action: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Initialize MediaPipe Hands and internal state helpers.

        Args:
            detection_confidence: Minimum confidence for initial hand
                detection.
            tracking_confidence: Minimum confidence for landmark
                tracking across frames.
            dwell_seconds: Seconds a gesture must be held before it is
                confirmed (FR-6.4). Exposed here so a settings/
                accessibility panel can configure it per user instead
                of it being fixed at the module default.
            on_action: Optional callback invoked as
                ``on_action(gesture, action)`` on the single frame a
                gesture is confirmed. This is the integration point
                for the rest of the Profile Information Hub (e.g. a
                UI layer that needs to actually move focus to the next
                field, submit the form, etc.) instead of the action
                only ever being printed to the console.
        """
        self._hands = _MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._dwell_timer = DwellTimer(dwell_seconds=dwell_seconds)
        self._sweep_detector = SideSweepDetector()
        self._cursor_smoother = CursorSmoother()
        self._stabilizer = GestureStabilizer()
        self._on_action = on_action

    def process(
        self,
        frame_bgr: np.ndarray,
        process_size: Optional[tuple[int, int]] = None,
    ) -> GestureResult:
        """Process a single BGR frame and produce a gesture result.

        Args:
            frame_bgr: The current webcam frame in BGR color order, as
                returned by ``cv2.VideoCapture.read``.
            process_size: Optional ``(process_width, process_height)``
                to downscale to before running MediaPipe inference.
                MediaPipe Hands is by far the most expensive operation
                in the capture loop, and its cost scales with input
                resolution; the landmark model itself works from a
                small internal crop regardless, so feeding it a full
                1280x720 frame buys nothing but wasted CPU/GPU cycles.
                Landmarks are normalized to [0, 1] by MediaPipe, so
                every pixel-scale output below (cursor position,
                bounding box) is computed from the *original* frame's
                width/height and needs no remapping - only the array
                handed to the detector itself is smaller. Pass
                ``None`` to process at native resolution.

        Returns:
            A fully populated :class:`GestureResult` for this frame.
        """
        height, width = frame_bgr.shape[:2]
        if process_size is not None and process_size != (width, height):
            detect_frame = cv2.resize(frame_bgr, process_size, interpolation=cv2.INTER_LINEAR)
        else:
            detect_frame = frame_bgr
        rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        detection = self._hands.process(rgb_frame)

        if not detection.multi_hand_landmarks:
            self._cursor_smoother.reset()
            self._sweep_detector.reset()
            self._stabilizer.reset()
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

        gesture_raw = classify_gesture(points, finger_states, sweep_detected)
        gesture = self._stabilizer.update(gesture_raw)
        confirmed, progress, action_triggered = self._dwell_timer.update(gesture)

        cursor_position: Optional[tuple[int, int]] = None
        if gesture == GESTURE_INDEX_POINT:
            raw_x = points[INDEX_TIP][0] * width
            raw_y = points[INDEX_TIP][1] * height
            cursor_position = self._cursor_smoother.update(raw_x, raw_y)
        else:
            self._cursor_smoother.reset()

        bounding_box = _compute_bounding_box(points, width, height)
        diagnostics = compute_diagnostics(points, finger_states, self._sweep_detector.current_spread())

        if action_triggered:
            action = GESTURE_ACTIONS[gesture]
            print(f"[ACTION TRIGGERED] {gesture} -> {action}")
            if self._on_action is not None:
                self._on_action(gesture, action)

        return GestureResult(
            gesture=gesture,
            action=GESTURE_ACTIONS[gesture],
            confirmed=confirmed,
            dwell_progress=progress,
            cursor_position=cursor_position,
            bounding_box=bounding_box,
            hand_landmarks=hand_landmarks,
            action_triggered=action_triggered,
            diagnostics=diagnostics,
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
    """Draw the virtual cursor as a high-contrast labeled circle.

    Renders a white core ring over a black outer ring (rather than a
    single flat color) so the cursor stays visible against any
    background in the live webcam feed, per the AAA-target contrast
    requirement in NFR-2 (>= 7:1) for input-accessibility elements.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        position: Pixel coordinates ``(x, y)`` of the cursor.
    """
    x, y = position
    cv2.circle(frame, (x, y), 14, COLOR_CURSOR_OUTLINE, 3)
    cv2.circle(frame, (x, y), 10, COLOR_CURSOR_CORE, -1)
    cv2.circle(frame, (x, y), 10, COLOR_CURSOR_OUTLINE, 2)
    cv2.putText(
        frame,
        "CURSOR",
        (x + 20, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_CURSOR_OUTLINE,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "CURSOR",
        (x + 20, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_CURSOR_CORE,
        1,
        cv2.LINE_AA,
    )


def draw_info_panel(frame: np.ndarray, fps: float, result: GestureResult) -> None:
    """Draw the top information panel summarizing recognition state.

    Displays FPS, current gesture, mapped action, confirmation status,
    dwell timer progress, and cursor position. FPS is rendered in a
    warning color when it drops below :data:`MIN_ACCEPTABLE_FPS`, the
    NFR-3 gesture-tracking frame-rate benchmark.

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

    fps_below_spec = fps > 0 and fps < MIN_ACCEPTABLE_FPS
    fps_color = COLOR_WARNING if fps_below_spec else COLOR_TEXT
    fps_label = f"FPS: {fps:.1f}" + (" (below NFR-3 target)" if fps_below_spec else "")

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (fps_label, fps_color),
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


def draw_diagnostics_panel(frame: np.ndarray, diagnostics: dict[str, float]) -> None:
    """Draw a bottom overlay of live raw values against their thresholds.

    For tuning the recognition constants (:data:`PINCH_DISTANCE_RATIO`,
    :data:`FINGER_EXTENSION_RATIO`, :data:`THUMB_UP_VERTICAL_RATIO`,
    :data:`SWEEP_MIN_DISPLACEMENT`) to a specific camera setup, hand
    size, and distance from the camera, instead of guessing values
    blind. Each line shows the current measured value next to the
    threshold it's compared against; a value crossing its threshold is
    highlighted. Intended to be toggled on/off (see ``main.py``), not
    shown by default.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        diagnostics: The ``diagnostics`` dict from a
            :class:`GestureResult`. If empty (no hand detected),
            nothing is drawn.
    """
    if not diagnostics:
        return

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    panel_top = frame_height - 140

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (frame_width, frame_height), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    def _fmt(label: str, value: float, threshold: float, lower_is_active: bool) -> tuple[str, tuple[int, int, int]]:
        active = (value < threshold) if lower_is_active else (value > threshold)
        color = COLOR_CONFIRMED if active else COLOR_TEXT
        return f"{label}: {value:.2f} (thr {threshold:.2f})", color

    rows = [
        _fmt("Pinch ratio", diagnostics["pinch_ratio"], diagnostics["pinch_ratio_threshold"], lower_is_active=True),
        _fmt(
            "Thumb-up margin",
            diagnostics["thumb_margin_ratio"],
            diagnostics["thumb_margin_threshold"],
            lower_is_active=False,
        ),
        _fmt(
            "Sweep spread",
            diagnostics["sweep_spread"],
            diagnostics["sweep_spread_threshold"],
            lower_is_active=False,
        ),
        _fmt("Ext: index", diagnostics["ext_index"], diagnostics["ext_threshold"], lower_is_active=False),
        _fmt("Ext: middle", diagnostics["ext_middle"], diagnostics["ext_threshold"], lower_is_active=False),
        _fmt("Ext: ring", diagnostics["ext_ring"], diagnostics["ext_threshold"], lower_is_active=False),
    ]

    col_x = [10, 10, 10, frame_width // 2, frame_width // 2, frame_width // 2]
    row_y = [panel_top + 25, panel_top + 55, panel_top + 85, panel_top + 25, panel_top + 55, panel_top + 85]

    header = "DIAGNOSTICS (press D to hide) - green = condition currently true"
    cv2.putText(frame, header, (10, panel_top + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    for (text, color), x, y in zip(rows, col_x, row_y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)