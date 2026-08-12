"""Static hand-shape fingerspelling for the Hand Gesture Recognition System's
Spelling Mode.

Lets a user compose text letter-by-letter using hand signals alone, per the
accessibility goals in SRS 3.6 / FR-6.1-FR-6.4 (dwell-time confirmation,
real-time landmark tracking). This module is intentionally separate from
``gesture.py``: gesture.py owns the five *action* gestures (THUMBS UP, OPEN
PALM, SIDE SWEEP, INDEX POINTING, PINCH) used for navigation; this module
owns the letter shapes and the three spelling *control* signals (space,
backspace, submit), used only while Spelling Mode is active. Keeping them
apart avoids two classifiers competing to explain the same landmark frame.

Coverage (v2 - full alphabet)
------------------------------
All 26 letters are now recognized, split across three detection strategies
depending on how much information a single static frame can actually give
us:

    Tier 1 - static, 2D, high confidence (17 letters):
        A, B, C, D, E, F, I, K, L, O, R, S, U, V, W, X, Y
        Distinguishable purely from finger-extension pattern + fingertip
        spread. No hand-rotation or depth (z) assumptions needed. These are
        the most reliable letters in the set.

    Tier 2 - static, 2D orientation-based (7 letters): G, H, P, Q, M, N, T
        G/H/P/Q differ from L/U/K by which way the extended finger(s) point
        (up vs. sideways vs. down) rather than by finger-extension pattern
        alone. Rather than requiring MediaPipe's z-axis (its least reliable
        output on a single RGB camera - see below), these are detected from
        the 2D angle of the extended finger in the image plane, which a
        normal webcam captures accurately as long as the user's forearm
        rotation is roughly visible to the camera (i.e. not aiming the
        camera straight down the arm).
        M/N/T depend on how deep the thumb tucks under the folded fingers,
        which *is* fundamentally a depth cue. These are approximated in 2D
        by measuring which folded finger's knuckle the thumb tip sits
        closest to. This is a best-effort heuristic, more sensitive to hand
        size/camera angle than Tier 1, and is the first place to look if a
        user reports trouble with a specific letter - see the tuning notes
        below.

    Tier 3 - motion-based (2 letters): J, Z
        These are *traced* letters in ASL, not static poses - a single
        frame structurally cannot capture them. :class:`_TraceTracker`
        watches the relevant fingertip's path over roughly one second and
        matches it against a simplified J-hook or Z-zigzag shape, the same
        way gesture.SideSweepDetector tracks palm motion over many frames
        for SIDE SWEEP. Because the motion itself takes about a second to
        complete, J/Z letters confirm as soon as the trace matches rather
        than going through the usual dwell timer on top of that.

Unclassified frames still return ``None`` from :func:`classify_letter`
rather than a low-confidence guess, so the caller can show "no match"
instead of a wrong letter.

Tuning: if a letter is unreliable for a given hand size/camera distance,
the thresholds below (``_SPREAD_*``, ``_CURL_*``, ``_TOUCH_RATIO``,
``_ANGLE_*``, ``_TUCK_*``, and the ``_Z_*``/``_J_*`` motion thresholds) are
the ones to adjust - consider extending gesture.draw_diagnostics_panel with
these values the same way pinch/extension are exposed there today, rather
than tuning blind. Tier 2/3 letters are inherently more sensitive to camera
angle and hand size than Tier 1; if a user consistently struggles with one
of them, that is a hardware/positioning issue to solve (camera height,
distance) before assuming the threshold is wrong.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import gesture as gu

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

LETTER_DWELL_SECONDS: float = 0.8
"""Shorter than gesture.DWELL_TIME_SECONDS (1.5s): spelling involves many
sequential confirmations, so a long per-letter hold would make composing
even short words impractical."""

LETTER_STABILITY_FRAMES: int = 4
"""Same debounce rationale as gesture.GESTURE_STABILITY_FRAMES - absorbs
single-frame landmark jitter so a letter doesn't flicker before dwell can
complete."""

_CURL_FULL: float = gu.FINGER_EXTENSION_RATIO
"""Extension ratio above which a finger counts as fully extended - reuses
gesture.py's existing threshold so the two modules agree on what
"extended" means."""

_CURL_HALF_MIN: float = 1.05
"""Extension ratio above which a finger counts as at least half-curled
(bent at the PIP joint, e.g. a "claw" or "hook"), used for C, O, and X."""

_TOUCH_RATIO: float = gu.PINCH_DISTANCE_RATIO
"""Fingertip-to-fingertip distance (relative to hand scale) below which two
tips count as "touching" - reuses gesture.py's pinch threshold."""

_SPREAD_TOGETHER: float = 0.35
"""Fingertip-pair spread (relative to hand scale) below which two extended
fingers count as held together (U) rather than apart (V)."""

_SPREAD_APART: float = 0.55
"""Fingertip-pair spread above which two extended fingers count as clearly
apart (V/K/W) rather than ambiguous."""

_ANGLE_UP_MAX: float = 35.0
"""A finger direction within this many degrees of straight up (0 deg)
counts as "pointing up" (used to tell D/L/U/K/V apart from their sideways
or downward Tier-2 counterparts G/Q/H/P)."""

_ANGLE_SIDEWAYS_MIN: float = 55.0
_ANGLE_SIDEWAYS_MAX: float = 125.0
"""A finger direction whose absolute angle from straight up falls in this
band counts as "pointing sideways" (roughly horizontal, left or right) -
the orientation used for G and H."""

_ANGLE_DOWN_MIN: float = 145.0
"""A finger direction whose absolute angle from straight up is at least
this large counts as "pointing down" - the orientation used for P and Q."""

_TUCK_MAX_DISTANCE: float = _TOUCH_RATIO * 1.6
"""Maximum normalized thumb-tip-to-knuckle distance for the thumb to count
as "tucked under" a folded finger (used for M/N/T). Above this distance the
thumb is considered resting to the side instead (A/S/E territory), not
tucked under a specific finger."""

# Motion-letter (J, Z) trace-matching thresholds. See _TraceTracker.
_MOTION_WINDOW_SECONDS: float = 1.2
_MOTION_HISTORY_SIZE: int = 24
_MOTION_MIN_SAMPLES: int = 8
_Z_MIN_SEGMENT_RATIO: float = 0.9
_J_MIN_DROP_RATIO: float = 0.9
_J_MIN_HOOK_RATIO: float = 0.6

# Control symbols returned alongside letters by classify_symbol().
CONTROL_SPACE: str = "SPACE"
CONTROL_BACKSPACE: str = "BACKSPACE"
CONTROL_SUBMIT: str = "SUBMIT"

SUPPORTED_LETTERS: frozenset[str] = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
"""All 26 letters are now recognized - see the module docstring's Tier 1/2/3
breakdown for how each one is detected and how reliable it is."""

# Short on-screen descriptions for draw_cheat_sheet(), kept in the same
# priority order as _LETTER_PREDICATES purely for readability here (this
# list has no effect on classification). Wording is deliberately terse -
# this is a reminder while the user's hands are already busy in front of
# the camera, not a tutorial; the fuller description of each shape lives
# in the corresponding _is_<letter> predicate's docstring/comments above.
CHEAT_SHEET_LETTERS: list[tuple[str, str]] = [
    ("A", "Fist, thumb out to the side"),
    ("B", "4 fingers up together, thumb in"),
    ("C", "Curved claw, thumb curved too"),
    ("D", "Index up, thumb touches middle"),
    ("E", "Fingers curled, thumb across front"),
    ("F", "Thumb+index touch, 3 fingers out"),
    ("G", "Index+thumb parallel, pointing sideways"),
    ("H", "Index+middle together, pointing sideways"),
    ("I", "Pinky up alone"),
    ("J", "Pinky-up, trace a J in the air"),
    ("K", "Index+middle apart, thumb between"),
    ("L", "Index+thumb at right angle"),
    ("M", "Fist, thumb tucked under ring finger"),
    ("N", "Fist, thumb tucked under middle finger"),
    ("O", "Fingertips + thumb form a circle"),
    ("P", "Like K, but pointing downward"),
    ("Q", "Like G, but pointing downward"),
    ("R", "Index+middle close together/crossed"),
    ("S", "Fist, thumb across the front"),
    ("T", "Fist, thumb tucked under index finger"),
    ("U", "Index+middle together, upright"),
    ("V", "Index+middle apart (peace sign)"),
    ("W", "Index+middle+ring apart"),
    ("X", "Index hooked, rest curled"),
    ("Y", "Thumb+pinky out (hang loose)"),
    ("Z", "Index-point, trace a Z in the air"),
]

CHEAT_SHEET_CONTROLS: list[tuple[str, str]] = [
    ("SPACE", "Open palm, all 5 fingers spread"),
    ("BACKSPACE", "Side sweep (open palm swept sideways)"),
    ("SUBMIT", "Thumbs up (moves to next field)"),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _extension_ratio(points: np.ndarray, tip_idx: int, mcp_idx: int) -> float:
    """Same measurement as gesture._is_finger_extended, returned as a
    continuous ratio instead of a threshold-crossing bool, so callers can
    distinguish "fully extended" from "half-curled" from "fully folded"."""
    wrist = points[gu.WRIST][:2]
    tip_dist = float(np.linalg.norm(points[tip_idx][:2] - wrist))
    mcp_dist = float(np.linalg.norm(points[mcp_idx][:2] - wrist))
    return (tip_dist / mcp_dist) if mcp_dist > 0 else 0.0


def _tip_distance(points: np.ndarray, a_idx: int, b_idx: int, scale: float) -> float:
    """Normalized 2D distance between two landmarks, relative to hand scale."""
    if scale <= 0:
        return 0.0
    return float(np.linalg.norm(points[a_idx][:2] - points[b_idx][:2])) / scale


def _direction_angle(points: np.ndarray, tip_idx: int, base_idx: int) -> float:
    """Angle, in degrees, of the vector from ``base_idx`` to ``tip_idx`` in
    the image plane, measured from straight up (0 deg). Positive angles
    rotate clockwise (90 deg = pointing right), negative counter-clockwise
    (-90 deg = pointing left), +-180 deg = pointing down.

    This is a purely 2D measurement - no z-axis involved - which is what
    lets G/H/P/Q be classified without relying on MediaPipe's less
    reliable depth output. It works as long as the camera can see the
    forearm's rotation (i.e. the user isn't pointing straight down the
    camera's optical axis), which is the normal webcam-facing-user setup
    this whole system assumes elsewhere (NFR-4).
    """
    vec = points[tip_idx][:2] - points[base_idx][:2]
    return math.degrees(math.atan2(float(vec[0]), -float(vec[1])))


def _points_up(angle: float) -> bool:
    return abs(angle) <= _ANGLE_UP_MAX


def _points_sideways(angle: float) -> bool:
    return _ANGLE_SIDEWAYS_MIN <= abs(angle) <= _ANGLE_SIDEWAYS_MAX


def _points_down(angle: float) -> bool:
    return abs(angle) >= _ANGLE_DOWN_MIN


def _closest_tuck_target(points: np.ndarray, scale: float) -> tuple[Optional[str], float]:
    """Find which folded finger's knuckle the thumb tip is tucked closest
    to, used to distinguish M/N/T (which differ only in how deep the thumb
    is tucked - see the module docstring's Tier 2 note on why this is a
    2D approximation of what is really a depth cue).

    Returns:
        A tuple of (finger name or None, normalized distance to it). The
        finger name is one of "index", "middle", "ring" - the PIP joints
        used as tuck-depth landmarks for T, N, and M respectively.
    """
    if scale <= 0:
        return None, float("inf")
    thumb_tip = points[gu.THUMB_TIP][:2]
    candidates = {
        "index": points[gu.INDEX_PIP][:2],
        "middle": points[gu.MIDDLE_PIP][:2],
        "ring": points[gu.RING_PIP][:2],
    }
    best_name, best_dist = None, float("inf")
    for name, joint in candidates.items():
        dist = float(np.linalg.norm(thumb_tip - joint)) / scale
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name, best_dist


@dataclass
class _LetterFeatures:
    """Precomputed per-frame measurements consumed by the letter predicates.

    Computing these once per frame (rather than inline in every predicate)
    keeps classify_letter's predicate list readable and avoids redundant
    landmark math when several letters check the same measurement.
    """

    ext: dict[str, float] = field(default_factory=dict)  # index/middle/ring/pinky extension ratios
    thumb_index_dist: float = 0.0
    thumb_middle_dist: float = 0.0
    index_middle_spread: float = 0.0
    middle_ring_spread: float = 0.0
    ring_pinky_spread: float = 0.0
    thumb_extended: bool = False
    index_angle: float = 0.0
    tuck_target: Optional[str] = None
    tuck_distance: float = float("inf")


def _compute_features(points: np.ndarray, finger_states: dict[str, bool]) -> _LetterFeatures:
    scale = gu._hand_scale(points)
    tuck_target, tuck_distance = _closest_tuck_target(points, scale)
    return _LetterFeatures(
        ext={
            "index": _extension_ratio(points, gu.INDEX_TIP, gu.INDEX_MCP),
            "middle": _extension_ratio(points, gu.MIDDLE_TIP, gu.MIDDLE_MCP),
            "ring": _extension_ratio(points, gu.RING_TIP, gu.RING_MCP),
            "pinky": _extension_ratio(points, gu.PINKY_TIP, gu.PINKY_MCP),
        },
        thumb_index_dist=_tip_distance(points, gu.THUMB_TIP, gu.INDEX_TIP, scale),
        thumb_middle_dist=_tip_distance(points, gu.THUMB_TIP, gu.MIDDLE_TIP, scale),
        index_middle_spread=_tip_distance(points, gu.INDEX_TIP, gu.MIDDLE_TIP, scale),
        middle_ring_spread=_tip_distance(points, gu.MIDDLE_TIP, gu.RING_TIP, scale),
        ring_pinky_spread=_tip_distance(points, gu.RING_TIP, gu.PINKY_TIP, scale),
        thumb_extended=finger_states["thumb"],
        index_angle=_direction_angle(points, gu.INDEX_TIP, gu.INDEX_MCP),
        tuck_target=tuck_target,
        tuck_distance=tuck_distance,
    )


# ---------------------------------------------------------------------------
# Letter predicates
#
# Each predicate takes (finger_states, feats) and returns bool. Ordered from
# most to least distinctive in classify_letter() so that a frame matching
# several loosely-defined shapes resolves to the most specific one.
# ---------------------------------------------------------------------------


def _folded(f: dict[str, bool], *names: str) -> bool:
    return not any(f[n] for n in names)


def _is_f(fs, feats):
    return feats.thumb_index_dist < _TOUCH_RATIO and fs["middle"] and fs["ring"] and fs["pinky"]


def _is_o(fs, feats):
    all_half_curled = all(_CURL_HALF_MIN < r < _CURL_FULL for r in feats.ext.values())
    return feats.thumb_index_dist < _TOUCH_RATIO * 1.5 and all_half_curled


def _is_x(fs, feats):
    index_hooked = _CURL_HALF_MIN < feats.ext["index"] < _CURL_FULL
    others_curled = feats.ext["middle"] <= _CURL_HALF_MIN and feats.ext["ring"] <= _CURL_HALF_MIN
    return index_hooked and others_curled and not fs["thumb"]


def _is_t(fs, feats):
    """T: fist with the thumb tip tucked under the index finger's PIP
    joint. See the module docstring - this is a 2D approximation of a
    depth cue, so it's the first to check (most specific / smallest tuck
    distance among M/N/T) to avoid being shadowed by the looser A/S/E
    checks that also require "all curled"."""
    all_curled = all(r <= _CURL_HALF_MIN for r in feats.ext.values())
    return all_curled and feats.tuck_target == "index" and feats.tuck_distance < _TUCK_MAX_DISTANCE


def _is_n(fs, feats):
    """N: fist with the thumb tip tucked between/under the index and
    middle fingers - approximated as "closest to the middle finger's PIP,
    while still genuinely tucked in" (see _is_t)."""
    all_curled = all(r <= _CURL_HALF_MIN for r in feats.ext.values())
    return all_curled and feats.tuck_target == "middle" and feats.tuck_distance < _TUCK_MAX_DISTANCE


def _is_m(fs, feats):
    """M: fist with the thumb tucked furthest in, under index+middle+ring
    - approximated as "closest to the ring finger's PIP, while still
    genuinely tucked in" (see _is_t)."""
    all_curled = all(r <= _CURL_HALF_MIN for r in feats.ext.values())
    return all_curled and feats.tuck_target == "ring" and feats.tuck_distance < _TUCK_MAX_DISTANCE


def _is_e(fs, feats):
    all_curled = all(r <= _CURL_HALF_MIN for r in feats.ext.values())
    thumb_crossing_front = feats.thumb_index_dist < _SPREAD_TOGETHER and feats.thumb_middle_dist < _SPREAD_TOGETHER
    return all_curled and thumb_crossing_front


def _is_a(fs, feats):
    all_curled = all(r <= _CURL_HALF_MIN for r in feats.ext.values())
    thumb_to_side = feats.thumb_index_dist >= _SPREAD_TOGETHER
    return all_curled and thumb_to_side


def _is_s(fs, feats):
    all_curled = all(r <= _CURL_HALF_MIN for r in feats.ext.values())
    thumb_covering_front = _TOUCH_RATIO <= feats.thumb_index_dist < _SPREAD_TOGETHER
    return all_curled and thumb_covering_front


def _is_c(fs, feats):
    all_half_curled = all(_CURL_HALF_MIN < r < _CURL_FULL for r in feats.ext.values())
    return all_half_curled and _TOUCH_RATIO * 1.5 <= feats.thumb_index_dist < _SPREAD_TOGETHER


def _is_b(fs, feats):
    fingers_up = fs["index"] and fs["middle"] and fs["ring"] and fs["pinky"]
    together = (
        feats.index_middle_spread < _SPREAD_TOGETHER
        and feats.middle_ring_spread < _SPREAD_TOGETHER
        and feats.ring_pinky_spread < _SPREAD_TOGETHER
    )
    return fingers_up and together and not fs["thumb"]


def _is_w(fs, feats):
    three_up = fs["index"] and fs["middle"] and fs["ring"] and not fs["pinky"]
    spread = feats.index_middle_spread >= _SPREAD_TOGETHER and feats.middle_ring_spread >= _SPREAD_TOGETHER
    return three_up and spread


def _is_g(fs, feats):
    """G: index and thumb extended roughly parallel to each other,
    pointing sideways (hand rotated ~90 deg from L) rather than upward.
    See the module docstring's Tier 2 note - 2D orientation, no z needed."""
    index_only = fs["index"] and _folded(fs, "middle", "ring", "pinky")
    return (
        index_only
        and fs["thumb"]
        and _points_sideways(feats.index_angle)
        and feats.thumb_index_dist < _SPREAD_APART
    )


def _is_q(fs, feats):
    """Q: same shape as G, rotated to point downward instead of sideways."""
    index_only = fs["index"] and _folded(fs, "middle", "ring", "pinky")
    return (
        index_only
        and fs["thumb"]
        and _points_down(feats.index_angle)
        and feats.thumb_index_dist < _SPREAD_APART
    )


def _is_h(fs, feats):
    """H: same shape as U (index+middle together), rotated to point
    sideways instead of upward."""
    two_up = fs["index"] and fs["middle"] and _folded(fs, "ring", "pinky")
    together = feats.index_middle_spread < _SPREAD_APART
    return two_up and together and _points_sideways(feats.index_angle)


def _is_p(fs, feats):
    """P: same shape as K (index+middle apart, thumb between), rotated to
    point downward instead of upward."""
    two_up = fs["index"] and fs["middle"] and _folded(fs, "ring", "pinky")
    spread_apart = feats.index_middle_spread >= _SPREAD_APART
    return two_up and fs["thumb"] and spread_apart and _points_down(feats.index_angle)


def _is_d(fs, feats):
    index_only = fs["index"] and _folded(fs, "middle", "ring", "pinky")
    thumb_touches_middle = feats.thumb_middle_dist < _SPREAD_TOGETHER
    return index_only and thumb_touches_middle and _points_up(feats.index_angle)


def _is_l(fs, feats):
    return (
        fs["index"]
        and fs["thumb"]
        and _folded(fs, "middle", "ring", "pinky")
        and feats.thumb_index_dist >= _SPREAD_TOGETHER
        and _points_up(feats.index_angle)
    )


def _is_k(fs, feats):
    two_up = fs["index"] and fs["middle"] and _folded(fs, "ring", "pinky")
    spread_apart = feats.index_middle_spread >= _SPREAD_APART
    return two_up and fs["thumb"] and spread_apart and _points_up(feats.index_angle)


def _is_v(fs, feats):
    two_up = fs["index"] and fs["middle"] and _folded(fs, "ring", "pinky")
    spread_apart = feats.index_middle_spread >= _SPREAD_APART
    return two_up and not fs["thumb"] and spread_apart


def _is_r(fs, feats):
    two_up = fs["index"] and fs["middle"] and _folded(fs, "ring", "pinky")
    crossed_or_close = feats.index_middle_spread < _SPREAD_TOGETHER * 0.6
    return two_up and not fs["thumb"] and crossed_or_close


def _is_u(fs, feats):
    two_up = fs["index"] and fs["middle"] and _folded(fs, "ring", "pinky")
    together = _SPREAD_TOGETHER * 0.6 <= feats.index_middle_spread < _SPREAD_APART
    return two_up and not fs["thumb"] and together and _points_up(feats.index_angle)


def _is_i(fs, feats):
    return fs["pinky"] and _folded(fs, "index", "middle", "ring") and not fs["thumb"]


def _is_y(fs, feats):
    return fs["thumb"] and fs["pinky"] and _folded(fs, "index", "middle", "ring")


# Ordered most- to least-specific. classify_letter returns the first match.
# T/N/M (tuck-based) are checked right after the other high-confidence
# curled-fist shapes (F/O/X) and before the looser E/A/S checks, since a
# genuine tuck-distance match is more specific than "thumb roughly
# somewhere in front of a fist". G/Q/H/P are checked alongside their
# upright counterparts (L, K, U) using the same finger-extension pattern,
# split apart only by _points_* on the index angle.
_LETTER_PREDICATES: list[tuple[str, "Callable"]] = [  # noqa: F821 (Callable resolved at runtime via typing import below)
    ("F", _is_f),
    ("O", _is_o),
    ("X", _is_x),
    ("T", _is_t),
    ("N", _is_n),
    ("M", _is_m),
    ("E", _is_e),
    ("Y", _is_y),
    ("W", _is_w),
    ("G", _is_g),
    ("Q", _is_q),
    ("H", _is_h),
    ("P", _is_p),
    ("K", _is_k),
    ("V", _is_v),
    ("R", _is_r),
    ("U", _is_u),
    ("D", _is_d),
    ("L", _is_l),
    ("I", _is_i),
    ("B", _is_b),
    ("C", _is_c),
    ("S", _is_s),
    ("A", _is_a),
]


def classify_letter(points: np.ndarray, finger_states: dict[str, bool]) -> Optional[str]:
    """Classify the current static hand pose as one of the Tier 1/2
    supported letters (everything except the motion letters J and Z,
    which need :class:`_TraceTracker` instead - see :meth:`SpellingSession.
    process`).

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks, as
            produced by gesture._landmarks_to_array.
        finger_states: Precomputed finger extension states, as produced by
            gesture.compute_finger_states.

    Returns:
        The uppercase letter, or ``None`` if the current pose doesn't
        confidently match a supported static letter.
    """
    feats = _compute_features(points, finger_states)
    for letter, predicate in _LETTER_PREDICATES:
        if predicate(finger_states, feats):
            return letter
    return None


def classify_symbol(points: np.ndarray, finger_states: dict[str, bool], sweep_detected: bool) -> Optional[str]:
    """Classify the frame as a static letter or a spelling-mode control
    symbol. Motion letters (J, Z) are layered on top of this by
    :class:`SpellingSession`, since they need multi-frame trajectory data
    this function doesn't have access to.

    Control symbols take priority over letters for the same reason
    gesture.classify_gesture prioritizes SIDE SWEEP: a sustained,
    multi-frame motion signal (backspace) is stronger evidence than a
    single frame's static shape, and a shape that happens to coincide with
    SPACE/SUBMIT should not be silently reinterpreted as a letter.

    Args:
        points: Array of shape ``(21, 3)`` of normalized landmarks.
        finger_states: Precomputed finger extension states.
        sweep_detected: Output of a gesture.SideSweepDetector tracking this
            spelling session, used as the BACKSPACE trigger.

    Returns:
        CONTROL_BACKSPACE, CONTROL_SPACE, CONTROL_SUBMIT, a letter from
        SUPPORTED_LETTERS (excluding J/Z), or ``None``.
    """
    if sweep_detected:
        return CONTROL_BACKSPACE
    # SPACE: open palm, all five fingers extended and spread - distinct
    # from letter B, which requires the thumb folded and fingers together.
    if all(finger_states.values()):
        return CONTROL_SPACE
    # SUBMIT: reuse gesture.py's exact thumbs-up predicate so "submit" means
    # the same physical gesture in both Navigation and Spelling Mode.
    if gu._is_thumbs_up(points, finger_states):
        return CONTROL_SUBMIT
    return classify_letter(points, finger_states)


# ---------------------------------------------------------------------------
# Motion-letter tracking (J, Z)
# ---------------------------------------------------------------------------


class _TraceTracker:
    """Tracks a short fingertip trajectory to detect the two ASL letters
    that are traced motions rather than static poses (J and Z).

    This deliberately mirrors gesture.SideSweepDetector's approach (a
    rolling position history checked against a displacement threshold)
    rather than a full trajectory-matching algorithm (e.g. DTW against a
    reference template) - simple, cheap, and consistent with how the rest
    of this codebase favors direct geometric heuristics over ML matching.
    It is a *simplified* shape match (rightward - diagonal - rightward for
    Z; drop - hook for J), not a faithful reproduction of ASL motion
    capture, so it's more forgiving of an imperfect trace than it is
    precise about rejecting a sloppy one. If this proves too permissive in
    practice, tighten ``_Z_MIN_SEGMENT_RATIO`` / ``_J_MIN_DROP_RATIO`` /
    ``_J_MIN_HOOK_RATIO`` first.

    The base handshape gates which letter is being attempted (pinky-only
    "I" shape primes J-detection, index-only "point" shape primes
    Z-detection); switching shape or losing tracking mid-trace clears the
    history so a stray earlier motion can't bleed into a new attempt.
    """

    def __init__(
        self,
        history_size: int = _MOTION_HISTORY_SIZE,
        window_seconds: float = _MOTION_WINDOW_SECONDS,
    ) -> None:
        self._history: deque[tuple[float, float, float]] = deque(maxlen=history_size)
        self._window_seconds = window_seconds
        self._shape: Optional[str] = None

    def update(self, points: np.ndarray, finger_states: dict[str, bool]) -> Optional[str]:
        """Feed one frame's landmarks and return "J", "Z", or ``None``.

        Args:
            points: Array of shape ``(21, 3)`` of normalized landmarks.
            finger_states: Precomputed finger extension states.
        """
        is_point_shape = finger_states["index"] and _folded(finger_states, "middle", "ring", "pinky")
        is_i_shape = (
            finger_states["pinky"]
            and _folded(finger_states, "index", "middle", "ring")
            and not finger_states["thumb"]
        )

        if is_point_shape:
            shape, tip_idx = "Z", gu.INDEX_TIP
        elif is_i_shape:
            shape, tip_idx = "J", gu.PINKY_TIP
        else:
            self._history.clear()
            self._shape = None
            return None

        if shape != self._shape:
            self._history.clear()
            self._shape = shape

        scale = gu._hand_scale(points)
        if scale <= 0:
            return None

        now = time.time()
        tip = points[tip_idx][:2]
        self._history.append((float(tip[0]) / scale, float(tip[1]) / scale, now))
        while self._history and now - self._history[0][2] > self._window_seconds:
            self._history.popleft()

        if len(self._history) < _MOTION_MIN_SAMPLES:
            return None

        if shape == "Z" and self._matches_z():
            self._history.clear()
            return "Z"
        if shape == "J" and self._matches_j():
            self._history.clear()
            return "J"
        return None

    def reset(self) -> None:
        """Clear tracked history and any in-progress shape gating."""
        self._history.clear()
        self._shape = None

    def _matches_z(self) -> bool:
        """Simplified Z shape: rightward stroke, then a down-and-back
        diagonal, then a final rightward stroke (or the full mirror image,
        since a user may trace it starting from either side)."""
        pts = list(self._history)
        third = len(pts) // 3
        if third < 2:
            return False
        seg1, seg2, seg3 = pts[:third], pts[third : 2 * third], pts[2 * third :]
        dx1 = seg1[-1][0] - seg1[0][0]
        dx2 = seg2[-1][0] - seg2[0][0]
        dy2 = seg2[-1][1] - seg2[0][1]
        dx3 = seg3[-1][0] - seg3[0][0]
        min_disp = _Z_MIN_SEGMENT_RATIO
        rightward = dx1 > min_disp and dx2 < -min_disp * 0.4 and dy2 > min_disp * 0.4 and dx3 > min_disp
        leftward = dx1 < -min_disp and dx2 > min_disp * 0.4 and dy2 > min_disp * 0.4 and dx3 < -min_disp
        return rightward or leftward

    def _matches_j(self) -> bool:
        """Simplified J shape: a downward drop, then a hook to either side
        (traced starting from the top, as in ASL)."""
        pts = list(self._history)
        half = len(pts) // 2
        if half < 3:
            return False
        first_half, second_half = pts[:half], pts[half:]
        dy1 = first_half[-1][1] - first_half[0][1]  # positive = moved down (image y grows downward)
        dx2 = second_half[-1][0] - second_half[0][0]
        return dy1 > _J_MIN_DROP_RATIO and abs(dx2) > _J_MIN_HOOK_RATIO


# ---------------------------------------------------------------------------
# Text buffer
# ---------------------------------------------------------------------------


class SpellingBuffer:
    """Accumulates confirmed letters/controls into a text string."""

    def __init__(self) -> None:
        self._text: str = ""

    def add_letter(self, letter: str) -> None:
        self._text += letter

    def add_space(self) -> None:
        if self._text and not self._text.endswith(" "):
            self._text += " "

    def backspace(self) -> None:
        self._text = self._text[:-1]

    def clear(self) -> None:
        self._text = ""

    def load(self, text: str) -> None:
        """Replace the buffer outright, e.g. to restore an
        already-typed value when the caller switches which Profile
        Information Hub field is active (FR-6.9/FR-6.12)."""
        self._text = text

    @property
    def text(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Spelling session (parallels gesture.HandGestureProcessor)
# ---------------------------------------------------------------------------

_HINT_IDLE_SECONDS: float = 2.5
"""Seconds a hand can be visible with no matching letter/control before
SpellingResult.hint starts surfacing a nudge toward the cheat sheet. This
is a pure discoverability aid (users learning a 26-letter alphabet will
often sit in an unrecognized pose while thinking) - it never affects
classification itself."""

_HINT_TEXT: str = "Not sure which letter? Press H for the letter guide."


@dataclass
class SpellingResult:
    """Container for a single frame's fingerspelling output."""

    symbol: Optional[str]
    confirmed: bool
    dwell_progress: float
    buffer_text: str
    bounding_box: Optional[tuple[int, int, int, int]]
    hand_landmarks: Optional[object]
    submitted: bool = False
    hint: Optional[str] = None


class SpellingSession:
    """Coordinates letter/control classification, debouncing, dwell
    confirmation, and buffer updates for Spelling Mode.

    Deliberately reuses gesture.GestureStabilizer and gesture.DwellTimer
    rather than reimplementing debounce/dwell logic - both are already
    generic over the string label they're tracking (a gesture name here is
    just a letter or control name instead), and letters need exactly the
    same "hold steady before it counts" treatment gestures do (FR-6.4).
    The two motion letters (J, Z) are the exception: :class:`_TraceTracker`
    already requires roughly a second of consistent motion to match, so
    they confirm immediately rather than additionally going through the
    stabilizer/dwell timer on top of that (see :meth:`process`).
    """

    def __init__(
        self,
        dwell_seconds: float = LETTER_DWELL_SECONDS,
        on_symbol: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize the session.

        Args:
            dwell_seconds: Seconds a letter/control must be held before
                it's confirmed.
            on_symbol: Optional callback invoked as ``on_symbol(symbol)``
                on the single frame each letter or control (CONTROL_SPACE,
                CONTROL_BACKSPACE - CONTROL_SUBMIT is reported separately
                via SpellingResult.submitted, not through this callback)
                is confirmed. This is the integration point for actually
                filling a real form field - e.g. forwarding each
                confirmed letter as a synthetic keystroke to whatever
                control currently has OS keyboard focus - instead of the
                composed text only ever living in this session's internal
                buffer_text. Mirrors gesture.HandGestureProcessor's
                on_action callback.
        """
        self._stabilizer = gu.GestureStabilizer(required_frames=LETTER_STABILITY_FRAMES)
        self._dwell_timer = gu.DwellTimer(dwell_seconds=dwell_seconds)
        self._sweep_detector = gu.SideSweepDetector()
        self._motion_tracker = _TraceTracker()
        self._on_symbol = on_symbol
        self._no_match_since: Optional[float] = None
        self.buffer = SpellingBuffer()

    def process(self, points: Optional[np.ndarray], finger_states: Optional[dict[str, bool]],
                bounding_box: Optional[tuple[int, int, int, int]], hand_landmarks: Optional[object]) -> SpellingResult:
        """Advance the spelling state machine with one frame's landmarks.

        Args:
            points: Array of shape ``(21, 3)`` of normalized landmarks, or
                ``None`` if no hand was detected this frame.
            finger_states: Precomputed finger extension states, or ``None``.
            bounding_box: Pixel bounding box of the hand, or ``None``.
            hand_landmarks: Raw MediaPipe landmarks for drawing, or ``None``.

        Returns:
            A populated :class:`SpellingResult`.
        """
        if points is None or finger_states is None:
            self._stabilizer.reset()
            self._sweep_detector.reset()
            self._motion_tracker.reset()
            self._dwell_timer.update(gu.GESTURE_NONE)
            self._no_match_since = None
            return SpellingResult(
                symbol=None, confirmed=False, dwell_progress=0.0,
                buffer_text=self.buffer.text, bounding_box=None, hand_landmarks=None,
            )

        # Motion letters (J, Z) are checked first and, when the trace
        # matches, confirm immediately - the ~1s of tracked motion already
        # serves as this letter's own "hold", so running it back through
        # the static stabilizer/dwell timer on top would just add latency
        # for no accuracy benefit. Any in-progress static letter/control
        # dwell is cleared so it can't fire on the same frame.
        motion_letter = self._motion_tracker.update(points, finger_states)
        if motion_letter is not None:
            self.buffer.add_letter(motion_letter)
            if self._on_symbol is not None:
                self._on_symbol(motion_letter)
            self._stabilizer.reset()
            self._dwell_timer.update(gu.GESTURE_NONE)
            self._no_match_since = None
            return SpellingResult(
                symbol=motion_letter,
                confirmed=True,
                dwell_progress=1.0,
                buffer_text=self.buffer.text,
                bounding_box=bounding_box,
                hand_landmarks=hand_landmarks,
            )

        palm_x = float(np.mean([points[i][0] for i in
                                 (gu.INDEX_MCP, gu.MIDDLE_MCP, gu.RING_MCP, gu.PINKY_MCP, gu.WRIST)]))
        palm_open = all(finger_states.values())
        sweep_detected = self._sweep_detector.update(palm_x, palm_open)

        raw_symbol = classify_symbol(points, finger_states, sweep_detected) or gu.GESTURE_NONE
        symbol = self._stabilizer.update(raw_symbol)
        confirmed, progress, just_triggered = self._dwell_timer.update(symbol)

        now = time.time()
        if raw_symbol == gu.GESTURE_NONE:
            if self._no_match_since is None:
                self._no_match_since = now
        else:
            self._no_match_since = None
        hint = (
            _HINT_TEXT
            if self._no_match_since is not None and (now - self._no_match_since) > _HINT_IDLE_SECONDS
            else None
        )

        submitted = False
        if just_triggered and symbol != gu.GESTURE_NONE:
            if symbol == CONTROL_SPACE:
                self.buffer.add_space()
                if self._on_symbol is not None:
                    self._on_symbol(CONTROL_SPACE)
            elif symbol == CONTROL_BACKSPACE:
                self.buffer.backspace()
                if self._on_symbol is not None:
                    self._on_symbol(CONTROL_BACKSPACE)
            elif symbol == CONTROL_SUBMIT:
                submitted = True
            else:
                self.buffer.add_letter(symbol)
                if self._on_symbol is not None:
                    self._on_symbol(symbol)

        return SpellingResult(
            symbol=None if symbol == gu.GESTURE_NONE else symbol,
            confirmed=confirmed,
            dwell_progress=progress,
            buffer_text=self.buffer.text,
            bounding_box=bounding_box,
            hand_landmarks=hand_landmarks,
            submitted=submitted,
            hint=hint,
        )

    def reset(self) -> None:
        self._stabilizer.reset()
        self._sweep_detector.reset()
        self._motion_tracker.reset()
        self._no_match_since = None
        self.buffer.clear()

    def load_text(self, text: str) -> None:
        """Preload the buffer with a stored field value, e.g. when
        navigating to a field that was already spelled out earlier
        (FR-6.9/FR-6.12). Only the buffer is touched - stabilizer/dwell
        state is left alone so this can't be mistaken for a confirmed
        letter on the next frame."""
        self.buffer.load(text)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_spelling_panel(
    frame: np.ndarray,
    result: SpellingResult,
    typing_enabled: bool = False,
    top_offset: int = 0,
    field_label: Optional[str] = None,
) -> None:
    """Draw the spelling-mode overlay: composed text, current symbol, dwell
    progress, and whether confirmed letters are being typed into an
    externally focused field. Mirrors gesture.draw_info_panel's layout
    conventions (same panel styling / color usage) so Navigation and
    Spelling Mode feel like one consistent app rather than two
    bolted-together UIs.

    Args:
        frame: The BGR frame to draw onto, modified in place.
        result: The spelling result for the current frame.
        typing_enabled: Whether confirmed symbols are currently being
            forwarded to the OS as keystrokes (see main.TypingBridge).
            Shown explicitly so the user always knows whether spelling
            "ABC" is about to type into their focused field or only
            preview here - silently doing one or the other would be
            surprising either way.
        top_offset: Pixels to push this whole panel down by, so it sits
            below the shared top bar (which now carries the active
            Profile Information Hub field name) instead of the two
            overlapping (FR-6.5-FR-6.12).
        field_label: Name of the Profile Information Hub field this
            spelled text is being composed for, shown next to the
            composed text so it's unambiguous which field Thumbs Up
            will submit.
    """
    import cv2

    panel_height = 130
    frame_width = frame.shape[1]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top_offset), (frame_width, top_offset + panel_height), gu.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    symbol_text = result.symbol if result.symbol else "-"
    status_color = gu.COLOR_NONE if result.symbol is None else (gu.COLOR_CONFIRMED if result.confirmed else gu.COLOR_DETECTING)

    cv2.putText(frame, f"SPELLING MODE  |  Current: {symbol_text}", (10, top_offset + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    typing_color = gu.COLOR_CONFIRMED if typing_enabled else gu.COLOR_WARNING
    typing_label = "TYPING INTO FOCUSED FIELD" if typing_enabled else "PREVIEW ONLY (press T to type live)"
    cv2.putText(frame, typing_label, (frame_width - 420, top_offset + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, typing_color, 2, cv2.LINE_AA)

    field_prefix = f"[{field_label}] " if field_label else ""
    display_text = result.buffer_text if result.buffer_text else "(empty - spell a letter to begin)"
    cv2.putText(frame, f"{field_prefix}Text: {display_text}", (10, top_offset + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, gu.COLOR_TEXT, 2, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 10, top_offset + 75, frame_width - 20, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    filled_w = int(bar_w * result.dwell_progress)
    if filled_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), status_color, -1)

    # The idle hint takes over the bottom help line when active, rather
    # than fighting for space alongside the normal controls reminder - a
    # user who's stuck benefits more from the nudge than from re-reading
    # controls they already know.
    if result.hint:
        cv2.putText(frame, result.hint, (10, top_offset + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, gu.COLOR_WARNING, 1, cv2.LINE_AA)
    else:
        cv2.putText(
            frame,
            "Open palm=space  |  Side sweep=backspace  |  Thumbs up=submit field & move to next",
            (10, top_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gu.COLOR_TEXT, 1, cv2.LINE_AA,
        )
    cv2.putText(frame, "T=toggle live typing  |  H=letter guide  |  Tab=return to menu",
                (10, top_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gu.COLOR_TEXT, 1, cv2.LINE_AA)


def draw_cheat_sheet(frame: np.ndarray) -> None:
    """Draw an on-screen reference of every supported letter shape and
    control signal, so the user isn't tabbing away to a printed chart
    while their hands are occupied in front of the camera.

    Intended to be toggled on/off with a dedicated key (see main.py's
    CHEAT_SHEET_KEY), not shown by default, the same way
    gesture.draw_diagnostics_panel is opt-in rather than always-on.

    Args:
        frame: The BGR frame to draw onto, modified in place.
    """
    import cv2

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    panel_top = 135  # Start just below draw_spelling_panel's 130px panel.
    panel_bottom = frame_height - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (frame_width, panel_bottom), gu.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "LETTER GUIDE (press H to hide) - all 26 letters (J/Z are traced, not held)",
                (10, panel_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gu.COLOR_TEXT, 1, cv2.LINE_AA)

    # Four columns of letters so all 26 fit without scrolling.
    col_count = 4
    rows_per_col = -(-len(CHEAT_SHEET_LETTERS) // col_count)  # ceil division
    col_width = frame_width // col_count
    row_height = 22
    row_start_y = panel_top + 48

    for i, (letter, desc) in enumerate(CHEAT_SHEET_LETTERS):
        col = i // rows_per_col
        row = i % rows_per_col
        x = 10 + col * col_width
        y = row_start_y + row * row_height
        if y > panel_bottom - 70:
            continue
        cv2.putText(frame, f"{letter}: {desc}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, gu.COLOR_TEXT, 1, cv2.LINE_AA)

    controls_y = panel_bottom - 55
    cv2.putText(frame, "CONTROLS", (10, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, gu.COLOR_DETECTING, 1, cv2.LINE_AA)
    for i, (name, desc) in enumerate(CHEAT_SHEET_CONTROLS):
        x = 10 + i * (frame_width // len(CHEAT_SHEET_CONTROLS))
        cv2.putText(frame, f"{name}: {desc}", (x, controls_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, gu.COLOR_TEXT, 1, cv2.LINE_AA)