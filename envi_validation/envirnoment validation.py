"""
FR-1.3: Environmental Validation
---------------------------------
Real-time verification of:
  1. Face presence
  2. Lighting sufficiency
  3. Frame positioning
before confirming biometric registration.

Requires: opencv-python, numpy
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    face_presence: CheckResult
    lighting: CheckResult
    positioning: CheckResult

    @property
    def passed(self) -> bool:
        return (
            self.face_presence.passed
            and self.lighting.passed
            and self.positioning.passed
        )

    def failure_reasons(self) -> List[str]:
        reasons = []
        for check in (self.face_presence, self.lighting, self.positioning):
            if not check.passed:
                reasons.append(check.message)
        return reasons


# ---------------------------------------------------------------------------
# FR-1.3.1 — Face Presence Detection
# ---------------------------------------------------------------------------

class FacePresenceValidator:
    """Detects whether exactly one face is present in the frame."""

    def __init__(self, min_face_ratio: float = 0.05):
        # Haar cascade ships with opencv-python, no separate download needed
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load face cascade classifier")
        self.min_face_ratio = min_face_ratio  # face area / frame area minimum

    def validate(self, frame: np.ndarray) -> Tuple[CheckResult, Optional[Tuple[int, int, int, int]]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
        )

        frame_area = frame.shape[0] * frame.shape[1]
        # Filter out spuriously small detections
        faces = [
            f for f in faces
            if (f[2] * f[3]) / frame_area >= self.min_face_ratio
        ]

        if len(faces) == 0:
            return CheckResult(False, "No face detected. Please face the camera."), None

        if len(faces) > 1:
            return CheckResult(
                False,
                "Multiple faces detected. Only one person should be in frame.",
                {"count": len(faces)},
            ), None

        face_box = tuple(faces[0])  # (x, y, w, h)
        return CheckResult(True, "Face detected.", {"box": face_box}), face_box


# ---------------------------------------------------------------------------
# FR-1.3.2 — Lighting Sufficiency
# ---------------------------------------------------------------------------

class LightingValidator:
    """Checks brightness, contrast, and lighting uniformity over the face ROI."""

    def __init__(
        self,
        min_brightness: float = 60.0,
        max_brightness: float = 200.0,
        min_contrast: float = 20.0,
        max_quadrant_imbalance: float = 60.0,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_quadrant_imbalance = max_quadrant_imbalance

    def validate(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> CheckResult:
        x, y, w, h = face_box
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return CheckResult(False, "Unable to sample lighting on face region.")

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray_roi))
        contrast = float(np.std(gray_roi))

        if mean_brightness < self.min_brightness:
            return CheckResult(
                False,
                "Lighting too dim. Move to a brighter area.",
                {"brightness": mean_brightness},
            )

        if mean_brightness > self.max_brightness:
            return CheckResult(
                False,
                "Lighting too bright / glare detected. Reduce direct light or backlight.",
                {"brightness": mean_brightness},
            )

        if contrast < self.min_contrast:
            return CheckResult(
                False,
                "Low contrast / flat lighting. Facial features are not clearly visible.",
                {"contrast": contrast},
            )

        # Quadrant balance check: catches harsh side-lighting or shadows
        h2, w2 = gray_roi.shape[0] // 2, gray_roi.shape[1] // 2
        quadrants = [
            gray_roi[0:h2, 0:w2],
            gray_roi[0:h2, w2:],
            gray_roi[h2:, 0:w2],
            gray_roi[h2:, w2:],
        ]
        quad_means = [float(np.mean(q)) for q in quadrants if q.size > 0]
        imbalance = max(quad_means) - min(quad_means) if quad_means else 0.0

        if imbalance > self.max_quadrant_imbalance:
            return CheckResult(
                False,
                "Uneven lighting across face. Avoid strong side lighting or shadows.",
                {"imbalance": imbalance},
            )

        return CheckResult(
            True,
            "Lighting sufficient.",
            {"brightness": mean_brightness, "contrast": contrast, "imbalance": imbalance},
        )


# ---------------------------------------------------------------------------
# FR-1.3.3 — Frame Positioning
# ---------------------------------------------------------------------------

class FramePositioningValidator:
    """Checks that the face is centered, correctly sized, and fully within frame."""

    def __init__(
        self,
        center_tolerance: float = 0.15,   # fraction of frame dimension
        min_face_ratio: float = 0.12,     # face area / frame area
        max_face_ratio: float = 0.55,
        edge_margin_px: int = 5,
    ):
        self.center_tolerance = center_tolerance
        self.min_face_ratio = min_face_ratio
        self.max_face_ratio = max_face_ratio
        self.edge_margin_px = edge_margin_px

    def validate(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> CheckResult:
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = face_box

        # 1. Edge containment — face must not be cut off by frame border
        if (
            x <= self.edge_margin_px
            or y <= self.edge_margin_px
            or (x + w) >= (frame_w - self.edge_margin_px)
            or (y + h) >= (frame_h - self.edge_margin_px)
        ):
            return CheckResult(
                False,
                "Face is too close to the frame edge. Center yourself in view.",
                {"box": face_box},
            )

        # 2. Centering
        face_cx, face_cy = x + w / 2, y + h / 2
        frame_cx, frame_cy = frame_w / 2, frame_h / 2

        dx = abs(face_cx - frame_cx) / frame_w
        dy = abs(face_cy - frame_cy) / frame_h

        if dx > self.center_tolerance or dy > self.center_tolerance:
            return CheckResult(
                False,
                "Face not centered. Please align with the on-screen guide.",
                {"dx": dx, "dy": dy},
            )

        # 3. Distance / scale via face-to-frame area ratio
        face_ratio = (w * h) / (frame_w * frame_h)

        if face_ratio < self.min_face_ratio:
            return CheckResult(
                False,
                "Face too far from camera. Move closer.",
                {"face_ratio": face_ratio},
            )

        if face_ratio > self.max_face_ratio:
            return CheckResult(
                False,
                "Face too close to camera. Move back slightly.",
                {"face_ratio": face_ratio},
            )

        return CheckResult(
            True,
            "Positioning correct.",
            {"dx": dx, "dy": dy, "face_ratio": face_ratio},
        )


# ---------------------------------------------------------------------------
# Orchestrator — combines all three checks
# ---------------------------------------------------------------------------

class EnvironmentalValidator:
    def __init__(self):
        self.face_validator = FacePresenceValidator()
        self.lighting_validator = LightingValidator()
        self.positioning_validator = FramePositioningValidator()

    def validate_frame(self, frame: np.ndarray) -> ValidationResult:
        face_result, face_box = self.face_validator.validate(frame)

        if not face_result.passed:
            skipped = CheckResult(False, "Skipped — no valid face.")
            return ValidationResult(face_result, skipped, skipped)

        lighting_result = self.lighting_validator.validate(frame, face_box)
        positioning_result = self.positioning_validator.validate(frame, face_box)

        return ValidationResult(face_result, lighting_result, positioning_result)


# ---------------------------------------------------------------------------
# Overlay styling
# ---------------------------------------------------------------------------

_INK = (30, 30, 30)          # near-black text, BGR
_MUTED = (110, 110, 110)     # secondary text
_GREEN = (76, 175, 80)       # pass
_RED = (60, 60, 220)         # fail
_PANEL_BG = (255, 255, 255)  # white panel
_TRACK_BG = (224, 224, 224)  # progress bar track
_FONT = cv2.FONT_HERSHEY_DUPLEX  # cleaner / more legible than SIMPLEX


def _rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """Draws a filled (or outlined) rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)


def draw_status_panel(frame, result: ValidationResult, consecutive_passes: int,
                       required_passes: int) -> np.ndarray:
    """Renders a clean, card-style status panel in the top-left corner
    instead of raw cv2.putText lines: a translucent white card, a colored
    dot + label + detail per check, and a progress bar toward confirmation.
    """
    h, w = frame.shape[:2]
    margin = 14
    panel_w = min(360, w - 2 * margin)
    panel_h = 168
    x0, y0 = margin, margin

    # translucent white card behind the text for readability over any background
    card = frame.copy()
    _rounded_rect(card, (x0, y0), (x0 + panel_w, y0 + panel_h), _PANEL_BG, radius=14)
    frame = cv2.addWeighted(card, 0.88, frame, 0.12, 0)
    _rounded_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (210, 210, 210), radius=14, thickness=1)

    checks = [
        ("Face", result.face_presence),
        ("Lighting", result.lighting),
        ("Position", result.positioning),
    ]

    pad_x, pad_y = 18, 30
    line_gap = 34
    for i, (label, check) in enumerate(checks):
        cy = y0 + pad_y + i * line_gap
        dot_color = _GREEN if check.passed else _RED
        cv2.circle(frame, (x0 + pad_x, cy - 5), 6, dot_color, -1)

        label_text = f"{label}"
        cv2.putText(frame, label_text, (x0 + pad_x + 18, cy), _FONT, 0.55, _INK, 1, cv2.LINE_AA)

        detail = "OK" if check.passed else check.message
        (label_w, _), _ = cv2.getTextSize(label_text, _FONT, 0.55, 1)
        cv2.putText(frame, detail, (x0 + pad_x + 30 + label_w, cy), _FONT, 0.5, _MUTED, 1, cv2.LINE_AA)

    # progress bar toward stable confirmation
    bar_y = y0 + pad_y + len(checks) * line_gap + 6
    bar_x0 = x0 + pad_x
    bar_w = panel_w - 2 * pad_x
    bar_h = 8
    progress = min(consecutive_passes / required_passes, 1.0)

    _rounded_rect(frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + bar_h), _TRACK_BG, radius=4)
    if progress > 0:
        fill_color = _GREEN if progress >= 1.0 else (66, 133, 244)
        _rounded_rect(frame, (bar_x0, bar_y), (bar_x0 + int(bar_w * progress), bar_y + bar_h),
                      fill_color, radius=4)

    status_text = "Ready — hold still" if result.passed else "Adjust and hold still"
    cv2.putText(frame, status_text, (bar_x0, bar_y + 28), _FONT, 0.5, _INK, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Real-time registration gate demo
# ---------------------------------------------------------------------------

def run_registration_gate(required_consecutive_passes: int = 15):
    """
    Opens the webcam and only confirms registration once all three checks
    pass for `required_consecutive_passes` consecutive frames (~0.5s at
    30fps), preventing a single lucky frame from triggering registration.
    """
    validator = EnvironmentalValidator()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    consecutive_passes = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = validator.validate_frame(frame)

            if result.passed:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            frame = draw_status_panel(frame, result, consecutive_passes, required_consecutive_passes)

            if consecutive_passes >= required_consecutive_passes:
                cv2.imshow("Environmental Validation (FR-1.3)", frame)
                cv2.waitKey(1500)
                print("Registration confirmed.")
                break

            cv2.imshow("Environmental Validation (FR-1.3)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_registration_gate()