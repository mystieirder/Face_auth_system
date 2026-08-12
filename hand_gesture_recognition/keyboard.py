"""Professional transparent virtual keyboard controlled by index-pointing."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import (
    KEY_DWELL_SECONDS, KEY_FILL_ALPHA, KEY_GAP, KEY_H, KEYBOARD_ALPHA,
    KEYBOARD_MARGIN, KEYBOARD_ROWS, KEYBOARD_BOTTOM_MARGIN,
    RELEASE_AFTER_SELECT_SECONDS, WHITE, MUTED,
    PANEL, PANEL_2, BORDER, ACCENT, TOP_BAR_HEIGHT,
)


@dataclass(frozen=True)
class Key:
    label: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, point: Optional[tuple[int, int]]) -> bool:
        if point is None:
            return False
        x, y = point
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h


class VirtualKeyboard:
    """Full keyboard with point-and-hold selection and no per-key open-palm action."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # Anchor the keyboard to the bottom of whatever frame size this
        # instance was actually built with, rather than a KEYBOARD_TOP
        # constant baked in from config.FRAME_HEIGHT. If the webcam
        # doesn't honor the requested capture resolution (very common),
        # the real frame can be shorter than the configured one - a
        # fixed pixel offset would then place the lower rows below the
        # visible/reachable frame area entirely.
        block_h = len(KEYBOARD_ROWS) * (KEY_H + KEY_GAP) + 10 + 18
        self.keyboard_top = self.height - KEYBOARD_BOTTOM_MARGIN - block_h + 18
        self.text = ""
        self.shift = True
        self.caps_lock = False
        self.cursor: Optional[tuple[int, int]] = None
        self.hover_key: Optional[str] = None
        self.hover_started = 0.0
        self.cooldown_until = 0.0
        self.last_action = "Point at a key"
        self.last_action_time = 0.0
        self.enter_pressed = False
        self.keys: list[Key] = []
        self._build()

    def _build(self) -> None:
        self.keys.clear()
        rows = KEYBOARD_ROWS
        y = self.keyboard_top
        usable = self.width - 2 * KEYBOARD_MARGIN

        def _weight(label: str) -> float:
            if label == "SPACE":
                return 2.6
            if label in {"BACKSPACE", "SHIFT", "CLEAR", "ENTER"}:
                return 1.15
            return 1.0

        for row in rows:
            # Widths are proportional to each key's weight, then scaled
            # so the row's total width (keys + gaps) exactly fills
            # `usable`. The previous version multiplied an equal-width
            # `unit` by each key's weight without renormalizing, so a
            # row with several wide control keys (e.g. the bottom
            # SHIFT/SPACE/BACKSPACE/CLEAR/ENTER row) silently overflowed
            # past the right edge of the frame - which meant ENTER, the
            # key FR-6.8 (field submit) depends on, was unreachable.
            total_weight = sum(_weight(label) for label in row)
            gap_total = KEY_GAP * (len(row) - 1)
            unit = (usable - gap_total) / total_weight
            x = float(KEYBOARD_MARGIN)
            for i, label in enumerate(row):
                w = int(round(unit * _weight(label)))
                # Last key in the row absorbs any rounding remainder so
                # the row's right edge lines up with `usable` exactly.
                if i == len(row) - 1:
                    w = int(KEYBOARD_MARGIN + usable - x)
                self.keys.append(Key(label, int(x), y, w, KEY_H))
                x += w + KEY_GAP
            y += KEY_H + KEY_GAP

    def set_cursor(self, point: Optional[tuple[int, int]]) -> None:
        self.cursor = point

    def _key_at_cursor(self) -> Optional[Key]:
        for key in self.keys:
            if key.contains(self.cursor):
                return key
        return None

    def _append(self, value: str) -> None:
        if value.isalpha():
            value = value.upper() if (self.shift or self.caps_lock) else value.lower()
            self.text += value
            # One-shot shift makes normal typing quicker and predictable.
            if self.shift and not self.caps_lock:
                self.shift = False
        else:
            self.text += value

    def _activate(self, label: str) -> None:
        if label == "SPACE":
            self.text += " "
        elif label == "BACKSPACE":
            self.text = self.text[:-1]
        elif label == "CLEAR":
            self.text = ""
        elif label == "ENTER":
            # ENTER finishes the current typing session. The keyboard
            # does not own or create any profile fields.
            self.enter_pressed = True
            self.last_action = "Text entered"
        elif label in {"SHIFT", "CAPS"}:
            if label == "CAPS":
                self.caps_lock = not self.caps_lock
            else:
                self.shift = not self.shift
        else:
            self._append(label)

        if label != "ENTER":
            self.last_action = f"Selected: {label}"
        self.last_action_time = time.time()
        # Pause before the next hold can start counting, so one long
        # hold can't immediately roll into a second accidental select.
        self.cooldown_until = self.last_action_time + RELEASE_AFTER_SELECT_SECONDS

    def update(self, cursor: Optional[tuple[int, int]]) -> None:
        self.set_cursor(cursor)
        key = self._key_at_cursor()
        now = time.time()

        if key is None:
            self.hover_key = None
            self.hover_started = 0.0
            return

        if now < self.cooldown_until:
            return

        if key.label != self.hover_key:
            self.hover_key = key.label
            self.hover_started = now
            return

        if now - self.hover_started >= KEY_DWELL_SECONDS:
            self._activate(key.label)
            self.hover_started = now

    def reset_enter(self) -> None:
        self.enter_pressed = False

    def load_text(self, text: str) -> None:
        """Replace the composed buffer, e.g. when the caller switches
        the active Profile Information Hub field (FR-6.9) or restores
        an already-typed value after returning to a previous field
        (FR-6.12). Deliberately leaves hover/dwell state untouched so a
        field switch doesn't also cancel an in-progress key hold."""
        self.text = text
        self.shift = True
        self.caps_lock = False

    def hovered_key(self) -> Optional[Key]:
        """Return the key currently under the hand pointer."""
        return self._key_at_cursor()

    @property
    def progress(self) -> float:
        if self.hover_key is None or self.hover_started == 0:
            return 0.0
        return min((time.time() - self.hover_started) / KEY_DWELL_SECONDS, 1.0)

    def _keyboard_rect(self) -> tuple[int, int, int, int]:
        bottom = self.keyboard_top + len(KEYBOARD_ROWS) * (KEY_H + KEY_GAP) + 10
        return KEYBOARD_MARGIN - 10, self.keyboard_top - 18, self.width - KEYBOARD_MARGIN + 10, bottom

    def draw(self, frame: np.ndarray) -> None:
        """Draw only the virtual keyboard; no profile/text-entry box."""
        x1, y1, x2, y2 = self._keyboard_rect()
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, self.width), min(y2, self.height)
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), PANEL, -1)
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), BORDER, 1)
        cv2.addWeighted(overlay, KEYBOARD_ALPHA, roi, 1.0 - KEYBOARD_ALPHA, 0, roi)

        active = self._key_at_cursor()
        for key in self.keys:
            hovered = active is not None and key.label == active.label
            fill = PANEL_2 if not hovered else (70, 70, 70)
            # Blend each key's fill with the frame beneath it (rather
            # than a solid cv2.rectangle) so the camera feed - and the
            # user's face if it's behind the keyboard - stays visible
            # through every key, not just the panel background.
            kx1, ky1 = max(key.x, 0), max(key.y, 0)
            kx2, ky2 = min(key.x + key.w, self.width), min(key.y + key.h, self.height)
            if kx2 > kx1 and ky2 > ky1:
                key_roi = frame[ky1:ky2, kx1:kx2]
                key_overlay = np.full_like(key_roi, fill, dtype=np.uint8)
                cv2.addWeighted(key_overlay, KEY_FILL_ALPHA, key_roi, 1.0 - KEY_FILL_ALPHA, 0, key_roi)
            cv2.rectangle(
                frame, (key.x, key.y), (key.x + key.w, key.y + key.h),
                ACCENT if hovered else BORDER, 2 if hovered else 1
            )

            label = key.label
            if len(label) == 1 and label.isalpha():
                label = label.upper() if (self.shift or self.caps_lock) else label.lower()

            scale = 0.48 if len(label) > 8 else 0.58 if len(label) > 4 else 0.65
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            tx = key.x + (key.w - tw) // 2
            ty = key.y + (key.h + th) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, WHITE, 1, cv2.LINE_AA)

        if active is not None and self.progress > 0:
            cx = active.x + active.w // 2
            cy = max(self.keyboard_top - 8, active.y - 8)
            radius = 8
            cv2.ellipse(
                frame, (cx, cy), (radius, radius), -90,
                0, int(360 * self.progress), ACCENT, 3, cv2.LINE_AA
            )

        # ------------------------------------------------------------
        # TYPED TEXT PREVIEW
        # ------------------------------------------------------------
        # This is ONLY a live preview of what the user has typed.
        # It is not a profile-entry field and does not create/require
        # Legal Name, Email, Phone, etc.
        preview_x1 = KEYBOARD_MARGIN
        preview_x2 = self.width - KEYBOARD_MARGIN
        preview_y2 = self.keyboard_top - 28
        preview_y1 = max(TOP_BAR_HEIGHT + 12, preview_y2 - 58)

        # Alpha-blended (not solid) fill, matching the keys, so the
        # preview strip doesn't punch an opaque hole over the camera
        # feed / face when the keyboard sits over them.
        py1, py2 = max(preview_y1, 0), min(preview_y2, self.height)
        px1, px2 = max(preview_x1, 0), min(preview_x2, self.width)
        if py2 > py1 and px2 > px1:
            preview_roi = frame[py1:py2, px1:px2]
            preview_overlay = np.full_like(preview_roi, PANEL_2, dtype=np.uint8)
            cv2.addWeighted(preview_overlay, KEY_FILL_ALPHA, preview_roi, 1.0 - KEY_FILL_ALPHA, 0, preview_roi)
        cv2.rectangle(
            frame,
            (preview_x1, preview_y1),
            (preview_x2, preview_y2),
            BORDER,
            1
        )

        # Show the most recent portion if the text becomes longer than
        # the available width. The actual buffer is never truncated.
        display_text = self.text if self.text else ""
        prefix = ""

        if not display_text:
            display_text = "Start typing..."
            text_color = MUTED
        else:
            text_color = WHITE

            (text_w, _), _ = cv2.getTextSize(
                display_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                1
            )

            max_text_width = preview_x2 - preview_x1 - 24

            if text_w > max_text_width:
                # Keep the end of the text visible, like a normal input
                # field with a horizontal cursor position.
                while display_text:
                    test = "..." + display_text
                    (test_w, _), _ = cv2.getTextSize(
                        test,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        1
                    )
                    if test_w <= max_text_width:
                        display_text = test
                        break
                    display_text = display_text[1:]

        cv2.putText(
            frame,
            display_text,
            (preview_x1 + 12, preview_y2 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            text_color,
            1,
            cv2.LINE_AA
        )

        caps_state = "CAPS ON" if self.caps_lock else "CAPS OFF"
        cv2.putText(
            frame,
            f"{caps_state}  |  Point + hold a key  |  ENTER: finish typing",
            (18, max(TOP_BAR_HEIGHT - 8, 68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            MUTED,
            1,
            cv2.LINE_AA
        )

