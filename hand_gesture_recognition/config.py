"""UI and performance configuration for the non-verbal communication page."""

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540
WINDOW_NAME = "GestureAccess - Non-Verbal Communication"

# Point-and-hold selection. This is intentionally independent of the
# gesture dwell timer so pointer movement remains immediate.
# Increased from the original 0.55 / 0.70 / 0.20 so there is more time
# to hold a key/card before it registers, and more of a pause after a
# selection before the next hold can start.
KEY_DWELL_SECONDS = 1.1
MODE_DWELL_SECONDS = 1.1
RELEASE_AFTER_SELECT_SECONDS = 0.4

# Cursor smoothing. This is an exponential-moving-average factor
# applied to the tracked fingertip position (see gesture.CursorSmoother):
# smoothed = alpha*raw + (1-alpha)*previous_smoothed. Higher = follows
# the raw fingertip more closely (less lag, more jitter); lower = more
# lag but steadier. 0.45 felt noticeably delayed against a fast-moving
# finger, especially right after switching gestures, causing the
# cursor to land on the wrong key - raised to cut that lag while still
# filtering out single-frame tracking jitter.
# Lowered from 0.65: the cursor was tracking the raw fingertip too
# closely, so fast index-finger movement overshot keys before the
# app could register a hold. A lower alpha trades a bit of
# responsiveness for a steadier, easier-to-land cursor.
CURSOR_SMOOTHING_ALPHA = 0.35

# The raw fingertip position was previously mapped 1:1 onto the frame
# (raw_x = normalized_x * frame_width), which meant reaching a key in
# the leftmost/rightmost column, the top row, or the bottom
# SHIFT/SPACE/.../ENTER row required pointing at the literal edge of
# the webcam frame. MediaPipe's hand tracking gets unreliable right
# at the frame boundary (part of the hand starts leaving the visible
# area), so those outer keys were effectively unreachable even though
# they are laid out correctly on screen.
#
# Instead, only fingertip motion within a smaller, comfortable
# "active region" is tracked, and that region is remapped onto the
# full 0-1 cursor range. Each edge has its own margin (rather than one
# shared value) because comfortable reach is NOT symmetric: pointing
# with the right hand (the common case) naturally has less reach
# toward the left before crossing the body, and dropping the arm to
# point low is more awkward than raising it. LEFT and BOTTOM cut a
# much bigger slice off the tracked region than RIGHT and TOP, so
# reaching the left column and the bottom SPACE/CAPS/ENTER row needs
# far less physical movement.
CURSOR_ACTIVE_MARGIN_LEFT = 0.34
CURSOR_ACTIVE_MARGIN_RIGHT = 0.14
CURSOR_ACTIVE_MARGIN_TOP = 0.12
CURSOR_ACTIVE_MARGIN_BOTTOM = 0.34

# UI
WHITE = (255, 255, 255)
MUTED = (180, 180, 180)
PANEL = (28, 28, 28)
PANEL_2 = (38, 38, 38)
BORDER = (90, 90, 90)
ACCENT = (255, 210, 90)
SUCCESS = (110, 235, 140)
WARNING = (90, 210, 255)
DANGER = (100, 100, 240)

# Transparency of the menu ("option window") panel, same idea as
# KEYBOARD_ALPHA below - the camera feed shows through instead of a
# solid fill. Lower = more see-through.
MENU_ALPHA = 0.72

# Top bar now carries a second row showing the active Profile
# Information Hub field and submit/navigation toasts (FR-6.5-FR-6.12),
# so it is taller than the original single-row title bar. Every other
# vertical layout constant below is derived from this one value so the
# whole UI shifts down consistently instead of hardcoding a second set
# of magic numbers.
TOP_BAR_HEIGHT = 70

# Keyboard layout. The keyboard now docks to the bottom of the frame
# (see KEYBOARD_TOP below) instead of sitting just under the header,
# and does not draw a separate text/profile box.
KEY_GAP = 5
KEY_H = 42
KEYBOARD_MARGIN = 18

# Lowered the keyboard's own transparency (and gave individual keys an
# alpha-blended fill instead of a solid one - see VirtualKeyboard.draw)
# so the camera feed, including the user's face, stays visible through
# the keyboard rather than being covered by it.
KEYBOARD_ALPHA = 0.45
KEY_FILL_ALPHA = 0.45

KEYBOARD_ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["@", ".", ",", "_", "-", "+", "=", "!", "?", "#"],
    ["CAPS", "SPACE", "BACKSPACE", "CLEAR", "ENTER"],
]

# Anchored to the bottom of the frame: the whole block (panel padding +
# rows + bottom padding) is measured first, then KEYBOARD_TOP is set so
# the panel's bottom edge sits KEYBOARD_BOTTOM_MARGIN above the frame's
# bottom edge.
KEYBOARD_BOTTOM_MARGIN = 16
_KEYBOARD_BLOCK_H = len(KEYBOARD_ROWS) * (KEY_H + KEY_GAP) + 10 + 18
KEYBOARD_TOP = FRAME_HEIGHT - KEYBOARD_BOTTOM_MARGIN - _KEYBOARD_BLOCK_H + 18
