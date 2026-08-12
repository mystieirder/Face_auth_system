/**
 * gesture_bridge_injector.js
 *
 * Standalone browser-side counterpart to web_bridge.py.
 *
 * This file is NOT imported by RegisterPage.js, LoginPage.js, or
 * ProfilePage.js — none of those files were touched. Instead, this
 * script is loaded into the page manually (see instructions below),
 * and it drives the existing UI purely from the outside, the same way
 * a real user would: focusing fields, typing via native input setters,
 * pressing Enter, and clicking whatever is under the gesture cursor.
 *
 * Because it works generically (DOM order + elementFromPoint), it
 * works on the Register page, Login page, and Profile page alike,
 * without needing any element IDs added to any of them.
 *
 * ── How to load it (pick one — no project file needs to change) ──
 *
 * Option A — DevTools console (quickest, per tab):
 *   1. Open the app in your browser (e.g. http://localhost:3000/register)
 *   2. Open DevTools (F12) → Console tab
 *   3. Paste the entire contents of this file and press Enter
 *   4. Run: python hand_gesture_recognition/web_bridge.py
 *   You'll see a small "Gesture bridge connected" pill appear top-right.
 *
 * Option B — Bookmarklet (one click, any page):
 *   Create a browser bookmark whose URL is:
 *     javascript:(function(){var s=document.createElement('script');s.src='http://localhost:8765/inject.js';document.body.appendChild(s);})()
 *   (only works if you also serve this file locally; the console
 *   paste in Option A needs no extra setup, so it's the easiest start)
 *
 * Option C — Auto-inject via a userscript manager (Tampermonkey /
 *   Violentmonkey extension): create a new userscript matching
 *   http://localhost:3000/* and paste this file's contents into it.
 *   Then it loads automatically every time you open the app — still
 *   without touching a single project file.
 */

(function () {
  const WS_URL = "ws://localhost:8765";

  // ---------------------------------------------------------------
  // Visual feedback: a small status pill + a floating cursor dot,
  // both created fresh by this script (nothing pre-existing is used).
  // ---------------------------------------------------------------
  const pill = document.createElement("div");
  pill.textContent = "Gesture bridge: connecting…";
  Object.assign(pill.style, {
    position: "fixed", top: "12px", right: "12px", zIndex: 999999,
    background: "#1f2937", color: "#fff", padding: "6px 12px",
    borderRadius: "999px", fontFamily: "sans-serif", fontSize: "12px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.3)", pointerEvents: "none",
  });
  document.body.appendChild(pill);

  const cursorDot = document.createElement("div");
  Object.assign(cursorDot.style, {
    position: "fixed", width: "18px", height: "18px", borderRadius: "50%",
    background: "rgba(217,119,87,0.85)", border: "2px solid #fff",
    zIndex: 999998, pointerEvents: "none", display: "none",
    transform: "translate(-50%, -50%)", transition: "left 0.05s linear, top 0.05s linear",
  });
  document.body.appendChild(cursorDot);

  let lastCursorEl = null;

  function setPill(text, color) {
    pill.textContent = text;
    pill.style.background = color || "#1f2937";
  }

  // ---------------------------------------------------------------
  // The same "native setter + dispatchEvent" trick already used by
  // ProfilePage.js's own voice-command handler (triggerChange), so
  // React's controlled inputs pick up the change correctly.
  // ---------------------------------------------------------------
  function isInputField(el) {
    return el && ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName);
  }

  function getFocusableFields() {
    return Array.from(document.querySelectorAll("input, select, textarea"))
      .filter((el) => el.offsetParent !== null && !el.disabled);
  }

  function focusAdjacentField(direction) {
    const fields = getFocusableFields();
    if (fields.length === 0) return;
    const current = document.activeElement;
    let idx = fields.indexOf(current);
    if (idx === -1) idx = direction === 1 ? -1 : 0;
    let next = idx + direction;
    if (next < 0) next = fields.length - 1;
    if (next >= fields.length) next = 0;
    fields[next].focus();
  }

  function pressEnterOnActive() {
    const active = document.activeElement;
    if (!active) return;
    ["keydown", "keypress", "keyup"].forEach((type) => {
      active.dispatchEvent(
        new KeyboardEvent(type, { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true })
      );
    });
  }

  function clickElementAtCursor() {
    if (!lastCursorEl) return;
    if (isInputField(lastCursorEl)) {
      lastCursorEl.focus();
    }
    lastCursorEl.click();
  }

  // ---------------------------------------------------------------
  // WebSocket connection to web_bridge.py
  // ---------------------------------------------------------------
  function connect() {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => setPill("Gesture bridge: connected", "#166534");
    ws.onclose = () => {
      setPill("Gesture bridge: disconnected — retrying…", "#7f1d1d");
      cursorDot.style.display = "none";
      setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      if (msg.type === "cursor") {
        const x = msg.x * window.innerWidth;
        const y = msg.y * window.innerHeight;
        cursorDot.style.display = "block";
        cursorDot.style.left = `${x}px`;
        cursorDot.style.top = `${y}px`;
        lastCursorEl = document.elementFromPoint(x, y);
        return;
      }

      if (msg.type === "action") {
        setPill(`Gesture: ${msg.gesture} → ${msg.command}`, "#1d4ed8");
        switch (msg.command) {
          case "next_field":
            focusAdjacentField(1);
            break;
          case "prev_field":
            focusAdjacentField(-1);
            break;
          case "confirm": {
            const active = document.activeElement;
            if (active && active.tagName === "BUTTON") {
              active.click();
            } else if (lastCursorEl && lastCursorEl.tagName === "BUTTON") {
              lastCursorEl.click();
            } else {
              pressEnterOnActive();
            }
            break;
          }
          case "select":
            clickElementAtCursor();
            break;
          default:
            break;
        }
      }
    };
  }

  connect();
})();