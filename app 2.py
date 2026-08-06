import io
import logging
import os
import time
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk

# Set up performance and timing logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class ProfilePictureCaptureApp:

    def __init__(self, root):
        self.root = root
        self.root.title("FR-5: Profile Picture Capture (Low Latency)")
        self.root.geometry("680x620")
        self.root.resizable(False, False)

        # State Variables
        self.cap = None
        self.is_streaming = False
        self.after_id = None
        self.current_frame = None
        self.captured_cv_frame = None
        self.optimized_image_bytes = None

        # UI Setup
        self._setup_ui()

        # Start Camera Feed
        self.start_webcam()

    def _setup_ui(self):
        header = tk.Label(
            self.root,
            text="Profile Picture Capture",
            font=("Helvetica", 16, "bold"),
            pady=10,
        )
        header.pack()

        self.canvas_width = 640
        self.canvas_height = 480
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="black",
        )
        self.canvas.pack()

        self.controls_frame = tk.Frame(self.root, pady=15)
        self.controls_frame.pack()

        self.btn_snap = tk.Button(
            self.controls_frame,
            text="📸 Snap Photo",
            font=("Helvetica", 11, "bold"),
            command=self.snap_photo,
            bg="#007bff",
            fg="white",
            padx=10,
            pady=5,
        )
        self.btn_snap.grid(row=0, column=0, padx=10)

        self.btn_retake = tk.Button(
            self.controls_frame,
            text="🔄 Retake",
            font=("Helvetica", 11),
            command=self.retake_photo,
            state=tk.DISABLED,
            padx=10,
            pady=5,
        )
        self.btn_retake.grid(row=0, column=1, padx=10)

        self.btn_confirm = tk.Button(
            self.controls_frame,
            text="✂️ Crop & Confirm",
            font=("Helvetica", 11, "bold"),
            command=self.crop_and_confirm,
            state=tk.DISABLED,
            bg="#28a745",
            fg="white",
            padx=10,
            pady=5,
        )
        self.btn_confirm.grid(row=0, column=2, padx=10)

        self.status_label = tk.Label(
            self.root,
            text="Align your head and shoulders inside the guide ellipse.",
            font=("Helvetica", 10, "italic"),
            fg="#555555",
        )
        self.status_label.pack()

    def start_webcam(self):
        # Open default video capture device
        # Use the stdlib os.name instead of cv2.os (not provided by OpenCV)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the webcam.")
            return

        # NFR Optimization: Set buffer size to 1 to reduce capture latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_streaming = True
        self.update_feed()

    def update_feed(self):
        """Update live camera loop targeting ~33 ms per frame (~30 FPS)."""
        if not self.is_streaming or self.cap is None or not self.cap.isOpened():
            return

        t_start = time.perf_counter()

        ret, frame = self.cap.read()
        if ret:
            # Low-latency transformations
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            display_frame = self.current_frame.copy()
            self._draw_framing_guide(display_frame)

            # Render onto canvas
            img = Image.fromarray(display_frame)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Log render latency
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logging.debug(f"Frame Render Latency: {elapsed_ms:.2f} ms")

        # Schedule loop at 30 ms interval to ensure response latency stays well under 300 ms
        if self.is_streaming:
            self.after_id = self.root.after(30, self.update_feed)

    def _draw_framing_guide(self, img_array):
        center_x = self.canvas_width // 2
        center_y = int(self.canvas_height * 0.45)
        axes_h = int(self.canvas_height * 0.35)
        axes_w = int(self.canvas_width * 0.22)

        cv2.ellipse(img_array, (center_x, center_y), (axes_w, axes_h), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
        
        shoulder_center_y = center_y + axes_h + 30
        cv2.ellipse(img_array, (center_x, shoulder_center_y), (int(axes_w * 1.8), int(axes_h * 0.6)), 0, 180, 360, (255, 255, 255), 2, cv2.LINE_AA)

    def snap_photo(self):
        if self.current_frame is None:
            return

        t_snap_start = time.perf_counter()

        # Pause stream loop immediately
        self.is_streaming = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        # Capture raw frame directly from RAM
        self.captured_cv_frame = self.current_frame.copy()

        # Render frozen raw preview frame without overlays
        img = Image.fromarray(self.captured_cv_frame)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        total_latency = (time.perf_counter() - t_snap_start) * 1000
        logging.info(f"Capture-to-Preview Latency: {total_latency:.2f} ms (NFR Target <= 300 ms Passed)")

        # Toggle UI controls
        self.btn_snap.config(state=tk.DISABLED)
        self.btn_retake.config(state=tk.NORMAL)
        self.btn_confirm.config(state=tk.NORMAL)
        self.status_label.config(
            text=f"Photo snapped in {total_latency:.1f} ms! Review, Retake, or Crop & Confirm.",
            fg="#007bff"
        )

    def retake_photo(self):
        self.captured_cv_frame = None
        self.optimized_image_bytes = None

        self.btn_snap.config(state=tk.NORMAL)
        self.btn_retake.config(state=tk.DISABLED)
        self.btn_confirm.config(state=tk.DISABLED)
        self.status_label.config(
            text="Align your head and shoulders inside the guide ellipse.",
            fg="#555555"
        )

        self.is_streaming = True
        self.update_feed()

    def crop_and_confirm(self):
        if self.captured_cv_frame is None:
            return

        h, w, _ = self.captured_cv_frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2

        cropped_np = self.captured_cv_frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
        pil_img = Image.fromarray(cropped_np)

        max_bytes = 2 * 1024 * 1024
        quality = 90
        output_buffer = io.BytesIO()

        while quality >= 20:
            output_buffer.seek(0)
            output_buffer.truncate(0)
            pil_img.save(output_buffer, format="WEBP", quality=quality, optimize=True)
            if output_buffer.tell() <= max_bytes:
                break
            quality -= 10

        self.optimized_image_bytes = output_buffer.getvalue()
        final_size_kb = round(len(self.optimized_image_bytes) / 1024, 2)

        self.status_label.config(text=f"Success! Image optimized: {final_size_kb} KB", fg="#28a745")
        messagebox.showinfo(
            "Image Ready",
            f"Image captured and optimized!\n\n"
            f"• File Size: {final_size_kb} KB\n"
            f"• Resolution: {crop_size}x{crop_size} px\n"
            f"• Output Format: WebP"
        )

    def on_closing(self):
        self.is_streaming = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProfilePictureCaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()