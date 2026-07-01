#!/usr/bin/env python3
"""
focusViewer.py
==============
Focus-check helper for the capture rig. It turns on all four directional lights
and opens a small window showing a live-view preview frame from the camera so
the operator can set focus before a real capture.

Nothing is saved permanently: each "Capture New Photo" press grabs a fresh
preview into a temporary file that is deleted when the window is closed. It uses
gphoto2's --capture-preview (the camera's live-view frame), so it does not fire
the shutter or download a full RAW.

Launched by run.py's "Open Focus Viewer" button. Requires the same hardware as
the capture stage (camera over USB + Arduino on a serial port) plus gphoto2
reachable through msys2, and Pillow for displaying the preview.
"""

import os
import shutil
import subprocess
import tempfile
import time
import tkinter as tk

import serial

try:
    from PIL import Image, ImageTk
    # LANCZOS moved under Image.Resampling in Pillow >= 9.1; fall back for older.
    RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
except ImportError:                      # Pillow is installed via the modeling deps
    Image = None
    ImageTk = None
    RESAMPLE = None

# ── Hardware config — keep in sync with arduinoIntegration.py ────────────────
SERIAL_PORT = "COM3"
BAUD_RATE   = 115200
MSYS2_SHELL = r"C:\msys64\msys2_shell.cmd"

# message_arduino bit order: (n, e, s, w, g, b, step, dir)
ALL_LIGHTS_ON  = (1, 1, 1, 1, 0, 1, 0, 1)   # same combo as the "allLight" capture
ALL_LIGHTS_OFF = (0, 0, 0, 0, 1, 0, 0, 1)

MAX_VIEW     = (900, 600)   # fallback canvas size before the window is laid out
IMAGE_EXTS   = (".jpg", ".jpeg", ".png", ".ppm")
MIN_SCALE    = 0.1
MAX_SCALE    = 10.0
ZOOM_STEP    = 1.25         # per wheel notch / zoom-button press


class FocusViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Focus Viewer — set camera focus")
        self.tmpdir = tempfile.mkdtemp(prefix="focus_")
        self.ser = None
        self.photo = None     # keep a reference so Tk doesn't garbage-collect it

        # Pan/zoom state for the current preview.
        self.full_img = None  # full-resolution PIL image of the latest preview
        self.canvas_img_id = None
        self.scale = 1.0
        self.img_x = 0.0      # canvas coords of the image's top-left corner
        self.img_y = 0.0
        self._drag = None     # (mouse_x, mouse_y, img_x, img_y) while panning

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Defer hardware startup until the window is up so errors are visible.
        self.root.after(100, self._startup)

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=6)
        self.capture_btn = tk.Button(top, text="Capture New Photo",
                                     command=self.on_capture)
        self.capture_btn.pack(side=tk.LEFT)
        tk.Button(top, text="–", width=3,
                  command=lambda: self._zoom_center(1 / ZOOM_STEP)).pack(side=tk.LEFT, padx=(12, 2))
        tk.Button(top, text="+", width=3,
                  command=lambda: self._zoom_center(ZOOM_STEP)).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Fit", command=self._fit).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Close", command=self._on_close).pack(side=tk.RIGHT)

        self.status = tk.Label(self.root, text="Starting…", anchor="w", fg="#333333")
        self.status.pack(fill=tk.X, padx=8)

        self.canvas = tk.Canvas(self.root, bg="#222222", highlightthickness=0,
                                cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Pan with left-drag; zoom with the mouse wheel (Win/mac use <MouseWheel>,
        # X11 uses Button-4/5).
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)

    def _set_status(self, text: str):
        self.status.config(text=text)
        self.root.update_idletasks()

    # ── Hardware ───────────────────────────────────────────────────────────────
    def message_arduino(self, *bits):
        """Send an 8-bit light/motor command and wait (bounded) for the 'e' ack."""
        if self.ser is None:
            return
        msg = "".join(str(int(b)) for b in bits)
        self.ser.write(msg.encode())
        deadline = time.time() + 10   # don't hang the UI forever if no ack comes
        while time.time() < deadline:
            if self.ser.read(1) == b"e":
                return

    def _startup(self):
        if Image is None:
            self._set_status("Pillow is not installed — click 'Install Python "
                             "dependencies' in the launcher (pip install pillow).")
            self.capture_btn.config(state=tk.DISABLED)
            return
        try:
            self.ser = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=2.5)
            time.sleep(2)
        except Exception as exc:
            self._set_status(f"Could not open serial {SERIAL_PORT}: {exc}")
            self.capture_btn.config(state=tk.DISABLED)
            return
        self._set_status("Turning on all four directional lights…")
        self.message_arduino(*ALL_LIGHTS_ON)
        self.on_capture()

    def _capture_preview(self):
        """Grab one live-view preview frame into the temp dir; return its path or None."""
        cmd = (rf'{MSYS2_SHELL} -mingw64 -defterm -no-start -here '
               rf'-c "gphoto2 --capture-preview --force-overwrite"')
        subprocess.run(cmd, shell=True, cwd=self.tmpdir)
        # gphoto2 --capture-preview writes 'capture_preview.jpg' by default and
        # does not reliably honour --filename, so just pick up whatever image it
        # dropped into the (otherwise temp) directory — newest wins.
        candidates = [
            os.path.join(self.tmpdir, f)
            for f in os.listdir(self.tmpdir)
            if f.lower().endswith(IMAGE_EXTS)
        ]
        if not candidates:
            print("Focus viewer: no image file in", self.tmpdir,
                  "after preview; dir contains:", os.listdir(self.tmpdir))
            return None
        return max(candidates, key=os.path.getmtime)

    # ── Actions ─────────────────────────────────────────────────────────────────
    def on_capture(self):
        if Image is None or self.ser is None:
            return
        self.capture_btn.config(state=tk.DISABLED)
        self._set_status("Capturing preview…")
        try:
            path = self._capture_preview()
            if path is None or not os.path.exists(path):
                self._set_status("No preview file was produced — check the launcher "
                                 "log for gphoto2's output (camera connected? "
                                 "preview supported?).")
                return
            first = self.full_img is None
            img = Image.open(path)
            img.load()                 # fully read before the temp file is reused
            self.full_img = img.copy()
            # Fit the very first frame; afterwards keep the current zoom/pan so the
            # operator can watch the same detail sharpen across captures.
            if first:
                self._fit_to_window()
            self._render()
            self._set_status("Preview updated — scroll to zoom, drag to pan. "
                             "Adjust focus and capture again. Nothing is saved.")
        except Exception as exc:
            self._set_status(f"Preview failed: {exc}")
        finally:
            self.capture_btn.config(state=tk.NORMAL)

    # ── Pan / zoom ──────────────────────────────────────────────────────────────
    def _canvas_size(self):
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:           # window not laid out yet
            return MAX_VIEW
        return w, h

    def _render(self):
        if self.full_img is None:
            return
        iw, ih = self.full_img.size
        disp_w = max(1, int(iw * self.scale))
        disp_h = max(1, int(ih * self.scale))
        self.photo = ImageTk.PhotoImage(self.full_img.resize((disp_w, disp_h), RESAMPLE))
        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(
            self.img_x, self.img_y, anchor="nw", image=self.photo)

    def _fit_to_window(self):
        if self.full_img is None:
            return
        cw, ch = self._canvas_size()
        iw, ih = self.full_img.size
        self.scale = max(MIN_SCALE, min(MAX_SCALE, min(cw / iw, ch / ih)))
        self.img_x = (cw - iw * self.scale) / 2
        self.img_y = (ch - ih * self.scale) / 2

    def _fit(self):
        self._fit_to_window()
        self._render()

    def _zoom_to(self, new_scale, cx, cy):
        """Zoom to `new_scale` keeping the image point under (cx, cy) fixed."""
        if self.full_img is None:
            return
        new_scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))
        if abs(new_scale - self.scale) < 1e-6:
            return
        ix = (cx - self.img_x) / self.scale
        iy = (cy - self.img_y) / self.scale
        self.scale = new_scale
        self.img_x = cx - ix * self.scale
        self.img_y = cy - iy * self.scale
        self._render()

    def _zoom_center(self, factor):
        cw, ch = self._canvas_size()
        self._zoom_to(self.scale * factor, cw / 2, ch / 2)

    def _on_wheel(self, event):
        # <MouseWheel> carries event.delta; X11 sends Button-4 (up) / Button-5 (down).
        up = getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4
        self._zoom_to(self.scale * (ZOOM_STEP if up else 1 / ZOOM_STEP),
                      event.x, event.y)

    def _on_pan_start(self, event):
        self._drag = (event.x, event.y, self.img_x, self.img_y)

    def _on_pan_move(self, event):
        if self._drag is None or self.canvas_img_id is None:
            return
        sx, sy, ox, oy = self._drag
        self.img_x = ox + (event.x - sx)
        self.img_y = oy + (event.y - sy)
        self.canvas.coords(self.canvas_img_id, self.img_x, self.img_y)

    def _on_close(self):
        try:
            if self.ser is not None:
                self.message_arduino(*ALL_LIGHTS_OFF)
                self.ser.close()
        except Exception:
            pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("960x760")
    FocusViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
