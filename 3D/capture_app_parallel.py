import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import time
import os
import re
import serial
import serial.tools.list_ports
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

CAMERA_FOLDER = "/store_00020001/DCIM/100CANON/"
PREVIEW_SIZE = (320, 240)


class CaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("T-Capture Multi-Camera Scanner (Parallel Capture)")
        self.root.minsize(720, 600)
        self.capture_thread = None
        self.stop_event = threading.Event()
        self._photo_refs = []  # keep Tk image refs alive
        self.detected_cam_ports = []  # last camera ports found by the Refresh button
        self._build_ui()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)

        # --- Config ---
        cfg = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        cfg.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="Output Folder:").grid(row=0, column=0, sticky="w")
        self.folder_var = tk.StringVar(value=str(Path.home() / "T-Capture"))
        ttk.Entry(cfg, textvariable=self.folder_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(cfg, text="Browse…", command=self._browse_folder).grid(row=0, column=2)

        ttk.Label(cfg, text="Serial Port:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(cfg, textvariable=self.port_var, width=22)
        self.port_combo.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Button(cfg, text="↻ Refresh Ports & Cameras", command=self._refresh_all).grid(
            row=1, column=2, pady=(6, 0))

        ttk.Label(cfg, text="Cameras:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.cameras_var = tk.StringVar(value="(click Refresh to detect)")
        ttk.Label(cfg, textvariable=self.cameras_var, foreground="#1a6ea8").grid(
            row=2, column=1, columnspan=2, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(cfg, text="Num Captures:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.caps_var = tk.StringVar(value="100")
        ttk.Entry(cfg, textvariable=self.caps_var, width=8).grid(row=3, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(cfg, text="Camera Delay (s):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        delay_row = ttk.Frame(cfg)
        delay_row.grid(row=4, column=1, columnspan=2, sticky="w", padx=6, pady=(6, 0))
        self.delay_var = tk.StringVar(value="0.12")
        ttk.Entry(delay_row, textvariable=self.delay_var, width=8).grid(row=0, column=0)
        ttk.Label(delay_row, text="delay between each camera's shot (0 = all at once)",
                  foreground="#666").grid(row=0, column=1, padx=(8, 0))

        # --- Controls ---
        ctrl = ttk.Frame(self.root, padding=(10, 4))
        ctrl.grid(row=1, column=0, sticky="ew", padx=10)

        self.start_btn = ttk.Button(ctrl, text="▶  Start", command=self._start_capture, width=12)
        self.start_btn.grid(row=0, column=0, padx=(0, 6))
        self.stop_btn = ttk.Button(ctrl, text="■  Stop", command=self._stop_capture, state="disabled", width=12)
        self.stop_btn.grid(row=0, column=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(ctrl, textvariable=self.status_var, font=("", 10, "bold"), foreground="#1a6ea8").grid(
            row=0, column=2, padx=20)

        # --- Progress ---
        prog = ttk.Frame(self.root, padding=(10, 0))
        prog.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))
        prog.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(prog, variable=self.progress_var, maximum=100).grid(row=0, column=0, sticky="ew")
        self.pct_label = ttk.Label(prog, text="0 %", width=6, anchor="e")
        self.pct_label.grid(row=0, column=1, padx=(6, 0))

        # --- Preview ---
        self.prev_frame = ttk.LabelFrame(
            self.root, text="Camera Preview  (updated after each side download)", padding=10)
        self.prev_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=4)
        self.cam_img_labels = []
        self._build_preview_slots(1)  # placeholder; rebuilt once cameras are detected

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=5)
        log_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.root.rowconfigure(4, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state="disabled",
                                                   wrap="word", font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True)

        # Populate the port and camera lists now that the whole UI exists.
        self._refresh_all()

    # ------------------------------------------------------------------ helpers

    def _browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.folder_var.get())
        if folder:
            self.folder_var.set(folder)

    def _refresh_all(self):
        """Refresh both the serial port list and the detected camera list."""
        self._refresh_ports()
        self._refresh_cameras()

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        # Keep the current selection if it is still present; otherwise pick a sensible default.
        if self.port_var.get() not in ports:
            self.port_var.set("")
            if ports:
                for p in ports:
                    if any(tag in p for tag in ("ACM", "USB", "COM", "tty")):
                        self.port_var.set(p)
                        break
                else:
                    self.port_var.set(ports[0])

    def _refresh_cameras(self):
        """Detect cameras with gphoto2 in the background and update the UI."""
        self.cameras_var.set("Detecting…")
        self._set_status("Detecting cameras…")

        def work():
            ports = self._detect_camera_ports_safe()

            def update():
                self.detected_cam_ports = ports
                n = len(ports)
                if n:
                    self.cameras_var.set(f"{n} detected — " + ", ".join(ports))
                else:
                    self.cameras_var.set("No cameras detected")
                # Show one preview slot per camera (at least one placeholder).
                self._build_preview_slots(max(n, 1))
                self._set_status("Ready")

            self.root.after(0, update)

        threading.Thread(target=work, daemon=True).start()

    def _log(self, msg):
        def _do():
            self.log_text.configure(state="normal")
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{ts}] {msg}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(0, _do)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _set_progress(self, pct: float):
        def _do():
            self.progress_var.set(pct)
            self.pct_label.configure(text=f"{pct:.0f} %")
        self.root.after(0, _do)

    def _build_preview_slots(self, n):
        """(Re)build one preview slot per detected camera. Runs on the main thread."""
        def _do():
            for child in self.prev_frame.winfo_children():
                child.destroy()
            self.cam_img_labels = []
            for c in range(n):
                self.prev_frame.columnconfigure(c, weight=1)

            if not PIL_AVAILABLE:
                placeholder = "No image yet\n(install Pillow for previews:\npip install Pillow)"
            else:
                placeholder = "No image yet\n(Pillow installed ✓)"

            for c in range(n):
                padx = (0 if c == 0 else 6, 0 if c == n - 1 else 6)
                lbl = ttk.Label(self.prev_frame, text=placeholder, relief="sunken",
                                anchor="center", width=44, padding=4)
                lbl.grid(row=0, column=c, padx=padx, sticky="nsew")
                ttk.Label(self.prev_frame, text=f"Camera {c + 1}", font=("", 9, "bold")).grid(
                    row=1, column=c, pady=(4, 0))
                self.cam_img_labels.append(lbl)

        # If called from a worker thread, marshal onto the Tk main thread.
        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.root.after(0, _do)

    def _update_preview(self, paths):
        """paths: list of image paths (or None) parallel to self.cam_img_labels."""
        if not PIL_AVAILABLE:
            return

        def _do():
            for path, label in zip(paths, self.cam_img_labels):
                if path and os.path.isfile(path):
                    try:
                        img = Image.open(path)
                        img.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self._photo_refs.append(photo)
                        if len(self._photo_refs) > 20:
                            self._photo_refs.pop(0)
                        label.configure(image=photo, text="")
                        label.image = photo
                    except Exception as exc:
                        label.configure(text=f"Preview error:\n{exc}")

        self.root.after(0, _do)

    def _ask_flip(self):
        done = threading.Event()

        def _do():
            messagebox.showinfo(
                "Flip Object",
                "Side 1 scan complete!\n\nFlip the object to its other side, then click OK to continue."
            )
            done.set()

        self.root.after(0, _do)
        done.wait()

    # ------------------------------------------------------------------ capture control

    def _start_capture(self):
        try:
            caps = int(self.caps_var.get())
            if caps < 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of captures must be a positive integer.")
            return

        try:
            delay = float(self.delay_var.get())
            if delay < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Invalid Input", "Camera delay must be a number ≥ 0 (seconds).")
            return

        if not self.port_var.get():
            messagebox.showerror("Invalid Input", "Please select a serial port.")
            return

        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_progress(0)
        self._set_status("Starting…")
        self.capture_thread = threading.Thread(target=self._run_capture, daemon=True)
        self.capture_thread.start()

    def _stop_capture(self):
        self.stop_event.set()
        self._set_status("Stopping…")
        self._log("Stop requested — will halt after current capture completes.")

    def _finish(self, success=True):
        def _do():
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            if success:
                self.status_var.set("Complete!")
                self._set_progress(100)
        self.root.after(0, _do)

    # ------------------------------------------------------------------ capture loop (background thread)

    def _run_capture(self):
        caps_int = int(self.caps_var.get())
        cam_delay = float(self.delay_var.get())
        port = self.port_var.get()
        base_folder = self.folder_var.get()

        # Open serial
        try:
            ser = serial.Serial(port, baudrate=115200, timeout=1)
        except serial.SerialException as exc:
            self._log(f"Serial port error: {exc}")
            self._set_status("Serial error")
            self._finish(False)
            return

        # Detect cameras
        self._log("Detecting cameras via gphoto2…")
        try:
            cam_ports = self._detect_camera_ports()
        except RuntimeError as exc:
            self._log(f"Camera detection failed: {exc}")
            self._set_status("Camera error")
            ser.close()
            self._finish(False)
            return
        n_cams = len(cam_ports)
        cam_names = [f"cam{i + 1}" for i in range(n_cams)]
        self._log(f"Detected {n_cams} camera(s):")
        for name, port in zip(cam_names, cam_ports):
            self._log(f"  {name}: {port}")
        self._build_preview_slots(n_cams)

        # Create output folder tree
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_path = os.path.join(base_folder, run_name)
        side_paths = {}
        for side in (1, 2):
            dests = []
            for name in cam_names:
                dest = os.path.join(run_path, f"side{side}", name)
                os.makedirs(dest, exist_ok=True)
                dests.append(dest)
            side_paths[side] = dests
        self._log(f"Output: {run_path}")

        # Motor parameters — steps per move scaled to 32 steps for 100 captures
        steps_per_move = int(32 * (100 / caps_int))
        caps_ld = str(caps_int).zfill(4)
        steps_ld = str(steps_per_move).zfill(4)
        ser.timeout = steps_per_move + 2.5
        delay_desc = "all at once" if cam_delay == 0 else f"{cam_delay:g}s between cameras"
        self._log(f"Captures: {caps_int}  |  Steps per move: {steps_per_move}  |  "
                  f"Capture: parallel ({delay_desc})")

        # Store images on camera card for speed; download in bulk after each side
        for port in cam_ports:
            subprocess.run(["gphoto2", "--port", port, "--set-config", "capturetarget=1"],
                           capture_output=True)

        total = caps_int * 2
        stopped = False

        for side in (1, 2):
            if self.stop_event.is_set():
                stopped = True
                break

            if side == 2:
                self._ask_flip()
                if self.stop_event.is_set():
                    stopped = True
                    break

            cam_dests = side_paths[side]
            self._log(f"--- Side {side} scan starting ---")

            for i in range(caps_int):
                if self.stop_event.is_set():
                    stopped = True
                    break

                self._set_status(f"Side {side}  —  capture {i + 1} / {caps_int}")

                # Fire cameras in parallel, offset by the GUI delay; retry misses.
                results = self._capture_all(cam_ports, cam_delay)
                for name, port in zip(cam_names, cam_ports):
                    ok, msg = results[port]
                    if not ok:
                        self._log(f"⚠ {name} capture {i + 1} FAILED: {msg or 'unknown error'}")

                # Signal Arduino to advance turntable and wait for 'e' acknowledgement
                ser.write(caps_ld.encode())
                ser.write(steps_ld.encode())
                ser.read_until(size=1)

                done = (side - 1) * caps_int + (i + 1)
                self._set_progress(done / total * 100)
                self._log(f"Side {side}  capture {i + 1}/{caps_int} done")

            else:
                # Inner loop completed without break → download and clear cameras
                self._set_status(f"Side {side}  —  downloading images…")
                self._log(f"Side {side} complete. Checking file counts…")

                # Each camera numbers its own DCIM folder independently (100CANON,
                # 101CANON, …) and may expose a different store id, so discover the
                # real image folder per camera instead of assuming a shared path.
                cam_folders = [self._find_image_folder(port) for port in cam_ports]
                for name, folder in zip(cam_names, cam_folders):
                    if not folder:
                        self._log(f"  {name}: no image folder found on camera!")

                counts = [self._count_camera_files(port, folder) if folder else 0
                          for port, folder in zip(cam_ports, cam_folders)]
                self._log("  ".join(f"{name}: {c} files" for name, c in zip(cam_names, counts))
                          + f"  (expected {caps_int} each)")

                # Download each camera and record how many files actually landed on
                # disk. We only ever clear a card after its download is verified, so a
                # failed download can never destroy the only copy of the photos.
                preview_paths = []
                downloaded = []  # files verified on disk per camera
                for name, port, dest, folder, on_cam in zip(
                        cam_names, cam_ports, cam_dests, cam_folders, counts):
                    if not folder:
                        self._log(f"Skipping {name} — no image folder found on camera.")
                        preview_paths.append(None)
                        downloaded.append(0)
                        continue
                    self._log(f"Downloading from {name} ({folder})…")
                    subprocess.run(
                        ["gphoto2", "--port", port, "--recurse", "--get-all-files",
                         "--folder", folder],
                        cwd=dest, capture_output=True)
                    files = [p for p in Path(dest).iterdir() if p.is_file()]
                    imgs = sorted(Path(dest).glob("*.[Jj][Pp][Gg]"))
                    downloaded.append(len(files))
                    preview_paths.append(str(imgs[-1]) if imgs else None)
                    note = "" if len(files) >= on_cam else "  ⚠ fewer than on camera!"
                    self._log(f"{name}: downloaded {len(files)} file(s){note}")

                # Update previews with the last downloaded image from each camera
                self._update_preview(preview_paths)

                self._log("Clearing camera cards (only where download verified)…")
                for name, port, folder, on_cam, n_dl in zip(
                        cam_names, cam_ports, cam_folders, counts, downloaded):
                    if folder and n_dl > 0 and n_dl >= on_cam:
                        subprocess.run(
                            ["gphoto2", "--port", port, "--delete-all-files", "--folder", folder],
                            capture_output=True)
                        self._log(f"{name}: card cleared.")
                    else:
                        self._log(f"{name}: NOT cleared — download unverified, "
                                  f"photos kept on card for safety.")
                self._log(f"Side {side} images saved to {os.path.join(run_path, f'side{side}')}")
                time.sleep(1)

        ser.close()

        if stopped:
            self._log("Scan stopped by user.")
            self._set_status("Stopped")
        else:
            self._log("All done! Scanning complete.")
            self._set_status("Complete!")

        self._finish(not stopped)

    # ------------------------------------------------------------------ gphoto2 helpers

    def _capture_all(self, cam_ports, delay):
        """Trigger all cameras in parallel, offsetting each start by `delay` seconds.

        Each camera fires in its own thread. With delay=0 they all fire at the
        exact same instant; with a small delay the triggers are spread out to
        avoid USB/PTP collisions while exposures still overlap. The right value
        depends on the USB topology, so it is set from the GUI. Each camera's
        result is checked and any that errors is retried once.

        Returns {port: (ok: bool, message: str)}.
        """
        def fire(ports):
            out = {}
            lock = threading.Lock()

            def worker(idx, port):
                if delay:
                    time.sleep(idx * delay)
                proc = subprocess.run(
                    ["gphoto2", "--port", port, "--capture-image", "--folder", CAMERA_FOLDER],
                    capture_output=True, text=True)
                lines = (proc.stdout + proc.stderr).strip().splitlines()
                with lock:
                    out[port] = (proc.returncode == 0, lines[-1] if lines else "")

            threads = [threading.Thread(target=worker, args=(i, p), daemon=True)
                       for i, p in enumerate(ports)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return out

        results = fire(cam_ports)
        failed = [p for p in cam_ports if not results[p][0]]
        if failed:
            time.sleep(1.0)  # let the bus settle before retrying
            results.update(fire(failed))
        return results

    def _detect_camera_ports_safe(self):
        """Return a list of gphoto2 camera ports, or [] on any failure (never raises)."""
        try:
            result = subprocess.run(["gphoto2", "--auto-detect"], capture_output=True, text=True)
        except FileNotFoundError:
            self._log("gphoto2 not found — install it to detect cameras.")
            return []
        return re.findall(r"(usb:\d+,\d+)", result.stdout)

    def _detect_camera_ports(self):
        ports = self._detect_camera_ports_safe()
        if not ports:
            raise RuntimeError("No cameras found by gphoto2.")
        return ports

    def _find_image_folder(self, port):
        """Return the camera folder that actually holds image files, or None.

        Each camera maintains its own DCIM folder numbering and may expose a
        different store id, so the folder cannot be assumed identical across
        cameras. We list the whole filesystem recursively and pick the folder
        that contains files (preferring a DCIM path).
        """
        result = subprocess.run(
            ["gphoto2", "--port", port, "--folder", "/", "--recurse", "--list-files"],
            capture_output=True, text=True,
        )
        matches = re.findall(r"There (?:is|are) (\d+) files? in folder '([^']+)'", result.stdout)
        folders = [folder for n, folder in matches if int(n) > 0]
        for folder in folders:
            if "DCIM" in folder.upper():
                return folder
        return folders[0] if folders else None

    def _count_camera_files(self, port, folder):
        result = subprocess.run(
            ["gphoto2", "--port", port, "--list-files", "--folder", folder],
            capture_output=True, text=True,
        )
        m = re.search(r"There (?:is|are) (\d+) files?", result.stdout)
        return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    CaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
