import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import time
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# The turntable/LED Arduino is optional: without pyserial the app still runs
# phone-only (and DSLR-only) scans, just without turntable or LED control.
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# The phone hub (phone_server.py) lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phone_server  # noqa: E402

CAMERA_FOLDER = "/store_00020001/DCIM/100CANON/"
PREVIEW_SIZE = (320, 240)
PHONE_PORT = 8000        # `tailscale serve --bg 8000` forwards the phones' HTTPS to this port
PHONE_POLL_MS = 1000     # how often the phone list in the UI refreshes
PHONE_TIMEOUT_S = 15     # per-attempt wait for a phone's upload (the hub retries once)
# Opening the serial port toggles DTR, which resets the Arduino; the sketch
# needs about this long to reboot before it will listen for commands again.
ARDUINO_RESET_S = 2.0
# After that, keep probing for up to this long before declaring the board dead.
ARDUINO_BOOT_TIMEOUT_S = 10.0
# Port-list entry meaning "no Arduino": scans then fire the cameras at an
# interval (the Settle time) with nothing advancing between shots.
NO_TURNTABLE = "(none — no turntable)"


class CaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("T-Capture Multi-Camera Scanner (Parallel Capture + Phones)")
        self.root.minsize(720, 720)
        self.capture_thread = None
        self.stop_event = threading.Event()
        self._photo_refs = []  # keep Tk image refs alive
        self.detected_cam_ports = []  # last camera ports found by the Refresh button
        self._warned_no_gphoto2 = False
        self.leds_on = False  # desired LED state, re-asserted after every board reset
        self.ser = None  # shared Arduino connection, opened on first use (see _get_serial)
        self.ser_port = None
        self._ser_lock = threading.RLock()
        self.hub = None  # phone hub, run in this process; None if its port was unavailable
        self.hub_error = ""
        self._start_hub()
        self._build_ui()

    def _start_hub(self):
        """Run the phone hub inside the app so scans can call it directly."""
        try:
            self.hub, server = phone_server.make_server(
                "127.0.0.1", PHONE_PORT, Path.home() / "T-Capture")
        except OSError as exc:
            self.hub_error = (f"port {PHONE_PORT} unavailable ({exc.strerror}) — "
                              f"is phone_server.py still running? Stop it and restart this app.")
            return
        threading.Thread(target=server.serve_forever, daemon=True).start()

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

        ttk.Label(cfg, text="Turntable Port:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.port_var = tk.StringVar(value=NO_TURNTABLE)
        self.port_var.trace_add("write", self._on_port_change)
        self.port_combo = ttk.Combobox(cfg, textvariable=self.port_var, width=22)
        self.port_combo.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Button(cfg, text="↻ Refresh Ports & Cameras", command=self._refresh_all).grid(
            row=1, column=2, pady=(6, 0))

        ttk.Label(cfg, text="DSLR Cameras:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.cameras_var = tk.StringVar(value="(click Refresh to detect)")
        ttk.Label(cfg, textvariable=self.cameras_var, foreground="#1a6ea8").grid(
            row=2, column=1, columnspan=2, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(cfg, text="Num Captures:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.caps_var = tk.StringVar(value="100")
        ttk.Entry(cfg, textvariable=self.caps_var, width=8).grid(row=3, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(cfg, text="Camera Delay (s):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        delay_row = ttk.Frame(cfg)
        delay_row.grid(row=4, column=1, columnspan=2, sticky="w", padx=6, pady=(6, 0))
        self.delay_var = tk.StringVar(value="0.5")
        ttk.Entry(delay_row, textvariable=self.delay_var, width=8).grid(row=0, column=0)
        ttk.Label(delay_row, text="delay between each camera's shot (0 = all at once)",
                  foreground="#666").grid(row=0, column=1, padx=(8, 0))

        self.both_sides_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg, text="Scan both sides (asks you to flip the object after side 1; "
                                  "untick for a single-side scan)",
                        variable=self.both_sides_var).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # --- Phone cameras ---
        ph = ttk.LabelFrame(self.root, text="Phone Cameras (iPhones over Tailscale)", padding=10)
        ph.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        ph.columnconfigure(1, weight=1)

        self.use_phones_var = tk.BooleanVar(value=self.hub is not None)
        use_cb = ttk.Checkbutton(ph, text="Use phones in scan", variable=self.use_phones_var)
        use_cb.grid(row=0, column=0, sticky="w")
        if self.hub is None:
            use_cb.configure(state="disabled")
        self.phones_var = tk.StringVar(value="")
        ttk.Label(ph, textvariable=self.phones_var, foreground="#1a6ea8").grid(
            row=0, column=1, sticky="w", padx=6)

        ttk.Label(ph, text="Phone URL:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.phone_url_var = tk.StringVar(value="(looking up Tailscale name…)")
        ttk.Entry(ph, textvariable=self.phone_url_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6, pady=(6, 0))

        ttk.Label(ph, text="Settle (s):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        settle_row = ttk.Frame(ph)
        settle_row.grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        self.settle_var = tk.StringVar(value="0.5")
        ttk.Entry(settle_row, textvariable=self.settle_var, width=8).grid(row=0, column=0)
        ttk.Label(settle_row, text="wait before each shot so phone autofocus can settle "
                                   "(with no turntable: the time between shots)",
                  foreground="#666").grid(row=0, column=1, padx=(8, 0))

        # --- Controls ---
        ctrl = ttk.Frame(self.root, padding=(10, 4))
        ctrl.grid(row=2, column=0, sticky="ew", padx=10)

        self.start_btn = ttk.Button(ctrl, text="▶  Start", command=self._start_capture, width=12)
        self.start_btn.grid(row=0, column=0, padx=(0, 6))
        self.stop_btn = ttk.Button(ctrl, text="■  Stop", command=self._stop_capture, state="disabled", width=12)
        self.stop_btn.grid(row=0, column=1)
        self.test_btn = ttk.Button(ctrl, text="📷  Test Shot", command=self._test_shot, width=14)
        self.test_btn.grid(row=0, column=2, padx=(6, 0))
        self.leds_on_btn = ttk.Button(ctrl, text="💡  LEDs On", command=self._leds_on, width=12)
        self.leds_on_btn.grid(row=0, column=3, padx=(6, 0))
        self.leds_off_btn = ttk.Button(ctrl, text="LEDs Off", command=self._leds_off, width=12)
        self.leds_off_btn.grid(row=0, column=4, padx=(6, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(ctrl, textvariable=self.status_var, font=("", 10, "bold"), foreground="#1a6ea8").grid(
            row=0, column=5, padx=20)

        # --- Progress ---
        prog = ttk.Frame(self.root, padding=(10, 0))
        prog.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 4))
        prog.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(prog, variable=self.progress_var, maximum=100).grid(row=0, column=0, sticky="ew")
        self.pct_label = ttk.Label(prog, text="0 %", width=6, anchor="e")
        self.pct_label.grid(row=0, column=1, padx=(6, 0))

        # --- Preview ---
        self.prev_frame = ttk.LabelFrame(
            self.root, text="Camera Preview  (updated after each side download)", padding=10)
        self.prev_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=4)
        self.cam_img_labels = []
        self._build_preview_slots(self._camera_labels(0, []))  # placeholder; rebuilt once cameras are detected

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=5)
        log_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.root.rowconfigure(5, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state="disabled",
                                                   wrap="word", font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True)

        # Populate the port and camera lists now that the whole UI exists.
        self._refresh_all()
        self._poll_phones()
        threading.Thread(target=self._lookup_phone_url, daemon=True).start()

    # ------------------------------------------------------------------ phones

    def _poll_phones(self):
        """Keep the connected-phones line current."""
        if self.hub is None:
            self.phones_var.set(f"⚠ phones unavailable: {self.hub_error}")
        else:
            phones = self.hub.status()["phones"]
            if phones:
                desc = ", ".join(
                    p["cam"] + (f" ({p['info']['width']}×{p['info']['height']})"
                                if p["info"].get("width") else "")
                    for p in phones)
                self.phones_var.set(f"{len(phones)} connected — {desc}")
            else:
                self.phones_var.set("none connected")
        self.root.after(PHONE_POLL_MS, self._poll_phones)

    def _lookup_phone_url(self):
        dns = phone_server.tailscale_dns_name()
        url = (f"https://{dns}/   (needs `tailscale serve --bg {PHONE_PORT}` running)"
               if dns else "(Tailscale name not found — is tailscale up?)")
        self.root.after(0, lambda: self.phone_url_var.set(url))

    def _active_phones(self):
        """Names of the phones this scan will use ([] if 'Use phones' is off)."""
        if self.hub is None or not self.use_phones_var.get():
            return []
        return self.hub.connected()

    def _camera_labels(self, n_dslr, phone_names):
        labels = [f"Camera {i + 1}" for i in range(n_dslr)] + [f"📱 {p}" for p in phone_names]
        return labels or ["Camera 1"]

    def _phone_shoot(self, root, dest_dir, name, phone_names):
        """Fire the given phones and wait for their uploads.

        Files land at <root>/<dest_dir>/<phone>/<name>.jpg. Returns
        {phone: {"ok": bool, "path": ..., "error": ...}}; a phone that dropped
        out reports ok=False rather than raising.
        """
        self.hub.out_dir = Path(root)
        try:
            return self.hub.shoot(dest_dir, name, timeout=PHONE_TIMEOUT_S, cams=phone_names)["results"]
        except phone_server.NoPhones:
            return {}

    def _ensure_phones(self, expected):
        """Block until every phone in `expected` is connected again.

        Returns False if the user gives up, which stops the scan. Runs on the
        capture thread; the dialog is shown on the Tk thread, like _ask_flip.
        """
        while True:
            missing = sorted(set(expected) - set(self.hub.connected()))
            if not missing:
                return True
            self._log(f"⚠ Phone(s) not connected: {', '.join(missing)}")
            answer = []
            done = threading.Event()

            def _do():
                answer.append(messagebox.askretrycancel(
                    "Phone Disconnected",
                    f"Not connected: {', '.join(missing)}\n\n"
                    "Reconnect them (open the Phone URL, tap Start camera), then click Retry.\n"
                    "Cancel stops the scan."))
                done.set()

            self.root.after(0, _do)
            done.wait()
            if not answer[0]:
                return False

    # ------------------------------------------------------------------ helpers

    def _browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.folder_var.get())
        if folder:
            self.folder_var.set(folder)

    def _refresh_all(self):
        """Refresh both the serial port list and the detected camera list."""
        self._refresh_ports()
        self._refresh_cameras()

    def _turntable_port(self):
        """The selected Arduino serial port, or "" when running without a turntable."""
        port = self.port_var.get().strip()
        return "" if port in ("", NO_TURNTABLE) else port

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()] if SERIAL_AVAILABLE else []
        self.port_combo["values"] = [NO_TURNTABLE] + ports
        # Keep the current selection if it is still present; otherwise auto-pick
        # an Arduino-looking port, and fall back to "no turntable" rather than a
        # random tty so a phone-only setup works without touching this box.
        if self.port_var.get() not in [NO_TURNTABLE] + ports:
            self.port_var.set(NO_TURNTABLE)
            for p in ports:
                if any(tag in p.upper() for tag in ("ACM", "USB", "COM")):
                    self.port_var.set(p)
                    break

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
                    self.cameras_var.set("none detected (phones only)")
                # Show one preview slot per camera (at least one placeholder).
                self._build_preview_slots(self._camera_labels(n, self._active_phones()))
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

    def _build_preview_slots(self, labels):
        """(Re)build one preview slot per camera, captioned with `labels`. Runs on the main thread."""
        n = len(labels)

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
                ttk.Label(self.prev_frame, text=labels[c], font=("", 9, "bold")).grid(
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
                        img.draft("RGB", PREVIEW_SIZE)  # JPEG: decode at reduced size (phone frames are ~12 MP)
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

        try:
            if float(self.settle_var.get()) < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Invalid Input", "Phone settle time must be a number ≥ 0 (seconds).")
            return

        if self._turntable_port() and not SERIAL_AVAILABLE:
            messagebox.showerror("pyserial Missing",
                                 "A turntable port is selected but pyserial is not installed.\n\n"
                                 "Install it (pip install pyserial) or choose "
                                 f"'{NO_TURNTABLE}'.")
            return

        if self.use_phones_var.get() and not self._active_phones():
            messagebox.showerror(
                "No Phones",
                "'Use phones in scan' is ticked but no phone is connected.\n\n"
                "Open the Phone URL on each iPhone and tap Start camera, or untick the box.")
            return

        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.test_btn.configure(state="disabled")
        self.leds_on_btn.configure(state="disabled")
        self.leds_off_btn.configure(state="disabled")
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
            self.test_btn.configure(state="normal")
            self.leds_on_btn.configure(state="normal")
            self.leds_off_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            if success:
                self.status_var.set("Complete!")
                self._set_progress(100)
        self.root.after(0, _do)

    # ------------------------------------------------------------------ test shot

    def _test_shot(self):
        """Capture and preview a single frame from every camera for color/brightness checks.

        Independent of the turntable/serial flow: it fires each detected camera
        once, downloads the frame straight to disk, and shows it in the preview
        slots. Runs on a background thread so the UI stays responsive.
        """
        if self.capture_thread and self.capture_thread.is_alive():
            messagebox.showinfo("Busy", "A scan is currently running. Stop it before taking a test shot.")
            return
        self.test_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self._set_status("Taking test shot…")
        threading.Thread(target=self._run_test_shot, daemon=True).start()

    def _run_test_shot(self):
        try:
            self._log("Test shot: detecting cameras…")
            cam_ports = self._detect_camera_ports_safe()
            phone_names = self._active_phones()
            if not cam_ports and not phone_names:
                self._log("Test shot: no cameras detected and no phones connected.")
                self._set_status("No cameras")
                return

            n_cams = len(cam_ports)
            cam_names = [f"cam{i + 1}" for i in range(n_cams)]
            self._build_preview_slots(self._camera_labels(n_cams, phone_names))

            # Dedicated folder that is overwritten on each test shot.
            base = self.folder_var.get()
            dest = os.path.join(base, "_test_shots")
            os.makedirs(dest, exist_ok=True)

            # Phones fire at the same time as the DSLRs; their files land in
            # _test_shots/<phone>/latest.jpg.
            phone_results = {}
            phone_thread = None
            if phone_names:
                phone_thread = threading.Thread(
                    target=lambda: phone_results.update(
                        self._phone_shoot(base, "_test_shots", "latest", phone_names)),
                    daemon=True)
                phone_thread.start()

            results = {}
            lock = threading.Lock()

            def worker(name, port):
                out_path = os.path.join(dest, f"{name}.jpg")
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                # Capture to the camera's internal RAM (capturetarget=0), NOT the
                # memory card, so test shots never land on the card and skew the
                # file counts the main scan relies on. The scan resets this to 1.
                subprocess.run(
                    ["gphoto2", "--port", port, "--set-config", "capturetarget=0"],
                    capture_output=True)
                proc = subprocess.run(
                    ["gphoto2", "--port", port, "--capture-image-and-download",
                     "--force-overwrite", "--filename", out_path],
                    capture_output=True, text=True)
                ok = proc.returncode == 0 and os.path.isfile(out_path)
                lines = (proc.stdout + proc.stderr).strip().splitlines()
                with lock:
                    results[port] = (out_path if ok else None,
                                     lines[-1] if lines else "")

            threads = [threading.Thread(target=worker, args=(nm, p), daemon=True)
                       for nm, p in zip(cam_names, cam_ports)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if phone_thread:
                phone_thread.join()

            preview_paths = []
            n_ok = 0
            for name, port in zip(cam_names, cam_ports):
                path, msg = results.get(port, (None, ""))
                preview_paths.append(path)
                if path:
                    n_ok += 1
                    self._log(f"Test shot: {name} captured.")
                else:
                    self._log(f"⚠ Test shot: {name} FAILED: {msg or 'unknown error'}")

            for pname in phone_names:
                r = phone_results.get(pname, {"ok": False, "error": "no result"})
                if r["ok"]:
                    n_ok += 1
                    preview_paths.append(os.path.join(base, r["path"]))
                    self._log(f"Test shot: phone {pname} captured ({r['width']}×{r['height']}).")
                else:
                    preview_paths.append(None)
                    self._log(f"⚠ Test shot: phone {pname} FAILED: {r.get('error') or 'unknown error'}")

            self._update_preview(preview_paths)
            self._set_status(f"Test shot: {n_ok}/{n_cams + len(phone_names)} cameras")
        finally:
            self.root.after(0, lambda: (self.test_btn.configure(state="normal"),
                                        self.start_btn.configure(state="normal")))

    # ------------------------------------------------------------------ LED control

    def _leds_on(self):
        self._send_led_command(True)

    def _leds_off(self):
        self._send_led_command(False)

    def _send_led_command(self, on):
        """Toggle the capture LEDs over the shared serial connection.

        Independent of the turntable/scan flow, like _test_shot. Refused while
        a scan is running since _run_capture is using the serial port.
        """
        if self.capture_thread and self.capture_thread.is_alive():
            messagebox.showinfo("Busy", "A scan is currently running. Stop it before controlling the LEDs.")
            return
        port = self._turntable_port()
        if not port:
            messagebox.showinfo("No Turntable",
                                "The LEDs are switched by the Arduino. Select its port under "
                                "Turntable Port to control them.")
            return
        if not SERIAL_AVAILABLE:
            messagebox.showerror("pyserial Missing", "Install pyserial to control the LEDs:\n\npip install pyserial")
            return

        self.start_btn.configure(state="disabled")
        self.test_btn.configure(state="disabled")
        self.leds_on_btn.configure(state="disabled")
        self.leds_off_btn.configure(state="disabled")
        threading.Thread(target=self._run_led_command, args=(port, on), daemon=True).start()

    # The Arduino resets whenever the port is opened (DTR toggles), which briefly
    # energises the motor driver. So the app opens the port once, on first use,
    # and keeps that connection for the LED buttons and every scan; it is
    # released when the port selection changes or the window closes.

    def _serial_alive(self):
        """True if the shared connection is open and its device is still there."""
        if self.ser is None or not self.ser.is_open:
            return False
        try:
            self.ser.in_waiting  # raises once the board has been unplugged
            return True
        except (serial.SerialException, OSError):
            return False

    def _get_serial(self, port):
        """Return the shared connection to the Arduino on `port`, opening it if needed.

        Opening resets the board, so this waits for it to reboot, then puts
        the LEDs back if the user had them on (the sketch boots with them off).
        Raises serial.SerialException if the port can't be opened.
        """
        with self._ser_lock:
            if self.ser_port == port and self._serial_alive():
                return self.ser
            self._close_serial()
            self._log(f"Opening {port} (the Arduino reboots once, about {ARDUINO_RESET_S:g}s)…")
            ser = serial.Serial(port, baudrate=115200, timeout=2)
            time.sleep(ARDUINO_RESET_S)
            if not self._wait_for_sketch(ser):
                ser.close()
                raise serial.SerialException(
                    f"{port} opened but the Arduino never answered within "
                    f"{ARDUINO_BOOT_TIMEOUT_S:g}s — check the port and that the sketch is uploaded.")
            ser.reset_input_buffer()
            self.ser, self.ser_port = ser, port
            self._restore_leds(ser)
            return ser

    @staticmethod
    def _wait_for_sketch(ser):
        """Probe with the harmless LEDs-off command until the sketch acks, or give up.

        Bootloader time varies by board and OS driver (a Mega on Windows needs
        longer than a fixed sleep allows); bytes sent before the sketch runs
        are dropped, so keep asking rather than guessing a delay.
        """
        old_timeout, ser.timeout = ser.timeout, 0.5
        try:
            deadline = time.monotonic() + ARDUINO_BOOT_TIMEOUT_S
            while time.monotonic() < deadline:
                ser.write(b'F')
                if ser.read(1) == b'e':
                    time.sleep(0.3)  # let acks from any extra probes land, then discard them
                    ser.reset_input_buffer()
                    return True
            return False
        finally:
            ser.timeout = old_timeout

    def _close_serial(self):
        with self._ser_lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                except (serial.SerialException, OSError):
                    pass
            self.ser = self.ser_port = None

    def _on_port_change(self, *_):
        """Release the Arduino as soon as another port (or none) is selected."""
        scanning = self.capture_thread and self.capture_thread.is_alive()
        if self.ser is not None and not scanning and self._turntable_port() != self.ser_port:
            self._close_serial()

    def close(self):
        """Window close: release the serial port, then quit."""
        self._close_serial()
        self.root.destroy()

    def _restore_leds(self, ser):
        """Re-send the desired LED state on a freshly opened connection.

        The sketch boots with the LEDs off, so a reconnect (e.g. after the
        board was unplugged) would otherwise leave them dark even though the
        user turned them on.
        """
        if not self.leds_on:
            return
        ser.reset_input_buffer()
        ser.write(b'N')
        ack = ser.read(1)
        if ack == b'e':
            self._log("LEDs re-enabled after Arduino reset.")
        else:
            self._log(f"⚠ Could not restore LEDs after the Arduino reset (got {ack!r}) — "
                      f"the scan may run with the lights off.")

    def _run_led_command(self, port, on):
        try:
            with self._ser_lock:
                ser = self._get_serial(port)
                ser.timeout = 2
                ser.reset_input_buffer()
                ser.write(b'N' if on else b'F')
                ack = ser.read(1)  # wait for 'e' acknowledgement (or time out)
            if ack == b'e':
                self.leds_on = on
                self._log(f"LEDs turned {'on' if on else 'off'}.")
            else:
                self._log(f"⚠ LED command sent but no acknowledgement from Arduino "
                          f"(got {ack!r}) — check the board is running the latest sketch.")
        except (serial.SerialException, OSError) as exc:
            self._close_serial()  # reconnect on the next use
            self._log(f"LED serial error: {exc}")
        finally:
            self.root.after(0, lambda: (self.start_btn.configure(state="normal"),
                                        self.test_btn.configure(state="normal"),
                                        self.leds_on_btn.configure(state="normal"),
                                        self.leds_off_btn.configure(state="normal")))

    # ------------------------------------------------------------------ capture loop (background thread)

    def _run_capture(self):
        caps_int = int(self.caps_var.get())
        cam_delay = float(self.delay_var.get())
        port = self._turntable_port()
        base_folder = self.folder_var.get()
        settle = float(self.settle_var.get())
        sides = (1, 2) if self.both_sides_var.get() else (1,)

        # Turntable + LEDs over the shared serial connection (opened here only if
        # nothing has used it yet). No port selected = no turntable: the scan
        # then just fires the cameras every `settle` seconds.
        ser = None
        if port:
            try:
                ser = self._get_serial(port)
            except serial.SerialException as exc:
                self._log(f"Serial port error: {exc}")
                self._set_status("Serial error")
                self._finish(False)
                return
        else:
            self._log("No turntable selected — cameras fire every "
                      f"{settle:g}s; move the object/cameras yourself between shots.")

        # Detect cameras: DSLRs via gphoto2, plus whichever phones are connected now.
        # Phones that connect later are ignored; the scan uses exactly this set.
        self._log("Detecting cameras via gphoto2…")
        cam_ports = self._detect_camera_ports_safe()
        phone_names = self._active_phones()
        n_cams = len(cam_ports)
        cam_names = [f"cam{i + 1}" for i in range(n_cams)]
        clash = sorted(set(cam_names) & set(phone_names))
        if not cam_ports and not phone_names:
            self._log("Camera detection failed: no DSLR found by gphoto2 and no phones connected.")
        elif clash:
            self._log(f"Phone name(s) {', '.join(clash)} collide with the DSLR folders "
                      f"(cam1, cam2, …). Rename the phone(s) (e.g. phone1) and reconnect.")
        if (not cam_ports and not phone_names) or clash:
            self._set_status("Camera error")
            self._finish(False)
            return
        self._log(f"Detected {n_cams} DSLR camera(s):")
        for name, port in zip(cam_names, cam_ports):
            self._log(f"  {name}: {port}")
        if phone_names:
            self._log(f"Using {len(phone_names)} phone(s): {', '.join(phone_names)}"
                      f"  (settle {settle:g}s after each move)")
        self._build_preview_slots(self._camera_labels(n_cams, phone_names))

        # Create output folder tree
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_path = os.path.join(base_folder, run_name)
        side_paths = {}
        for side in sides:
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
        if ser:
            ser.timeout = steps_per_move + 2.5
        delay_desc = "all at once" if cam_delay == 0 else f"{cam_delay:g}s between cameras"
        move_desc = f"Steps per move: {steps_per_move}" if ser else "No turntable"
        self._log(f"Captures: {caps_int}  |  {move_desc}  |  "
                  f"Capture: parallel ({delay_desc})  |  "
                  f"{'both sides' if len(sides) == 2 else 'single side'}")

        # Store images on camera card for speed; download in bulk after each side
        for port in cam_ports:
            subprocess.run(["gphoto2", "--port", port, "--set-config", "capturetarget=1"],
                           capture_output=True)

        total = caps_int * len(sides)
        stopped = False

        for side in sides:
            if self.stop_event.is_set():
                stopped = True
                break

            if side == 2:
                self._ask_flip()
                if self.stop_event.is_set():
                    stopped = True
                    break
                # Handling the tablet often wakes or reloads a phone; don't let side 2
                # silently run without one.
                if phone_names and not self._ensure_phones(phone_names):
                    stopped = True
                    break

            cam_dests = side_paths[side]
            # Phones write straight into the run folder: <run>/side<N>/<phone>/0001.jpg,
            # so the switch to side 2 after the flip is just this path.
            phone_dir = f"{run_name}/side{side}"
            self._log(f"--- Side {side} scan starting ---")

            for i in range(caps_int):
                if self.stop_event.is_set():
                    stopped = True
                    break

                self._set_status(f"Side {side}  —  capture {i + 1} / {caps_int}")

                # Let the tablet stop wobbling and the phones' autofocus settle
                # (with no turntable this is simply the pacing between shots).
                if (phone_names or not ser) and settle:
                    time.sleep(settle)

                # Fire cameras in parallel, offset by the GUI delay; retry misses.
                results, phone_results = self._capture_step(
                    cam_ports, cam_delay, base_folder, phone_dir, f"{i + 1:04d}", phone_names)
                for name, port in zip(cam_names, cam_ports):
                    ok, msg = results[port]
                    if not ok:
                        self._log(f"⚠ {name} capture {i + 1} FAILED: {msg or 'unknown error'}")
                for pname in phone_names:
                    r = phone_results.get(pname, {"ok": False, "error": "no result"})
                    if not r["ok"]:
                        self._log(f"⚠ phone {pname} capture {i + 1} FAILED: "
                                  f"{r.get('error') or 'unknown error'}")
                    elif r.get("warning"):
                        self._log(f"⚠ phone {pname}: {r['warning']}")
                if phone_names:
                    self._update_preview([None] * n_cams + [
                        os.path.join(base_folder, phone_results[p]["path"])
                        if phone_results.get(p, {}).get("ok") else None
                        for p in phone_names])

                # Signal Arduino to advance turntable and wait for 'e' acknowledgement
                if ser:
                    ser.reset_input_buffer()  # drop any stale ack so it can't pass for this move's
                    ser.write(caps_ld.encode())
                    ser.write(steps_ld.encode())
                    ser.read_until(size=1)

                done = (side - 1) * caps_int + (i + 1)
                self._set_progress(done / total * 100)
                self._log(f"Side {side}  capture {i + 1}/{caps_int} done")

            else:
                # Inner loop completed without break → download and clear cameras
                if cam_ports:
                    self._download_side(side, run_path, cam_names, cam_ports, cam_dests, caps_int)
                if phone_names:
                    self._check_phone_files(side, run_path, phone_names, caps_int)

        if ser:
            ser.timeout = 2  # the connection outlives the scan; back to the LED-command timeout

        if stopped:
            self._log("Scan stopped by user.")
            self._set_status("Stopped")
        else:
            self._log("All done! Scanning complete.")
            self._set_status("Complete!")

        self._finish(not stopped)

    def _download_side(self, side, run_path, cam_names, cam_ports, cam_dests, caps_int):
        """Download this side's photos from every DSLR, then clear the cards that verified."""
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

    def _check_phone_files(self, side, run_path, phone_names, caps_int):
        """Phones upload as they shoot, so there is nothing to download; just verify the counts."""
        side_dir = Path(run_path) / f"side{side}"
        counts = {p: len(list((side_dir / p).glob("*.jpg"))) for p in phone_names}
        self._log("  ".join(f"phone {p}: {n} files" for p, n in counts.items())
                  + f"  (expected {caps_int} each)")
        for p, n in counts.items():
            if n < caps_int:
                self._log(f"⚠ phone {p}: only {n}/{caps_int} photos for side {side}")
        self._log(f"Side {side} phone images saved to {side_dir}")

    def _capture_step(self, cam_ports, cam_delay, root, phone_dir, name, phone_names):
        """Fire the DSLRs and the phones at the same moment for one turntable position.

        Returns (dslr_results, phone_results): {port: (ok, msg)} and {phone: {...}}.
        """
        phone_results = {}
        phone_thread = None
        if phone_names:
            phone_thread = threading.Thread(
                target=lambda: phone_results.update(
                    self._phone_shoot(root, phone_dir, name, phone_names)),
                daemon=True)
            phone_thread.start()
        dslr_results = self._capture_all(cam_ports, cam_delay)
        if phone_thread:
            phone_thread.join()
        return dslr_results, phone_results

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
            # Phone-only setups don't need gphoto2, so say this once, not on every refresh/shot.
            if not self._warned_no_gphoto2:
                self._warned_no_gphoto2 = True
                self._log("gphoto2 not installed — DSLR cameras disabled (phones still work).")
            return []
        return re.findall(r"(usb:\d+,\d+)", result.stdout)

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
    app = CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
