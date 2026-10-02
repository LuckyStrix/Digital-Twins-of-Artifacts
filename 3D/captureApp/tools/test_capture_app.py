#!/usr/bin/env python3
"""End-to-end test of captureApp with fake phones and a fake Arduino (no hardware).

    python3 tools/test_capture_app.py

Needs a display (it creates the real Tk app, hidden) and Pillow. The app's hub
runs on a free port, so this can run while a real app or phone_server.py is up.

What it does:
  1. starts two fake phones and a fake Arduino on a pty,
  2. runs Test Shot,
  3. runs a 4-capture, 2-side scan; while the "flip the tablet" dialog is up it
     drops one phone, then reconnects it when the "phone disconnected" dialog appears,
  4. checks the files, the turntable commands and the log,
  5. runs a second scan with no turntable port selected (phone-only, no Arduino)
     and checks it takes every photo without touching the Arduino,
  6. runs a single-side scan (both-sides box unticked): one side, no flip prompt,
  7. checks the LED buttons and scans share one serial connection (the Arduino
     resets each time the port is opened), that deselecting the port releases it,
     and that a reconnect puts the LEDs back on,
  8. checks the app still imports when pyserial is not installed.

Tk runs its normal main loop on the main thread (the app's worker threads need
that); the test steps run on a second thread and poke the app via root.after.
"""
import os
import pty
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Never touch real cameras: the app shells out to gphoto2, which would fire, download
# from and (after a verified download) clear any camera plugged into this machine.
# Put a do-nothing gphoto2 first on PATH so detection finds no DSLRs.
_stub_dir = tempfile.mkdtemp(prefix="stub_gphoto2_")
_stub = Path(_stub_dir) / "gphoto2"
_stub.write_text("#!/bin/sh\nexit 0\n")
_stub.chmod(0o755)
os.environ["PATH"] = _stub_dir + os.pathsep + os.environ["PATH"]

sys.path.insert(0, str(ROOT))
import capture_app_parallel as capp  # noqa: E402
from PIL import Image  # noqa: E402

# Count real opens of the serial port: each one resets a real Arduino.
opens = []
_real_serial = capp.serial.Serial


def _counting_serial(*args, **kwargs):
    opens.append(args)
    return _real_serial(*args, **kwargs)


capp.serial.Serial = _counting_serial

PHONES = ["phoneA", "phoneB"]
CAPTURES = 4
WIDTH, HEIGHT = 800, 600


class FakeArduino(threading.Thread):
    """Speaks the sketch's protocol: 'N'/'F' toggle LEDs, 8 bytes = a move; both ack with 'e'."""

    def __init__(self):
        super().__init__(daemon=True)
        self.master, slave = pty.openpty()
        tty.setraw(slave)
        self._slave = slave  # keep open
        self.port = os.ttyname(slave)
        self.moves, self.leds = [], []

    def run(self):
        buf = b""
        while True:
            try:
                buf += os.read(self.master, 64)
            except OSError:
                return
            while buf:
                if buf[:1] in (b"N", b"F"):
                    self.leds.append(buf[:1])
                    buf = buf[1:]
                elif len(buf) >= 8:
                    self.moves.append(buf[:8].decode())
                    buf = buf[8:]
                else:
                    break
                os.write(self.master, b"e")


def spawn_phone(name):
    return subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "fake_phone.py"), name,
         "--url", f"http://127.0.0.1:{capp.PHONE_PORT}",
         "--width", str(WIDTH), "--height", str(HEIGHT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait(cond, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def run_test(root, app, arduino, tmp, problems):
    procs = {n: spawn_phone(n) for n in PHONES}

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
        if not cond:
            problems.append(msg)

    if shutil.which("gphoto2") != str(_stub):
        problems.append("gphoto2 stub is not first on PATH; refusing to run against real cameras")
        root.after(0, root.quit)
        return

    def on_tk(fn):
        root.after(0, fn)

    def side_files(side, phone):
        runs = sorted(d for d in tmp.iterdir() if d.name[:2] == "20")  # latest run last
        return sorted((runs[-1] / f"side{side}" / phone).glob("*.jpg")) if runs else []

    dialogs, state = [], {}

    # These run on the Tk thread when the app raises a dialog.
    def fake_flip(title, msg, **kw):
        if title != "Flip Object":
            dialogs.append(("info", msg))
            return
        # The app says "flip the tablet over". Side 1 must be complete and side 2 still empty.
        state["at_flip"] = {p: (len(side_files(1, p)), len(side_files(2, p))) for p in PHONES}
        if state.get("drop_on_flip", True):
            procs["phoneB"].kill()  # a phone drops while the operator handles the tablet
            wait(lambda: "phoneB" not in app.hub.connected(), 10)
        dialogs.append(("flip", msg))

    def fake_retry(title, msg, **kw):
        dialogs.append(("retry", msg))
        procs["phoneB"] = spawn_phone("phoneB")
        wait(lambda: "phoneB" in app.hub.connected(), 10)
        return True

    capp.messagebox.showinfo = fake_flip
    capp.messagebox.askretrycancel = fake_retry
    capp.messagebox.showerror = lambda t, m, **kw: dialogs.append(("error", m))

    try:
        print("phones connect to the app's hub")
        check(wait(lambda: sorted(app.hub.connected()) == PHONES, 15),
              f"both phones connected: {app.hub.connected()}")
        wait(lambda: "(800×600)" in app.phones_var.get(), 5)
        check("(800×600)" in app.phones_var.get(), f"UI shows phone list: {app.phones_var.get()!r}")

        print("test shot")
        on_tk(app._test_shot)
        wait(lambda: app.status_var.get().startswith("Test shot:"), 30)
        check(app.status_var.get() == "Test shot: 2/2 cameras", f"status: {app.status_var.get()!r}")
        for p in PHONES:
            f = tmp / "_test_shots" / p / "latest.jpg"
            check(f.is_file() and Image.open(f).size == (WIDTH, HEIGHT), f"_test_shots/{p}/latest.jpg")
        wait(lambda: str(app.start_btn["state"]) == "normal", 5)

        print("scan (2 sides, phone drops during the flip)")
        on_tk(app._start_capture)
        wait(lambda: app.capture_thread is not None, 5)
        finished = wait(lambda: not app.capture_thread.is_alive(), 120)
        check(finished, "scan thread finished")
        time.sleep(0.5)
        log = app.log_text.get("1.0", "end")

        check(state.get("at_flip") == {p: (CAPTURES, 0) for p in PHONES},
              f"at the flip: side1 full, side2 empty {state.get('at_flip')}")
        for side in (1, 2):
            for p in PHONES:
                files = side_files(side, p)
                good = len(files) == CAPTURES and all(Image.open(f).size == (WIDTH, HEIGHT) for f in files)
                check(good, f"side{side}/{p}: {len(files)} valid photos "
                            f"({', '.join(f.name for f in files)})")
        kinds = [d[0] for d in dialogs]
        check(kinds == ["flip", "retry"] and "phoneB" in dialogs[1][1],
              f"dialogs: flip, then retry naming phoneB {kinds}")
        check(arduino.moves == [f"{CAPTURES:04d}{int(32 * 100 / CAPTURES):04d}"] * (2 * CAPTURES),
              f"turntable moved {len(arduino.moves)}x with the right packet ({set(arduino.moves)})")
        check(f"phone phoneA: {CAPTURES} files" in log and f"phone phoneB: {CAPTURES} files" in log,
              "log reports per-phone file counts")
        check("All done! Scanning complete." in log, "scan completed")
        if "FAILED" in log or "⚠" in log:
            print("  log warnings:\n" + "\n".join("    " + l for l in log.splitlines() if "⚠" in l or "FAILED" in l))

        print("scan with no turntable (phones only, no Arduino)")
        moves_before, leds_before = list(arduino.moves), list(arduino.leds)
        dialogs.clear()
        state.update(drop_on_flip=False, at_flip=None)
        time.sleep(1.1)  # run folders are named to the second; keep this run's name distinct
        app.log_text.configure(state="normal")
        app.log_text.delete("1.0", "end")
        app.log_text.configure(state="disabled")
        on_tk(lambda: app.port_var.set(capp.NO_TURNTABLE))
        time.sleep(0.2)
        check(app._turntable_port() == "", "port set to no-turntable resolves to no port")
        on_tk(app._leds_on)  # LEDs need the Arduino: informs the user, sends nothing
        wait(lambda: any(d[0] == "info" for d in dialogs), 5)
        check([d[0] for d in dialogs] == ["info"], f"LED button explains it needs a turntable {dialogs}")
        dialogs.clear()
        on_tk(app._start_capture)
        wait(lambda: app.capture_thread is not None and app.capture_thread.is_alive(), 5)
        finished = wait(lambda: not app.capture_thread.is_alive(), 120)
        check(finished, "no-turntable scan thread finished")
        time.sleep(0.5)
        log = app.log_text.get("1.0", "end")
        check(state.get("at_flip") == {p: (CAPTURES, 0) for p in PHONES},
              f"at the flip: side1 full, side2 empty {state.get('at_flip')}")
        for side in (1, 2):
            for p in PHONES:
                files = side_files(side, p)
                good = len(files) == CAPTURES and all(Image.open(f).size == (WIDTH, HEIGHT) for f in files)
                check(good, f"side{side}/{p}: {len(files)} valid photos")
        check([d[0] for d in dialogs] == ["flip"], f"dialogs: only the flip prompt {[d[0] for d in dialogs]}")
        check(arduino.moves == moves_before and arduino.leds == leds_before,
              "nothing was sent to the Arduino")
        check("No turntable selected" in log and "All done! Scanning complete." in log,
              "log says no turntable, scan completed")
        check("FAILED" not in log and "Serial" not in log, "no failures or serial errors in the log")

        print("single-side scan (Arduino attached)")
        moves_before = len(arduino.moves)
        dialogs.clear()
        state.update(at_flip=None)
        time.sleep(1.1)  # distinct run folder name
        on_tk(lambda: (app.port_var.set(arduino.port), app.both_sides_var.set(False)))
        time.sleep(0.2)
        on_tk(app._start_capture)
        wait(lambda: app.capture_thread is not None and app.capture_thread.is_alive(), 5)
        finished = wait(lambda: not app.capture_thread.is_alive(), 120)
        check(finished, "single-side scan thread finished")
        time.sleep(0.5)
        log = app.log_text.get("1.0", "end")
        for p in PHONES:
            files = side_files(1, p)
            check(len(files) == CAPTURES, f"side1/{p}: {len(files)} photos")
            check(side_files(2, p) == [], f"side2/{p}: no photos")
        run = sorted(d for d in tmp.iterdir() if d.name[:2] == "20")[-1]
        check(not (run / "side2").exists(), "no side2 folder created")
        check(dialogs == [], f"no flip prompt {[d[0] for d in dialogs]}")
        check(len(arduino.moves) - moves_before == CAPTURES,
              f"turntable moved {len(arduino.moves) - moves_before}x (one side only)")
        check("single side" in log and "All done! Scanning complete." in log
              and app.pct_label.cget("text").strip() == "100 %", "single-side scan completed at 100%")

        print("one serial connection for LED buttons and scans")
        opens0, leds0, moves0 = len(opens), len(arduino.leds), len(arduino.moves)
        check(app.ser is not None and app.ser_port == arduino.port, "connection stays open after a scan")
        for on in (True, False, True):
            n = len(arduino.leds)
            on_tk(app._leds_on if on else app._leds_off)
            wait(lambda: len(arduino.leds) == n + 1, 10)
            wait(lambda: str(app.leds_on_btn["state"]) == "normal", 5)
        check(arduino.leds[leds0:] == [b"N", b"F", b"N"], f"LED commands sent {arduino.leds[leds0:]}")
        check(len(opens) == opens0, f"LED clicks reused the connection ({len(opens) - opens0} reopens)")
        time.sleep(1.1)  # distinct run folder name
        on_tk(app._start_capture)
        wait(lambda: app.capture_thread is not None and app.capture_thread.is_alive(), 5)
        check(wait(lambda: not app.capture_thread.is_alive(), 120), "scan after LED clicks finished")
        time.sleep(0.5)
        check(len(opens) == opens0, f"the scan reused the connection ({len(opens) - opens0} reopens)")
        check(len(arduino.moves) - moves0 == CAPTURES, f"turntable moved {len(arduino.moves) - moves0}x")
        check(arduino.leds[leds0:] == [b"N", b"F", b"N"], "scan sent no extra LED commands")

        print("release and reconnect")
        on_tk(lambda: app.port_var.set(capp.NO_TURNTABLE))
        wait(lambda: app.ser is None, 5)
        check(app.ser is None, "deselecting the port releases it")
        opens1, leds1 = len(opens), len(arduino.leds)
        on_tk(lambda: app.port_var.set(arduino.port))
        time.sleep(1.1)  # distinct run folder name
        on_tk(app._start_capture)
        wait(lambda: app.capture_thread is not None and app.capture_thread.is_alive(), 5)
        check(wait(lambda: not app.capture_thread.is_alive(), 120), "scan after reconnect finished")
        time.sleep(0.5)
        check(len(opens) == opens1 + 1, f"reconnect opened the port once ({len(opens) - opens1})")
        check(arduino.leds[leds1:] == [b"N"], f"LEDs put back on after the reboot {arduino.leds[leds1:]}")

        print("app imports without pyserial")
        blocked = ("import sys; sys.modules.update({'serial': None, 'serial.tools': None, "
                   "'serial.tools.list_ports': None}); sys.path.insert(0, %r); "
                   "import capture_app_parallel as c; assert c.SERIAL_AVAILABLE is False"
                   % str(ROOT))
        r = subprocess.run([sys.executable, "-c", blocked], capture_output=True, text=True)
        check(r.returncode == 0, f"import without pyserial {r.stderr.strip()[-200:]}")
    except Exception as exc:  # report, don't hang the Tk loop
        import traceback
        traceback.print_exc()
        problems.append(f"test crashed: {exc!r}")
    finally:
        for pr in procs.values():
            pr.kill()
        root.after(0, root.quit)


def main():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        sys.exit(f"SKIP: no display available ({exc})")
    root.withdraw()

    # Use a free port for the app's hub so the test can run next to a real app
    # (or phone_server.py) that already holds 8000; the fake phones follow capp.PHONE_PORT.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        capp.PHONE_PORT = sock.getsockname()[1]

    tmp = Path(tempfile.mkdtemp(prefix="phonecap_test_"))
    arduino = FakeArduino()
    arduino.start()
    app = capp.CaptureApp(root)
    if app.hub is None:
        sys.exit(f"FAIL: hub did not start: {app.hub_error}")

    app.folder_var.set(str(tmp))
    app.port_var.set(arduino.port)
    app.caps_var.set(str(CAPTURES))
    app.delay_var.set("0")
    app.settle_var.set("0.05")

    problems = []
    threading.Thread(target=run_test, args=(root, app, arduino, tmp, problems), daemon=True).start()
    root.mainloop()
    root.destroy()

    if problems:
        print(f"\nFAILED ({len(problems)}) — output kept in {tmp}")
        sys.exit(1)
    print(f"\nALL PASSED (output in {tmp})")


if __name__ == "__main__":
    main()
