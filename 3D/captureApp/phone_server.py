#!/usr/bin/env python3
"""Phone camera hub: use iPhones (any browser) as remote cameras.

Each phone opens the page served at "/", starts its camera, and holds a
Server-Sent-Events stream open at /events. The PC calls POST /api/shoot; the
hub broadcasts a "shoot" event to every connected phone, each phone grabs a
frame and POSTs the JPEG to /api/upload, and /api/shoot returns once every
phone has uploaded (missing phones get one retry).

Files land at   <out>/<dir>/<cam>/<name>.jpg
so a scan with dir="<run>/side1" produces the same <run>/side1/camN/ layout
the capture app writes for the DSLR rig.

Standard library only. Binds to loopback by default; expose it to the phones
with `tailscale serve` (HTTPS is required for camera access in Safari).
"""
import argparse
import json
import mimetypes
import os
import queue
import re
import secrets
import select
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

CAM_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DIR_RE = re.compile(r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*$")  # no dots: no traversal
MAX_UPLOAD = 64 * 1024 * 1024
KEEPALIVE_S = 5  # seconds between SSE keepalive comments
# Camera settings the control page may push to phones (MediaTrackConstraints names).
CAMERA_MODES = ("focusMode", "exposureMode", "whiteBalanceMode")
CAMERA_MODE_VALUES = ("continuous", "manual", "single-shot", "none")
CAMERA_NUMBERS = ("exposureCompensation", "exposureTime", "iso", "colorTemperature",
                  "focusDistance", "brightness", "contrast", "saturation", "sharpness")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class NoPhones(Exception):
    pass


class Phone:
    def __init__(self, cam):
        self.cam = cam
        self.events = queue.Queue()  # (event, data) tuples, or None to end the stream
        self.connected_at = time.time()
        self.info = {}
        self.alive = True


class Hub:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.shoot_lock = threading.Lock()  # one shot at a time
        self.phones = {}  # cam -> Phone
        self.shots = {}   # shot id -> record for the in-flight shot
        self.last = {}    # cam -> most recent upload summary

    # -- phone connections --------------------------------------------------

    def connect(self, cam):
        with self.lock:
            old = self.phones.get(cam)
            if old:
                # Same name from a new connection: newest wins. The old page is
                # told, so two devices sharing a name don't kick each other forever.
                old.alive = False
                old.events.put(("replaced", {}))
                old.events.put(None)
                log(f"{cam}: replaced by a new connection")
            phone = Phone(cam)
            self.phones[cam] = phone
        log(f"{cam}: connected ({len(self.phones)} phone(s))")
        return phone

    def disconnect(self, phone):
        with self.cond:
            phone.alive = False
            if self.phones.get(phone.cam) is phone:
                del self.phones[phone.cam]
                log(f"{phone.cam}: disconnected ({len(self.phones)} phone(s))")
            self.cond.notify_all()

    def set_info(self, cam, info):
        with self.lock:
            phone = self.phones.get(cam)
            if phone:
                phone.info = info

    def connected(self):
        """Names of the phones that are connected right now."""
        with self.lock:
            return sorted(self.phones)

    def status(self):
        with self.lock:
            return {
                "phones": [
                    {"cam": p.cam, "connected_at": p.connected_at, "info": p.info}
                    for p in sorted(self.phones.values(), key=lambda p: p.cam)
                ],
                "last": self.last,
            }

    def _tell(self, event, data, cams):
        """Send an event to the named phones (all connected ones if `cams` is None); returns their names."""
        with self.lock:
            targets = [p for c, p in sorted(self.phones.items()) if cams is None or c in cams]
            if not targets:
                raise NoPhones()
            for p in targets:
                p.events.put((event, data))
            return [p.cam for p in targets]

    def set_zoom(self, zoom, cams=None):
        """Ask phones to change zoom; returns the names asked.

        Fire and forget: each phone clamps to what its camera can do and reports the zoom it
        really ended up with through /api/info, so read that back from status().
        """
        return self._tell("zoom", {"zoom": zoom}, cams)

    def set_camera(self, change, cams=None):
        """Ask phones to change exposure/focus/white balance: {"set": {key: value}, "reset": bool}.

        Same fire-and-forget deal as set_zoom. Which keys a phone accepts depends on its browser;
        it lists them in status() -> info.caps and ignores the rest.
        """
        return self._tell("camera", change, cams)

    # -- shooting -----------------------------------------------------------

    def _missing(self, rec):
        """Cams we are still waiting on (call with the lock held)."""
        return [c for c in rec["want"] if c not in rec["results"] and c in self.phones]

    def shoot(self, dest_dir, name, timeout=10.0, attempts=2, cams=None):
        """Trigger phones and wait for their uploads.

        `cams` is the set of phone names the caller expects (a scan pins the
        phones it started with, so one that joins mid-scan is ignored). Any of
        them that is not connected is reported as failed instead of waited on.
        Default: every phone connected right now.

        Returns {"shot": id, "ok": bool, "results": {cam: {...}}}.
        """
        with self.shoot_lock:
            with self.lock:
                want = sorted(cams) if cams else sorted(self.phones)
                if not want:
                    raise NoPhones()
                shot = secrets.token_hex(6)
                # Snapshot the root: the capture app repoints out_dir between runs.
                rec = {"root": self.out_dir, "dir": dest_dir, "name": name,
                       "want": set(want), "results": {}}
                self.shots[shot] = rec
            t0 = time.monotonic()
            try:
                for attempt in range(attempts):
                    with self.lock:
                        for cam in self._missing(rec):
                            self.phones[cam].events.put(
                                ("shoot", {"shot": shot, "attempt": attempt}))
                    deadline = time.monotonic() + timeout
                    with self.cond:
                        while self._missing(rec):
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            self.cond.wait(remaining)
                        if not self._missing(rec):
                            break
                        if attempt + 1 < attempts:
                            log(f"shot {shot}: retrying {self._missing(rec)}")
            finally:
                with self.lock:
                    self.shots.pop(shot, None)
            results = {}
            for cam in sorted(rec["want"]):
                results[cam] = rec["results"].get(cam) or {
                    "ok": False,
                    "error": "phone disconnected" if cam not in self.phones else "no upload",
                }
            ok = all(r["ok"] for r in results.values())
            log(f"shot {shot} -> {dest_dir}/*/{name}.jpg  "
                f"{sum(r['ok'] for r in results.values())}/{len(results)} ok  "
                f"({time.monotonic() - t0:.2f}s)")
            return {"shot": shot, "ok": ok, "results": results}

    def save_upload(self, cam, shot, data, width, height):
        """Store an uploaded JPEG. Returns the result dict, or None if the shot is unknown."""
        with self.lock:
            rec = self.shots.get(shot)
            if rec is None or cam not in rec["want"]:
                return None
            dest = rec["root"] / rec["dir"] / cam / f"{rec['name']}.jpg"
            prev = self.last.get(cam)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{secrets.token_hex(3)}.part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)  # atomic, so a retry's duplicate upload can't tear the file
        result = {
            "ok": True,
            "path": dest.relative_to(rec["root"]).as_posix(),
            "bytes": len(data),
            "width": width,
            "height": height,
        }
        if prev and (prev.get("width"), prev.get("height")) != (width, height):
            result["warning"] = (f"frame size changed {prev.get('width')}x{prev.get('height')} "
                                 f"-> {width}x{height} (rotated, or zoom/lens changed?)")
            log(f"{cam}: {result['warning']}")
        with self.cond:
            rec["results"][cam] = result
            self.last[cam] = {**result, "at": time.time()}
            self.cond.notify_all()
        return result


HUB = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PhoneCamHub/1.0"

    # -- plumbing -----------------------------------------------------------

    def log_message(self, fmt, *args):
        if self.path.startswith(("/api/status", "/captures/")):
            return  # polled constantly; keep the console readable
        log(f"{self.address_string()} {fmt % args}")

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, code, msg):
        # Any request body we did not read would corrupt the next keep-alive request.
        self.close_connection = True
        self._json(code, {"ok": False, "error": msg})

    def _read_body(self, limit):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if n < 0 or n > limit:
            return None
        data = bytearray()
        while len(data) < n:
            chunk = self.rfile.read(min(65536, n - len(data)))
            if not chunk:
                return None
            data += chunk
        return bytes(data)

    def _read_json(self):
        body = self._read_body(1024 * 1024)
        if body is None:
            return None
        try:
            obj = json.loads(body or b"{}")
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None

    def _page(self, filename):
        try:
            self._send(200, (STATIC / filename).read_bytes(), "text/html; charset=utf-8")
        except OSError:
            self._error(404, "not found")

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._page("phone.html")
        elif url.path == "/control":
            self._page("control.html")
        elif url.path == "/events":
            self._events(parse_qs(url.query).get("cam", [""])[0])
        elif url.path == "/api/status":
            self._json(200, HUB.status())
        elif url.path.startswith("/captures/"):
            self._capture_file(unquote(url.path[len("/captures/"):]))
        else:
            self._error(404, "not found")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/upload":
            self._upload(parse_qs(url.query))
        elif url.path == "/api/info":
            self._info()
        elif url.path == "/api/shoot":
            self._shoot()
        elif url.path == "/api/zoom":
            self._zoom()
        elif url.path == "/api/camera":
            self._camera()
        else:
            self._error(404, "not found")

    def _capture_file(self, rel):
        root = HUB.out_dir.resolve()
        try:
            path = (root / rel).resolve()
            path.relative_to(root)
            data = path.read_bytes()
        except (ValueError, OSError):
            return self._error(404, "not found")
        self._send(200, data, mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def _events(self, cam):
        if not CAM_RE.match(cam):
            return self._error(400, "bad cam name")
        phone = HUB.connect(cam)
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self._sse("hello", {"cam": cam})
            quiet = 0
            while phone.alive:
                try:
                    item = phone.events.get(timeout=1)
                except queue.Empty:
                    # A vanished phone shows up as EOF on the socket (the client never
                    # sends anything after its request), far sooner than a write fails.
                    if self._peer_closed():
                        break
                    quiet += 1
                    if quiet >= KEEPALIVE_S:  # keep proxies from idling the stream out
                        self.wfile.write(b": keepalive\n\n")
                        quiet = 0
                    continue
                quiet = 0
                if item is None:
                    break
                self._sse(*item)
        except OSError:
            pass  # phone went away
        finally:
            HUB.disconnect(phone)

    def _peer_closed(self):
        readable, _, _ = select.select([self.connection], [], [], 0)
        if not readable:
            return False
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _sse(self, event, data):
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())

    def _info(self):
        body = self._read_json()
        if body is None or not CAM_RE.match(str(body.get("cam", ""))):
            return self._error(400, "bad request")
        info = {k: body[k] for k in ("width", "height", "ua", "settings", "caps") if k in body}
        HUB.set_info(body["cam"], info)
        self._json(200, {"ok": True})

    def _upload(self, qs):
        cam = qs.get("cam", [""])[0]
        shot = qs.get("shot", [""])[0]
        try:
            width = int(qs.get("w", ["0"])[0])
            height = int(qs.get("h", ["0"])[0])
        except ValueError:
            return self._error(400, "bad size")
        if not CAM_RE.match(cam) or not re.fullmatch(r"[0-9a-f]{12}", shot):
            return self._error(400, "bad request")
        data = self._read_body(MAX_UPLOAD)
        if data is None or data[:2] != b"\xff\xd8":
            return self._error(400, "expected a JPEG body")
        result = HUB.save_upload(cam, shot, data, width, height)
        if result is None:
            # Late upload for a shot that already finished (or a phone we did not ask).
            return self._json(410, {"ok": False, "error": "unknown or finished shot"})
        self._json(200, result)

    def _zoom(self):
        body = self._read_json()
        if body is None:
            return self._error(400, "bad request")
        try:
            zoom = float(body["zoom"])
        except (KeyError, TypeError, ValueError):
            return self._error(400, "zoom must be a number")
        if not 0.1 <= zoom <= 20:  # also rejects nan/inf; each phone clamps to its own range
            return self._error(400, "zoom out of range (0.1 to 20)")
        cams = body.get("cams")
        if cams is not None and not (isinstance(cams, list) and all(CAM_RE.match(str(c)) for c in cams)):
            return self._error(400, "cams must be a list of phone names")
        try:
            told = HUB.set_zoom(zoom, set(cams) if cams is not None else None)
        except NoPhones:
            return self._json(409, {"ok": False, "error": "no matching phone connected"})
        log(f"zoom {zoom:g}x -> {', '.join(told)}")
        self._json(200, {"ok": True, "zoom": zoom, "cams": told})

    def _camera(self):
        body = self._read_json()
        if body is None:
            return self._error(400, "bad request")
        cams = body.get("cams")
        if cams is not None and not (isinstance(cams, list) and all(CAM_RE.match(str(c)) for c in cams)):
            return self._error(400, "cams must be a list of phone names")
        settings = body.get("set", {})
        if not isinstance(settings, dict):
            return self._error(400, "set must be an object")
        clean = {}
        for key, value in settings.items():
            if key in CAMERA_MODES and value in CAMERA_MODE_VALUES:
                clean[key] = value
            elif (key in CAMERA_NUMBERS and isinstance(value, (int, float)) and not isinstance(value, bool)
                  and abs(value) <= 1e9):  # a JSON number, so never nan/inf; each phone clamps to its range
                clean[key] = value
            else:
                return self._error(400, f"unsupported setting {key!r}")
        reset = body.get("reset", False)
        if not isinstance(reset, bool) or not (clean or reset):
            return self._error(400, "send `set` and/or `reset: true`")
        try:
            told = HUB.set_camera({"set": clean, "reset": reset}, set(cams) if cams is not None else None)
        except NoPhones:
            return self._json(409, {"ok": False, "error": "no matching phone connected"})
        log(f"camera {'reset ' if reset else ''}{clean or ''} -> {', '.join(told)}")
        self._json(200, {"ok": True, "cams": told})

    def _shoot(self):
        body = self._read_json()
        if body is None:
            return self._error(400, "bad request")
        dest_dir = str(body.get("dir", "_test_shots"))
        name = str(body.get("name", "latest"))
        if not DIR_RE.match(dest_dir) or not NAME_RE.match(name):
            return self._error(400, "bad dir/name")
        try:
            timeout = min(max(float(body.get("timeout", 10)), 1), 60)
            attempts = min(max(int(body.get("attempts", 2)), 1), 3)
        except (TypeError, ValueError):
            return self._error(400, "bad timeout/attempts")
        try:
            result = HUB.shoot(dest_dir, name, timeout, attempts)
        except NoPhones:
            return self._json(409, {"ok": False, "error": "no phones connected"})
        self._json(200, result)


def tailscale_dns_name():
    try:
        out = subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                             text=True, timeout=5).stdout
        return (json.loads(out)["Self"]["DNSName"] or "").rstrip(".") or None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


def make_server(host, port, out_dir):
    """Create the hub and its HTTP server (not yet serving). Returns (hub, server).

    Raises OSError if the port is taken. The capture app calls this and runs
    server.serve_forever() on a daemon thread; main() below runs it standalone.
    """
    global HUB
    server = ThreadingHTTPServer((host, port), Handler)  # bind first: fail before touching HUB
    HUB = Hub(Path(out_dir))  # folders are created as photos arrive
    return HUB, server


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; tailscale serve proxies to it)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=HERE / "captures", help="output folder")
    args = ap.parse_args()

    _, server = make_server(args.host, args.port, args.out)

    dns = tailscale_dns_name()
    log(f"Listening on http://{args.host}:{args.port}   output: {args.out}")
    log(f"Control page (this PC):  http://localhost:{args.port}/control")
    if dns:
        log(f"Phones open:             https://{dns}/")
        log(f"  (needs:  tailscale serve --bg {args.port})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("bye")


if __name__ == "__main__":
    main()
