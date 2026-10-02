#!/usr/bin/env python3
"""Simulated phone(s) for testing the hub without a real iPhone.

    python3 tools/fake_phone.py cam1 cam2 cam3
    python3 tools/fake_phone.py cam1 --drop-first     # skip the first attempt of each shot (tests retry)
    python3 tools/fake_phone.py cam1 --delay 2        # slow phone

Each name becomes one thread that behaves like phone.html: opens /events,
and on every "shoot" event uploads a generated JPEG. It also obeys "zoom" events like a phone
without sensor zoom: 1x-4x, with the frame cropped to match, and "camera" events (exposure
compensation, exposure mode, shutter, ISO). Needs Pillow.
"""
import argparse
import http.client
import io
import json
import threading
import time
from urllib.parse import urlparse

from PIL import Image, ImageDraw


def make_jpeg(cam, shot, size):
    img = Image.new("RGB", size, (30, 60, 90))
    d = ImageDraw.Draw(img)
    d.text((20, 20), f"{cam}  shot {shot}", fill=(255, 255, 255))
    d.rectangle((10, 10, size[0] - 10, size[1] - 10), outline=(255, 200, 0), width=4)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


ZOOM_MIN, ZOOM_MAX = 1.0, 4.0  # like a phone with no sensor zoom: a centre crop up to 4x
# Camera controls this phone pretends its browser offers (what an Android Chrome track would list).
CAPS = {"exposureCompensation": {"min": -2, "max": 2, "step": 0.1}, "exposureMode": ["continuous", "manual"],
        "exposureTime": {"min": 3, "max": 2047, "step": 1}, "iso": {"min": 50, "max": 800, "step": 1}}
AUTO = {"exposureCompensation": 0, "exposureMode": "continuous", "exposureTime": 333, "iso": 100}


def run(base, cam, size, delay, drop_first):
    u = urlparse(base)
    Conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    zoom = 1.0
    cam_settings = dict(AUTO)

    def apply_camera(msg):
        """Like phone.html's setCamera/resetCamera: clamp, ignore what isn't offered, manual for shutter/ISO."""
        if msg.get("reset"):
            cam_settings.update(AUTO)
        for k, v in (msg.get("set") or {}).items():
            c = CAPS.get(k)
            if isinstance(c, list) and v in c:
                cam_settings[k] = v
            elif isinstance(c, dict) and isinstance(v, (int, float)):
                cam_settings[k] = min(c["max"], max(c["min"], v))
                if k in ("iso", "exposureTime"):
                    cam_settings["exposureMode"] = "manual"
            else:
                print(f"[{cam}] ignoring {k}={v!r}", flush=True)
        if cam_settings["exposureMode"] != "manual":  # back to auto: no manual shutter/ISO
            cam_settings["exposureTime"], cam_settings["iso"] = AUTO["exposureTime"], AUTO["iso"]

    def report():
        """Tell the hub the frame size, zoom and camera settings, like phone.html's reportInfo()."""
        c = Conn(u.hostname, u.port, timeout=10)
        c.request("POST", "/api/info", headers={"Content-Type": "application/json"},
                  body=json.dumps({"cam": cam, "width": round(size[0] / zoom), "height": round(size[1] / zoom),
                                   "settings": {"zoom": zoom, "digitalZoom": zoom, "zoomMin": ZOOM_MIN,
                                                "zoomMax": ZOOM_MAX, "zoomStep": 0.1, **cam_settings},
                                   "caps": CAPS}))
        c.getresponse().read()
        c.close()

    stream = Conn(u.hostname, u.port, timeout=60)
    stream.request("GET", f"/events?cam={cam}")
    resp = stream.getresponse()
    print(f"[{cam}] connected: HTTP {resp.status}", flush=True)
    event = None
    while True:
        line = resp.readline()
        if not line:
            print(f"[{cam}] stream closed", flush=True)
            return
        line = line.decode().rstrip("\r\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:])
            if event == "hello":  # like phone.html: report the frame size and zoom so the PC can show them
                report()
            elif event == "zoom":
                zoom = min(ZOOM_MAX, max(ZOOM_MIN, float(data["zoom"])))
                print(f"[{cam}] zoom -> {zoom:g}x", flush=True)
                report()
            elif event == "camera":
                apply_camera(data)
                print(f"[{cam}] camera -> {cam_settings}", flush=True)
                report()
            elif event == "shoot":
                if drop_first and data.get("attempt") == 0:
                    print(f"[{cam}] dropping shot {data['shot']} attempt 0", flush=True)
                    continue
                time.sleep(delay)
                w, h = round(size[0] / zoom), round(size[1] / zoom)  # a centre crop, like the real page
                jpg = make_jpeg(cam, data["shot"], (w, h))
                c = Conn(u.hostname, u.port, timeout=60)
                c.request("POST", f"/api/upload?cam={cam}&shot={data['shot']}&w={w}&h={h}",
                          body=jpg, headers={"Content-Type": "image/jpeg"})
                r = c.getresponse()
                print(f"[{cam}] uploaded {data['shot']} -> HTTP {r.status}", flush=True)
                r.read()
                c.close()
            elif event == "replaced":
                print(f"[{cam}] replaced by another connection", flush=True)
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cams", nargs="+")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds before each upload")
    ap.add_argument("--drop-first", action="store_true", help="ignore attempt 0 of every shot")
    args = ap.parse_args()
    threads = [threading.Thread(target=run, daemon=True,
                                args=(args.url, c, (args.width, args.height), args.delay, args.drop_first))
               for c in args.cams]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
