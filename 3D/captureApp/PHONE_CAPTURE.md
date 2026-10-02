# Phone capture (iPhones as remote cameras)

Experiment: use iPhones as remote cameras for the 3D capture rig, over Tailscale.

```
capture_app_parallel.py  the repo's capture GUI, extended to drive phones (see below)
(turntable firmware is in ../arduinoCode)
phone_server.py                     hub: serves the pages, broadcasts "shoot", collects uploads
static/phone.html                   page each iPhone opens (camera + upload)
static/control.html                 standalone PC page: connected phones, Test shot, sequence
tools/fake_phone.py                 simulated phones, for testing without an iPhone
tools/test_capture_app.py           end-to-end test of the capture app (fake phones + fake Arduino)
captures/                           output of the standalone hub (created on first shot)
```

Standard library only; nothing to `pip install` for the hub. The capture app needs `python3-tk` and
`pillow` (for previews). `pyserial` (turntable + LEDs) and `gphoto2` (DSLRs) are optional: a phone-only
setup needs neither. `tools/fake_phone.py` needs Pillow. 

## Scan with the turntable (capture app + phones)

```bash
tailscale serve --bg 8000                      # once
python3 capture_app_parallel.py     # the app runs the phone hub itself
```

1. On each iPhone open the **Phone URL** shown in the app's "Phone Cameras" panel, name it
   (`phone1`, `phone2`, …), tap **Start camera**. Give each phone a different name, and don't
   use `cam1`, `cam2`, … (those are the DSLR folders; the app refuses a clash).
2. The panel lists connected phones with their frame size. Tick **Use phones in scan**.
3. **Test Shot** fires every DSLR *and* phone and shows all previews.
4. Pick the **Turntable Port** and **Num Captures**, then **Start**. Each turntable position fires the
   DSLRs and phones together, then the Arduino steps. Photos land in
   `<Output Folder>/<run>/side1/<phone>/0001.jpg …`, the same layout as the DSLRs.
5. When side 1 finishes the app says **flip the tablet**. Flip it and click OK: phones switch
   to `side2/` automatically. If a phone dropped while you handled the tablet (screen locked,
   page reloaded), the app pauses with **Retry / Cancel** until it's reconnected.

For a single-side scan, untick **Scan both sides**: the app scans side 1 only, with no flip prompt
and no `side2/` folder.

DSLRs and the turntable are both optional. With no DSLR attached the scan runs on phones alone,
and with both it uses both. Leave **Turntable Port** on `(none — no turntable)` (the default unless
an Arduino-looking port is found) to run with no Arduino at all: the app then fires the phones every
**Settle (s)** seconds, N times per side, and nothing advances between shots (you move the object or
phones yourself; the LED buttons need the Arduino and are inactive). The flip prompt between sides
still applies. The phones used are the ones connected when you press Start; one that joins mid-scan
is ignored. **Settle (s)** waits before each shot so autofocus can settle (with no turntable it is
the time between shots).

**Serial port:** opening it resets the Arduino (and briefly energises the motor driver), so the app
opens it once, on the first LED click or Start, and reuses that connection for the LED buttons and
every scan. It is released when you pick another port (or `(none…)`) or close the window. While it's
held, the Arduino IDE can't upload or open the serial monitor, so close the app or deselect the port
first. If the board is unplugged the app reconnects on the next use, which resets it once and puts
the LEDs back on if they were on.

Don't run `phone_server.py` at the same time: both want port 8000 (the app says so in its
phone panel if it can't start its hub).

## Standalone hub (no turntable)

The original test setup, for trying phones without the capture app:

## Run it

```bash
python3 phone_server.py                # listens on 127.0.0.1:8000
tailscale serve --bg 8000              # HTTPS on your tailnet -> the hub (one-time; `tailscale serve reset` undoes it)
```

- **PC:** open <http://localhost:8000/control>
- **Each iPhone (Safari):** open `https://<this-pc>.<tailnet>.ts.net/` (the server prints the exact URL at startup),
  name the camera (`cam1`, `cam2`, …), tap **Start camera**, allow camera access.
  Prop the phone up with the screen on and the page in the foreground.

Then **Test shot** on the control page fires every connected phone and shows the frames.
**Run sequence** fires N shots at an interval, standing in for the turntable.

Safari only allows camera access on HTTPS, which is why `tailscale serve` is needed (it provides a real
certificate). Plain `http://<tailscale-ip>:8000` will not work.

## Output layout

`captures/<dir>/<cam>/<name>.jpg` — a sequence writes `captures/<run>/side<N>/<cam>/0001.jpg …`,
the same `side1/camN/` layout the DSLR rig produces, so it can be fed to `3D/modelingPipeline`.
Test shots overwrite `captures/_test_shots/<cam>/latest.jpg`.

## HTTP API (for wiring into the capture app later)

| Endpoint | |
|---|---|
| `POST /api/shoot` `{"dir": "run/side1", "name": "0007", "timeout": 10, "attempts": 2}` | Trigger all phones; blocks until every upload lands (missing phones are retried). Returns `{"ok", "shot", "results": {cam: {ok, path, bytes, width, height}}}`. 409 if no phones are connected. |
| `POST /api/zoom` `{"zoom": 2, "cams": ["phone1"]}` | Ask phones to zoom (`cams` omitted = all connected). Fire and forget: returns `{"ok", "zoom", "cams"}` for the phones asked; each phone clamps to its range and reports what it applied in `status` → `info.settings.zoom` (+ `digitalZoom`, `zoomMin`, `zoomMax`). 400 if `zoom` isn't 0.1–20, 409 if none of the named phones is connected. |
| `POST /api/camera` `{"set": {"exposureCompensation": -1}, "reset": false, "cams": ["phone1"]}` | Change exposure/focus/white balance (`cams` omitted = all). Keys: `focusMode` `exposureMode` `whiteBalanceMode` (`continuous`/`manual`/`single-shot`/`none`) and `exposureCompensation` `exposureTime` `iso` `colorTemperature` `focusDistance` `brightness` `contrast` `saturation` `sharpness` (numbers). `reset: true` returns to auto. Fire and forget like zoom: each phone skips what its browser lacks and reports the result in `status` → `info.settings`, with what it offers in `info.caps`. 400 for an unknown key or bad value, 409 if no named phone is connected. |
| `GET /api/status` | Connected phones + last upload per phone. |
| `GET /events?cam=NAME` | (phones) SSE stream carrying `shoot`, `zoom` and `camera` events. |
| `POST /api/upload?cam=&shot=&w=&h=` | (phones) JPEG body. |

The capture app doesn't use this API; it runs the hub in-process and calls `Hub.shoot(...)` directly.

## Testing without phones or hardware

```bash
python3 tools/test_capture_app.py    # needs a display; uses a free port, so it can run beside a live app
```

Runs the real capture app against two fake phones and a fake Arduino (a pty), including a phone
dropping during the flip, then a second scan with no turntable port (nothing may reach the Arduino)
and a check that the app imports without pyserial. It puts a do-nothing `gphoto2` first on `PATH` so it can never fire,
download from or clear a real camera plugged into the machine.

Just the hub:

```bash
python3 phone_server.py &
python3 tools/fake_phone.py cam1 cam2 cam3                 # three well-behaved phones
python3 tools/fake_phone.py cam4 --drop-first --delay 0.3  # exercises the retry path
curl -X POST localhost:8000/api/shoot -d '{"dir":"t/side1","name":"0001"}'
```

## Things to watch on a real phone

- **Frame size:** the on-phone log shows the resolution Safari actually delivered. If you asked for
  4032×3024 and got less, try 3840×2160.
- **Lens and zoom:** under the preview, the lens dropdown lists every camera the phone exposes (on
  iPhones with a telephoto, pick it for true optical 2×) and the zoom buttons/slider set the
  magnification. If the browser exposes sensor zoom the page uses it; if not it falls back to a
  centre crop, which shows in the label as "digital crop, fewer pixels" and shrinks the saved
  frame (2× of 4032×3024 saves 2016×1512). Both choices are remembered per phone. Set them
  before a scan: changing zoom mid-scan changes the frame size and the hub warns. Switching
  lens resets zoom to 1× and clears the focus lock.
  Zoom can also be set from the PC: the **Phone camera settings** section of the control page
  (`http://localhost:8000/control`, served by the capture app too while it runs) sets one phone or
  all of them, and its Zoom column shows what each phone actually applied (a phone clamps to its
  own range and says "crop" when it has no sensor zoom). The lens can only be changed on the phone.
- **Exposure, focus, white balance:** the same section has controls for exposure compensation,
  exposure mode, shutter time, ISO, focus mode/distance, white-balance mode/colour temperature and
  brightness/contrast/saturation/sharpness, plus **Lock exposure / focus / WB** and **Reset to auto**.
  Which of them exist depends on the phone's *browser*: each phone reports what it offers and the
  page builds the controls from that (for "All phones", only what every phone offers). Chrome on
  Android exposes most of them; Safari on iPhone exposes few or none, and then the page says so and
  lists what the browser does report. For a black backdrop that fools auto-exposure into
  overexposing the object, the useful control is **Exposure compensation** (negative = darker), or
  Lock once the object is in frame.
  Manual mode: shutter time, ISO, colour temperature and focus distance only take effect in their
  mode's *manual* setting, so they are tagged "(manual)" and dimmed until it is. Moving one switches
  the mode to manual by itself. The phone applies the mode switch first and the value in a second
  call, because Chrome ignores a shutter/ISO value sent in the same call as the switch. Below the
  controls, the page lists what each phone's browser offers and what it doesn't, and shows a ⚠
  line when a browser accepts a setting but doesn't apply it ("exposureTime: asked 100, got 333").
  A browser that offers only the mode switches (no value sliders) can lock those settings in place
  but can't set values. A lens switch clears all of these. After updating the software,
  restart the app (the hub) **and reload each phone's page**: a phone on an older page ignores
  new commands, and the control page says "reload" for a phone that hasn't reported capabilities.
- **Orientation:** the frame follows how the phone is held. Don't rotate it mid-scan; the hub warns
  if a phone's frame size changes.
- **Focus/exposure:** Safari may not expose manual focus. The page logs `capabilities` and tries
  **Lock focus/exposure**; the log says what worked. If focus hunts between frames, that is the ceiling of this approach.
- **Screen must stay on:** the page requests a wake lock; if the phone locks, it drops off the list.
- **Reconstruction:** each phone has its own lens, so run COLMAP with
  `FIPMESH_COLMAP_SINGLE_CAMERA_PER_FOLDER=1` (default is one shared camera model for everything).

## Security

The hub has no authentication; `tailscale serve` exposes it to your tailnet only (not the public internet;
that would be `tailscale funnel`, don't use it). Anyone on the tailnet can trigger the cameras.
