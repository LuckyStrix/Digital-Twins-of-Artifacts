"""OS integration: WSL detection, path conversion, viewer launch, folders.

Everything Windows-specific switches off automatically outside WSL, so the
app also runs on plain Linux (desktop or over SSH).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from . import SCRIPT_DIR


def find_python(path: Path) -> str:
    return str(path) if path.exists() else sys.executable


def venv_python() -> str:
    """The interpreter the stages run in.

    FIPMESH_PYTHON if set (the Docker image sets it, so a host ./venv in the
    bind-mounted checkout isn't used); otherwise the repo's venv if it exists,
    else the current one.
    """
    override = os.environ.get("FIPMESH_PYTHON")
    if override:
        return override
    return find_python(SCRIPT_DIR / "venv" / "bin" / "python3")


def running_under_wsl() -> bool:
    return "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ


def has_display() -> bool:
    """Whether a GUI window (viewer, file manager) can be shown.

    Under WSL, windows go through Windows itself (native Python / Explorer),
    so a display is always available there.
    """
    if sys.platform == "win32" or running_under_wsl():
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ── path conversion ───────────────────────────────────────────────────────────

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:([\\/]|$)")


def unquote(p: str) -> str:
    """Strip whitespace and one pair of matching surrounding quotes, as added
    by Explorer's "Copy as path" or a terminal drag-and-drop."""
    p = p.strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'":
        p = p[1:-1].strip()
    return p


def needs_normalising(p: str) -> bool:
    """A typed path that to_posix_path() would change (quoted or Windows)."""
    return p.strip() != unquote(p) or looks_like_windows_path(p)


def looks_like_windows_path(p: str) -> bool:
    """``D:\\scans``, ``D:/scans`` or ``\\\\server\\share`` (optionally quoted)."""
    p = unquote(p)
    return bool(_WIN_DRIVE_RE.match(p)) or p.startswith("\\\\")


def to_windows_path(p: Path) -> str:
    """Convert a WSL path to its Windows equivalent for handing to a native .exe."""
    try:
        r = subprocess.run(["wslpath", "-w", str(p)],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return str(p)


def to_posix_path(p: str) -> str:
    """Convert a Windows path typed or pasted by the user to a WSL path.

    Surrounding quotes are always removed. Otherwise returns ``p`` unchanged
    when it isn't a Windows path or can't be converted (not under WSL,
    wslpath failed).
    """
    p = unquote(p)
    if not looks_like_windows_path(p) or not running_under_wsl():
        return p
    try:
        r = subprocess.run(["wslpath", "-u", p],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return p


# ── viewer ────────────────────────────────────────────────────────────────────

def viewer_env() -> dict:
    """Environment for spawning viewer.py (Open3D window).

    Under WSLg, Open3D's bundled GLFW can fail to create a window two ways:
      - Mesa picks the Zink (Vulkan) GL backend and can't select a device
        ("ZINK: failed to choose pdev") -- LIBGL_ALWAYS_SOFTWARE forces the
        llvmpipe software rasterizer instead, skipping device selection.
      - GLFW's Wayland backend refuses to create the window at all because
        it tries to set an explicit window position, which the Wayland
        protocol doesn't allow apps to do ("The platform does not support
        setting the window position"). Dropping WAYLAND_DISPLAY makes GLFW's
        platform auto-detection fall back to X11 (via WSLg's XWayland);
        GLFW_PLATFORM=x11 forces it explicitly on GLFW 3.4+ (a no-op on
        older GLFW that doesn't read this variable).
    Harmless outside WSL -- these only affect the spawned viewer process.
    """
    env = os.environ.copy()
    env.pop("WAYLAND_DISPLAY", None)
    env.setdefault("GLFW_PLATFORM", "x11")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    return env


@lru_cache(maxsize=1)
def find_native_windows_python() -> str | None:
    """Under WSL, find a native Windows python.exe with open3d installed.

    Native Windows Open3D uses Win32/WGL windowing directly, which sidesteps
    WSLg's GLFW-over-Wayland problems entirely (some Open3D/GLFW builds can't
    create a window under WSLg's Wayland compositor at all -- see
    viewer_env()). Returns None outside WSL, or if no such interpreter is
    found.
    """
    if not running_under_wsl():
        return None
    for exe in sorted(Path("/mnt/c/Users").glob("*/AppData/Local/Programs/Python/Python3*/python.exe")):
        try:
            r = subprocess.run([str(exe), "-c", "import open3d"],
                               capture_output=True, timeout=15)
            if r.returncode == 0:
                return str(exe)
        except Exception:
            continue
    return None


def resolve_viewer_launch(path: Path) -> tuple[list[str], dict]:
    """Command + env to launch viewer.py for `path`.

    Prefers a native Windows Python (see find_native_windows_python) when
    running under WSL; falls back to the in-repo/venv interpreter with the
    WSLg workaround env otherwise.
    """
    native_py = find_native_windows_python()
    if native_py:
        return (
            [native_py, to_windows_path(SCRIPT_DIR / "viewer.py"), to_windows_path(path)],
            os.environ.copy(),
        )
    return (
        [venv_python(), str(SCRIPT_DIR / "viewer.py"), str(path)],
        viewer_env(),
    )


def launch_viewer(path: Path) -> subprocess.Popen:
    """Start viewer.py for `path`. Raises FileNotFoundError if the
    interpreter is missing; pass the result to ``wait_viewer``."""
    cmd, env = resolve_viewer_launch(path)
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )


def wait_viewer(proc: subprocess.Popen) -> tuple[bool, str]:
    """Block until the viewer exits; return (failed, output)."""
    out = proc.stdout.read() if proc.stdout else ""
    proc.wait()
    # Open3D can fail to create a window and still exit 0, so check the log
    # text too, not just the return code.
    failed = proc.returncode != 0 or "Failed creating OpenGL window" in out
    return failed, out


# ── folders ───────────────────────────────────────────────────────────────────

def open_folder(d: Path) -> str | None:
    """Open `d` in the desktop file manager. Returns an error message, or
    None on success."""
    try:
        if sys.platform == "win32":
            os.startfile(str(d))  # type: ignore[attr-defined]
        elif running_under_wsl():
            # WSL usually has no desktop file-open helper (xdg-open) of its
            # own; go through Windows Explorer via WSL interop instead.
            subprocess.Popen(["explorer.exe", to_windows_path(d)])
        elif has_display():
            subprocess.Popen(["xdg-open", str(d)])
        else:
            return f"No display available. Folder: {d}"
    except (FileNotFoundError, OSError) as exc:
        return f"Could not open folder automatically ({exc}). Folder: {d}"
    return None


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _run_powershell_dialog(script: str) -> str | None:
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except (FileNotFoundError, OSError):
        return None
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        return None
    return to_posix_path(out)


_PS_PRELUDE = (
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$owner = New-Object System.Windows.Forms.Form -Property @{TopMost=$true}; "
)


def native_folder_dialog(title: str = "Select folder", initial: str = "") -> str | None:
    """Under WSL, show the Windows folder picker. Blocks until closed; run it
    off the UI thread. Returns a WSL path, or None if cancelled/unavailable."""
    if not running_under_wsl():
        return None
    init = to_windows_path(Path(initial)) if initial and Path(initial).is_dir() else ""
    script = (
        _PS_PRELUDE
        + "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
        + f"$d.Description = {_ps_quote(title)}; "
        + "$d.ShowNewFolderButton = $true; "
        + (f"$d.SelectedPath = {_ps_quote(init)}; " if init else "")
        + "if ($d.ShowDialog($owner) -eq 'OK') { [Console]::Out.Write($d.SelectedPath) }"
    )
    return _run_powershell_dialog(script)


def native_file_dialog(title: str = "Select file", initial: str = "",
                       pattern: str = "") -> str | None:
    """Under WSL, show the Windows open-file picker (see native_folder_dialog).
    ``pattern`` is a glob like ``*.json``."""
    if not running_under_wsl():
        return None
    init = to_windows_path(Path(initial)) if initial and Path(initial).is_dir() else ""
    flt = "All files (*.*)|*.*"
    if pattern:
        flt = f"{pattern} files ({pattern})|{pattern}|" + flt
    script = (
        _PS_PRELUDE
        + "$d = New-Object System.Windows.Forms.OpenFileDialog; "
        + f"$d.Title = {_ps_quote(title)}; "
        + f"$d.Filter = {_ps_quote(flt)}; "
        + (f"$d.InitialDirectory = {_ps_quote(init)}; " if init else "")
        + "if ($d.ShowDialog($owner) -eq 'OK') { [Console]::Out.Write($d.FileName) }"
    )
    return _run_powershell_dialog(script)


def browse_locations() -> list[tuple[str, Path]]:
    """Starting points for the in-terminal folder browser: home, /, drives
    under /mnt (Windows drives in WSL) and /media, network mounts."""
    locs: list[tuple[str, Path]] = [("Home", Path.home()), ("/", Path("/"))]
    seen = {p for _, p in locs}
    for root in (Path("/mnt"), Path("/media"), Path("/run/media")):
        try:
            children = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for p in children:
            if p in seen or p.name in ("wsl", "wslg"):
                continue
            label = f"{p.name.upper()}:" if root == Path("/mnt") and len(p.name) == 1 else str(p)
            locs.append((label, p))
            seen.add(p)
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        mounts = []
    for line in mounts:
        parts = line.split()
        if len(parts) >= 3 and parts[2] in ("cifs", "smb3", "nfs", "nfs4", "9p", "drvfs", "fuse.sshfs"):
            p = Path(parts[1].replace("\\040", " "))
            if p not in seen and p != Path("/") and not str(p).startswith(("/usr/lib/wsl", "/init")):
                locs.append((str(p), p))
                seen.add(p)
    return locs
