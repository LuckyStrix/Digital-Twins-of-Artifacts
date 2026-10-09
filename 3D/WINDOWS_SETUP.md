# 3D Reconstruction on Native Windows (no WSL, no admin)

Use this guide to run the 3D reconstruction app (`modelingPipeline/app.py`)
directly on Windows, using the GPU, **without admin rights**: no WSL, no
Docker, no system installs. Everything installs inside the
`3D\modelingPipeline` folder.

If you *can* get WSL2, the standard setup in [`../SETUP.md`](../SETUP.md)
is still the reference. This guide is the fallback for lab or managed PCs
where WSL can't be turned on.

> **Covers reconstruction only.** The capture app (`captureApp/`) needs
> gphoto2, which has no Windows build. Capture on the Linux rig machine and
> copy the photos over.

## What you need first

| Need | How to check | If it's missing |
|---|---|---|
| NVIDIA GPU + driver | `nvidia-smi` in a terminal shows your GPU | Ask IT. The driver is the only part that needs admin |
| Python **3.12** (3.9–3.12 work; **not** 3.13+) | `py -3.12 --version` | `winget install -e --id Python.Python.3.12 --scope user`, or the python.org installer with **"Install for all users" unticked** |
| The repo | — | `git clone`, GitHub Desktop, or *Code → Download ZIP* on GitHub, on the `windows-native` branch |

About 10 GB of free disk space covers torch, COLMAP and the models.

## Install (one command)

Open **PowerShell** in `3D\modelingPipeline` and run:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

`-ExecutionPolicy Bypass` applies to this one command only and needs no
admin. The script:

1. Finds Python 3.12 and creates `venv\`.
2. Installs the **CUDA build of torch** (CUDA 12, from download.pytorch.org),
   then `requirements.txt`, which includes `rembg[gpu]` / `onnxruntime-gpu`.
   torch also provides the CUDA and cuDNN DLLs that rembg uses, so you don't
   install CUDA or cuDNN separately.
3. Downloads the official **COLMAP Windows CUDA** release into `colmap\`.
4. Downloads **exiftool** into `tools\`.
5. Runs `check_setup.py`. The setup is ready once every line says `ok`.

You can re-run it safely: it skips anything already done. Use `-Force` to
download COLMAP and exiftool again.

## Run

```powershell
cd 3D\modelingPipeline
venv\Scripts\python app.py
```

Use **Windows Terminal** if you can; the UI looks better than in the old
console. The app works the same as under WSL: pick the input and output
folders (the **Windows dialog…** button opens the normal Explorer picker),
then press `r` to run all four stages. **View** opens the Open3D viewer
directly.

Any drive works for input and output, including `D:\` and network shares.

## Check the setup any time

```powershell
venv\Scripts\python check_setup.py
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `running scripts is disabled on this system` | Use the exact `powershell -ExecutionPolicy Bypass -File ...` command above. If IT blocks even that, follow the manual steps below |
| `[warn] rembg is running on the CPU` | torch is the CPU build. Run `venv\Scripts\python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`, then re-run `check_setup.py` |
| check says `onnxruntime is CPU-only` | `venv\Scripts\python -m pip uninstall -y onnxruntime onnxruntime-gpu`, then `venv\Scripts\python -m pip install "onnxruntime-gpu<1.27"` |
| check says `COLMAP is a CPU-only build` | You have the `nocuda` zip. Delete `colmap\` and re-run the setup script |
| Stage 2 hangs right at the start | exiftool is still named `exiftool(-k).exe`, which waits for a keypress. Rename it to `exiftool.exe` |
| `torch` / `open3d` won't install | Python is 3.13 or newer. Install 3.12, delete `venv\`, and re-run the setup script |
| Out of memory in dense reconstruction | Lower *Image scale* or the cache sizes on the COLMAP tab. Windows has no WSL memory cap, so this means the PC itself is out of RAM |
| Firewall or proxy blocks downloads | Download the two zips by hand (see below) |

## Manual install (if the script can't run)

From `3D\modelingPipeline` in a normal Command Prompt:

```bat
py -3.12 -m venv venv
venv\Scripts\python -m pip install --upgrade pip wheel
venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python -m pip install -r requirements.txt
```

Then:

- **COLMAP:** from <https://github.com/colmap/colmap/releases>, download
  `colmap-x64-windows-cuda.zip` (**not** `nocuda`). Unzip it so that
  `3D\modelingPipeline\colmap\COLMAP.bat` exists.
- **exiftool:** from <https://exiftool.org>, download the *Windows
  Executable* (64-bit zip). Put `exiftool(-k).exe` and its `exiftool_files`
  folder in `3D\modelingPipeline\tools\`, and **rename the exe to
  `exiftool.exe`**.
- Run `venv\Scripts\python check_setup.py`.

If you keep COLMAP or exiftool somewhere else, point the app at them with
user environment variables (no admin needed): `FIPMESH_COLMAP_BIN` (full
path to `COLMAP.bat`) and `FIPMESH_EXIFTOOL_BIN` (full path to
`exiftool.exe`).

## How the Windows mode works

On Windows, the pipeline swaps out the parts that need Linux:

- **Stage 2** runs `src/run_stage2.py`, a Python port of `run.sh` +
  `src/main.sh` with the same settings and `FIPMESH_*` variables, instead of
  `bash run.sh`. Linux and WSL still use `run.sh`.
- **COLMAP** is found in this order: `FIPMESH_COLMAP_BIN` →
  `colmap_local` → `colmap\COLMAP.bat` → `colmap` on PATH.
- **rembg's GPU support** borrows the CUDA and cuDNN DLLs bundled with
  torch instead of the `apt` cuDNN package.
- **Stop** kills the whole process tree (`taskkill /T`), and stages run at
  below-normal priority so the UI stays responsive.
