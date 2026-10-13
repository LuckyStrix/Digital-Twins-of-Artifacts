# Running the reconstruction pipeline in Docker

The Docker image holds everything the reconstruction needs: CUDA COLMAP,
exiftool, cuDNN and the Python packages. You don't need the WSL CUDA toolkit
setup, the cuDNN repo step or the COLMAP build in
[`BUILDING_COLMAP.md`](BUILDING_COLMAP.md). The pipeline itself is the same
code as the native setup, which keeps working; Docker is an alternative, not a
replacement.

This covers reconstruction only (`app.py` and its four stages). Capture
(`../captureApp/`) talks to cameras and an Arduino over USB, which a container
can't easily reach, so it still runs natively.

## What you need

- **An NVIDIA GPU.** COLMAP's dense reconstruction (stage 2) only runs on CUDA;
  there is no CPU fallback and no AMD, Intel or Apple GPU support. See
  [Which GPUs work](#which-gpus-work).
- **An NVIDIA driver from the R570 series or newer** (version 570 or higher,
  early 2025 on). The image is built on CUDA 12.8, and Docker refuses to start
  it on older drivers with `unsatisfied condition: cuda>=12.8`. Check with
  `nvidia-smi`.
- **Windows:** [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
  with the WSL2 backend (the default). You don't need a CUDA toolkit or a
  particular Ubuntu version.
- **Linux:** Docker Engine with the Compose plugin, the NVIDIA driver, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- About **30 GB** of free disk while building. The finished image is about
  12 GB; `docker builder prune` reclaims the rest of the build cache afterward.

Check that Docker can see your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Get the code

Clone the repo anywhere on your machine. Git LFS isn't needed for the pipeline
(it only stores images and models for the website).

```bash
git clone https://github.com/LuckyStrix/Digital-Twins-of-Artifacts.git
cd Digital-Twins-of-Artifacts/3D/modelingPipeline
```

All the `docker compose` commands below run from this folder, in PowerShell,
Windows Terminal or a WSL/Linux shell.

## Build the image (once)

From this folder (`3D/modelingPipeline/`):

```bash
docker compose build
```

This downloads about 5 GB of base images and compiles COLMAP for every
supported GPU generation. The compile alone took about 12 minutes on a 20-core
machine; slower machines and connections take longer. Later builds reuse the
cache. If you only ever use one GPU, you can build just for it and save most
of the compile time (see [Build options](#build-options)).

Rebuild after `Dockerfile`, `requirements.txt` or `docker-entrypoint.sh`
changes; the container warns at startup when any of them differ from the
image's copy. Other code changes don't need a rebuild, because the repo is
mounted into the container (at `/repo`) rather than copied into the image.

## Run it

```bash
docker compose run --rm pipeline
```

This opens the same terminal app as `python3 app.py` (see
[`README.md`](README.md#usage)). Before that, the container checks each GPU
it can see. If something is wrong (no GPU, a GPU the image wasn't built for,
an outdated image), it prints a `[gpu]` or `[image]` warning and waits for
Enter, so the app doesn't cover the message. To see the check on its own:

```bash
docker compose run --rm pipeline gpu-check
```

```
[gpu] NVIDIA GeForce RTX 5060 Ti (compute 12.0): supported by this image.
```

Run the tests the same way:

```bash
docker compose run --rm pipeline python3 -m pytest tests
```

### Where your files go

The repo's `data/` folder (two levels up, gitignored) appears in the container
as **`/data`**. Put capture sets in `data/`, then choose `/data/<set>` as the
input folder in the app. Outputs written under `/data` show up in `data/` on
your machine.

Captures on a network share (such as the lab NAS) can't be mounted directly:
Docker Desktop doesn't see mapped drive letters or `\\server\share` paths.
Copy the capture set into `data/` first (local disk is also much faster for
the hundreds of reads COLMAP does).

To use a different local folder, set `DATA_LOCATION` when you run:

```bash
DATA_LOCATION=/mnt/d/scans docker compose run --rm pipeline       # Linux / WSL shell
$env:DATA_LOCATION="D:\scans"; docker compose run --rm pipeline   # PowerShell
```

Things that work differently from the native setup:

- **Paths are container paths.** Windows paths typed into the app
  (`D:\scans`) aren't converted, because there's no `wslpath` in the container.
  Use `/data/...`.
- **View buttons show the file path** instead of opening a window, as they do
  over SSH. Open the `.ply`/`.gltf` on your machine, or run `viewer.py` natively.
- **rembg's model** downloads on first use into a Docker volume
  (`fip3d_model-cache`), so it's fetched only once.
- **Memory:** on Windows, Docker Desktop runs inside WSL2 and shares its memory
  limit. If a run gets `Killed`, raise it as described in
  [Giving WSL more RAM](../../SETUP.md#giving-wsl-more-ram--a-bigger-swap-pagefile).
- **Linux hosts:** the app runs as the user who owns the checkout, so files it
  writes into the repo and `data/` are yours, not root's.
- **Several GPUs:** COLMAP uses all of them by default. To use one, set
  **GPU index** on the app's COLMAP tab to its number from the startup check,
  or start with `docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 pipeline`.

## Which GPUs work

Find your GPU's compute capability with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`, or on
[NVIDIA's list](https://developer.nvidia.com/cuda-gpus).

| Generation | Example cards | Compute | Default image |
|---|---|---|---|
| Blackwell | RTX 50xx, B200 | 12.0, 10.0 | yes |
| Hopper | H100 | 9.0 | yes |
| Ada | RTX 40xx, L4 | 8.9 | yes |
| Ampere | RTX 30xx, A100, A4000 | 8.6, 8.0 | yes |
| Turing | RTX 20xx, GTX 16xx, T4 | 7.5 | yes |
| Volta | V100, Titan V | 7.0 | yes |
| Pascal | GTX 10xx, P100 | 6.1, 6.0 | yes |
| Newer than Blackwell | — | > 12.0 | yes, compiled on first use (slower start) |
| Maxwell | GTX 9xx | 5.x | no; may work built with `52-real` (untested) |

The default image includes compiled COLMAP code for every generation marked
"yes", so one image works across the lab's machines without rebuilding. Code
for one GPU also runs on later GPUs of the same major version (8.6 code runs
on an 8.7 card), and anything newer than the list falls back to code the
driver compiles on first use. The startup check accounts for all of this.

One exception on Blackwell (RTX 50xx, B200): COLMAP deliberately doesn't
compile its PatchMatch stereo kernels for these cards, to avoid an NVCC
miscompile ([colmap#3514](https://github.com/colmap/colmap/issues/3514)). They
ship as generic code the driver compiles on first use, so the first dense
reconstruction on a Blackwell card takes longer while that compiles. The
result is kept in the `fip3d_model-cache` volume, so it happens once, not every
run. A native COLMAP build behaves the same way.

## Build options

Pass these with `docker compose build --build-arg NAME=value`:

| Arg | Default | Meaning |
|---|---|---|
| `CUDA_ARCHITECTURES` | `60-real;61-real;70-real;…;120-real;120-virtual` | GPU generations COLMAP is compiled for. Compute 8.6 is `86`. `-real` adds compiled code for that GPU; `-virtual` adds code the driver compiles for newer GPUs. One entry, e.g. `86-real`, builds much faster but only runs on that generation. |
| `COLMAP_VERSION` | `4.2.1` | COLMAP git tag to build. |
| `BUILD_JOBS` | `4` | Parallel compile jobs. Each can use several GB of RAM, and on Windows the build shares WSL2's memory with every running container, so raise this only if you have memory to spare. |

If the startup check says the image has no code for your GPU, it prints the
rebuild command that adds it to the current list, so the image keeps working
on the other GPUs. The pasted `--build-arg "CUDA_ARCHITECTURES=…"` works in
PowerShell and bash. A GPU newer than the image's CUDA toolkit (12.8) can't be
added this way; it needs a newer `CUDA_TAG` in the `Dockerfile`, and the check
says so.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[gpu] No NVIDIA GPU visible in the container` | Start with `docker compose run`, not plain `docker run` (or add `--gpus all`). On Windows, update the NVIDIA driver and check that Docker Desktop uses the WSL2 backend. On Linux, install the NVIDIA Container Toolkit. |
| `could not select device driver "nvidia"` | Same as above: Docker can't see the NVIDIA runtime. |
| `unsatisfied condition: cuda>=12.8, please update your driver` | The NVIDIA driver is older than R570. Update it (on Windows, the Windows driver; WSL uses it). |
| `[image] ... changed since this image was built` | Run `docker compose build`. |
| `no kernel image is available for execution on the device` | COLMAP hit a GPU the image wasn't built for, often the second of two GPUs. Run `docker compose run --rm pipeline gpu-check` and follow what it says. |
| `Killed` partway through a run | Out of memory; see the Memory note above. |
| The app looks garbled | Use Windows Terminal (or any modern terminal) and keep `docker compose run`'s terminal attached. |
