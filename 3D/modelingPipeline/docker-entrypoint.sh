#!/bin/bash
# Container entrypoint: say whether this GPU can run the image's COLMAP build,
# then run the command (the app by default). Warns only, never blocks, so the
# app and tests still start without a GPU.
#
# FIPMESH_FAKE_CC=<major.minor> pretends to be that GPU (for testing the check).

ARCH_FILE=/opt/colmap/CUDA_ARCHITECTURES

note() { printf "\e[33m[gpu]\e[0m %s\n" "$*" >&2; }
ok()   { printf "\e[32m[gpu]\e[0m %s\n" "$*" >&2; }

check_gpu() {
    local name cc
    if [ -n "${FIPMESH_FAKE_CC:-}" ]; then
        name="(FIPMESH_FAKE_CC)"
        cc="$FIPMESH_FAKE_CC"
    elif ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        note "No NVIDIA GPU visible in the container. COLMAP reconstruction (stage 2) needs one."
        note "Start it with 'docker compose run --rm pipeline' (GPU reserved in hwaccel.yml),"
        note "or 'docker run --gpus all ...'. On Windows, Docker Desktop needs the WSL2 backend"
        note "and a current NVIDIA driver. See DOCKER.md."
        return
    else
        # First GPU only; with several, the pipeline's gpu_index setting picks.
        IFS=',' read -r name cc < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1)
        name="$(echo "$name" | xargs)"
        cc="$(echo "$cc" | xargs)"
    fi

    local sm="${cc/./}"
    if ! [[ "$sm" =~ ^[0-9]+$ ]] || [ ! -r "$ARCH_FILE" ]; then
        note "$name: could not compare compute capability '$cc' with this image's build."
        return
    fi

    local real=() virt=() entry
    IFS=';' read -ra entries < "$ARCH_FILE"
    for entry in "${entries[@]}"; do
        case "$entry" in
            *-virtual) virt+=("${entry%-virtual}") ;;
            *-real)    real+=("${entry%-real}") ;;
            *)         real+=("$entry"); virt+=("$entry") ;;  # plain "86" = both
        esac
    done

    for entry in "${real[@]}"; do
        if [ "$entry" = "$sm" ]; then
            ok "$name (compute $cc): supported by this image."
            if [ "$sm" -ge 100 ]; then
                # COLMAP ships PatchMatch for Blackwell as sm_90 PTX only
                # (colmap/colmap#3514); CUDA_CACHE_PATH keeps the result.
                ok "Blackwell: the first dense reconstruction JIT-compiles COLMAP's PatchMatch"
                ok "kernels (slower first run, once); later runs reuse the cache."
            fi
            return
        fi
    done
    for entry in "${virt[@]}"; do
        if [ "$sm" -ge "$entry" ]; then
            note "$name (compute $cc): no prebuilt code for this GPU; the driver will JIT-compile"
            note "COLMAP's kernels from PTX, so the first run starts slower. To build for it:"
            note "  docker compose build --build-arg CUDA_ARCHITECTURES=${sm}-real"
            return
        fi
    done
    note "$name (compute $cc) is older than this image supports (Pascal, compute 6.0,"
    note "is the oldest). See DOCKER.md, 'Which GPUs work'."
}

check_gpu
exec "$@"
