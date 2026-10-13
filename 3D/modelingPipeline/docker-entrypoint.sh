#!/bin/bash
# Container entrypoint. Before running the command (the app by default):
#   1. warns if the checkout's Dockerfile, requirements.txt or this script
#      changed since the image was built,
#   2. says whether each visible GPU can run the image's COLMAP build,
#   3. on Linux hosts, switches from root to the owner of the bind-mounted
#      repo, so files written into the checkout and data/ belong to the user.
# Checks only warn, never block, so the app and tests still start without a
# GPU. If there was a warning, it waits for Enter before the app takes over
# the screen (which would hide it).
#
#   docker compose run --rm pipeline gpu-check     only the checks, then exit
#
# FIPMESH_FAKE_CC=<major.minor>[,...] pretends to be those GPUs (to test the
# check). FIPMESH_KEEP_ROOT=1 stays root (e.g. to apt-get something to debug).

ARCH_FILE=/opt/colmap/CUDA_ARCHITECTURES
NVCC_FILE=/opt/colmap/NVCC_GPU_CODE
BUILD_INPUTS=/opt/fip3d/build-inputs
APP_DIR=/repo/3D/modelingPipeline

warned=0
note() { printf "\e[33m[%s]\e[0m %s\n" "$1" "${*:2}" >&2; warned=1; }
ok()   { printf "\e[32m[%s]\e[0m %s\n" "$1" "${*:2}" >&2; }

check_image() {
    local f stale=()
    for f in Dockerfile requirements.txt docker-entrypoint.sh; do
        if [ -r "$APP_DIR/$f" ] && ! cmp -s "$BUILD_INPUTS/$f" "$APP_DIR/$f"; then
            stale+=("$f")
        fi
    done
    if [ ${#stale[@]} -gt 0 ]; then
        note image "${stale[*]} changed since this image was built. Rebuild it with"
        note image "  docker compose build"
    fi
}

# "8.6" -> "86"; prints nothing for anything else.
sm_of() { [[ "$1" =~ ^([0-9]+)\.([0-9])$ ]] && echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"; }

# Prints how one GPU (compute capability $2, e.g. 8.6) fares with this build.
# Returns 1 if the image has no usable code for it.
check_one_gpu() {
    local label="$1" cc="$2" sm entry
    sm="$(sm_of "$cc")"
    if [ -z "$sm" ]; then
        note gpu "$label: could not read its compute capability ('$cc')."
        return 0
    fi
    local major=$((sm / 10)) minor=$((sm % 10))

    # Same-major compatibility: code built for 8.6 runs on 8.9, but not on 9.0.
    # Arch-specific entries (120a) run only on exactly that GPU.
    for entry in "${real[@]}"; do
        if [ "$entry" = "$sm" ] || { [ $((entry / 10)) = "$major" ] && [ $((entry % 10)) -le "$minor" ]; }; then
            ok gpu "$label (compute $cc): supported by this image."
            blackwell_note "$sm"
            return 0
        fi
    done
    for entry in "${exact[@]}"; do
        if [ "$entry" = "$sm" ]; then
            ok gpu "$label (compute $cc): supported by this image."
            blackwell_note "$sm"
            return 0
        fi
    done
    for entry in "${virt[@]}"; do
        if [ "$sm" -ge "$entry" ]; then
            ok gpu "$label (compute $cc): no prebuilt code for this GPU, so the driver compiles"
            ok gpu "COLMAP's kernels on first use (slower first run; cached afterward)."
            rebuild_hint "$sm" ok
            return 0
        fi
    done

    local oldest
    oldest="$(printf '%s\n' "${real[@]}" "${exact[@]}" "${virt[@]}" | sort -n | head -n 1)"
    if [ -n "$oldest" ] && [ "$sm" -lt "$oldest" ]; then
        note gpu "$label (compute $cc) is older than any GPU this image was built for"
        note gpu "(oldest: compute $((oldest / 10)).$((oldest % 10)))."
    else
        note gpu "$label (compute $cc): this image has no COLMAP code for it"
        note gpu "(built for CUDA_ARCHITECTURES=$archs)."
    fi
    rebuild_hint "$sm" note
    return 1
}

blackwell_note() {
    if [ "$1" -ge 100 ]; then
        # COLMAP ships PatchMatch for Blackwell as sm_90 PTX only
        # (colmap/colmap#3514); CUDA_CACHE_PATH keeps the result.
        ok gpu "Blackwell: the first dense reconstruction compiles COLMAP's PatchMatch"
        ok gpu "kernels (slower first run, once); later runs reuse the cache."
    fi
}

# Rebuild command that adds this GPU to the current list, if this image's CUDA
# toolkit can compile for it. $2 is ok (the GPU works already) or note.
rebuild_hint() {
    local sm="$1" say="$2"
    if [ -r "$NVCC_FILE" ] && grep -qx "sm_$sm" "$NVCC_FILE"; then
        "$say" gpu "To build COLMAP for it (keeps the other GPUs working):"
        "$say" gpu "  docker compose build --build-arg \"CUDA_ARCHITECTURES=$archs;$sm-real\""
    elif [ "$say" = note ]; then
        "$say" gpu "This image's CUDA toolkit can't compile for compute $((sm / 10)).$((sm % 10)); building"
        "$say" gpu "for it needs a newer CUDA_TAG in the Dockerfile. See DOCKER.md, 'Which GPUs work'."
    fi
}

check_gpus() {
    local gpus=()
    if [ -n "${FIPMESH_FAKE_CC:-}" ]; then
        local i=0 cc
        IFS=',' read -ra ccs <<< "$FIPMESH_FAKE_CC"
        for cc in "${ccs[@]}"; do gpus+=("$i, (FIPMESH_FAKE_CC), $cc"); i=$((i + 1)); done
    elif ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        note gpu "No NVIDIA GPU visible in the container. COLMAP reconstruction (stage 2) needs one."
        note gpu "Start it with 'docker compose run --rm pipeline' (GPU reserved in hwaccel.yml),"
        note gpu "or 'docker run --gpus all ...'. On Windows, Docker Desktop needs the WSL2 backend"
        note gpu "and a current NVIDIA driver. See DOCKER.md."
        return
    else
        mapfile -t gpus < <(nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader)
    fi

    if [ ! -r "$ARCH_FILE" ]; then
        note gpu "Can't find this image's GPU list ($ARCH_FILE) to compare with."
        return
    fi
    archs="$(tr -d '[:space:]' < "$ARCH_FILE")"
    real=() virt=() exact=()
    local entry
    IFS=';' read -ra entries <<< "$archs"
    for entry in "${entries[@]}"; do
        case "$entry" in
            [0-9]*a-real|[0-9]*a)  exact+=("${entry%%a*}") ;;
            *[!0-9]*-virtual|*[!0-9]*-real) ;;  # other suffixes (e.g. 120f): skip
            [0-9]*-virtual)        virt+=("${entry%-virtual}") ;;
            [0-9]*-real)           real+=("${entry%-real}") ;;
            *[!0-9]*)
                note gpu "This image was built with CUDA_ARCHITECTURES=$archs, which can't be checked here."
                return ;;
            *)                     real+=("$entry"); virt+=("$entry") ;;  # plain "86" = both
        esac
    done

    # CUDA_VISIBLE_DEVICES limits what COLMAP sees; the image sets
    # CUDA_DEVICE_ORDER=PCI_BUS_ID, so its numbers match nvidia-smi's.
    local visible=",${CUDA_VISIBLE_DEVICES:-},"
    [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9,]+$ ]] || visible=""

    local line idx name cc shown=0 bad=0
    for line in "${gpus[@]}"; do
        IFS=',' read -r idx name cc <<< "$line"
        idx="$(echo "$idx" | xargs)"; name="$(echo "$name" | xargs)"; cc="$(echo "$cc" | xargs)"
        if [ -n "$visible" ] && [[ "$visible" != *",$idx,"* ]]; then continue; fi
        shown=$((shown + 1))
        [ ${#gpus[@]} -gt 1 ] && name="GPU $idx, $name"
        check_one_gpu "$name" "$cc" || bad=$((bad + 1))
    done
    if [ "$bad" -gt 0 ] && [ "$shown" -gt "$bad" ]; then
        note gpu "COLMAP uses every visible GPU unless told otherwise. Set 'GPU index' on the app's"
        note gpu "COLMAP tab to a supported GPU's number, or start with"
        note gpu "  docker compose run --rm -e CUDA_VISIBLE_DEVICES=<number> pipeline"
    fi
}

# Run as the repo's owner (Linux hosts). Docker Desktop shows Windows folders
# as root-owned, so there it stays root, which writes as the Windows user.
drop_root() {
    [ "$(id -u)" = 0 ] && [ -z "${FIPMESH_KEEP_ROOT:-}" ] || return 0
    local uid gid
    uid="$(stat -c %u "$APP_DIR" 2>/dev/null)" && gid="$(stat -c %g "$APP_DIR")" || return 0
    [ "$uid" != 0 ] || return 0
    # The cache volume starts out root-owned, and a FIPMESH_KEEP_ROOT run or
    # another user's checkout can leave files in it; dockerd creates a missing
    # data folder as root, so hand an empty one over.
    mkdir -p /cache/home
    if [ -n "$(find /cache \( ! -uid "$uid" -o ! -gid "$gid" \) -print -quit)" ]; then
        chown -R "$uid:$gid" /cache
    fi
    if [ -d /data ] && [ "$(stat -c %u /data)" = 0 ] && [ -z "$(ls -A /data)" ]; then
        chown "$uid:$gid" /data
    fi
    export HOME=/cache/home
    exec setpriv --reuid="$uid" --regid="$gid" --clear-groups -- "$@"
}

check_image
check_gpus
if [ "${1:-}" = gpu-check ]; then
    exit 0
fi
if [ "$warned" = 1 ] && [ "$*" = "python3 app.py" ] && [ -t 0 ] && [ -t 2 ]; then
    read -r -t 60 -p "Press Enter to open the app (opens in 60 s anyway)... " || echo
fi
drop_root "$@"
exec "$@"
