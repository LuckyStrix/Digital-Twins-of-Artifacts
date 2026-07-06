#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRAM=$0

echo "1"

# dependency lists
SYSTEM_DEPS=(python3 colmap)
# python dependencies
PYTHON_DEPS=(numpy open3d)
src/check_config.sh \
"${SYSTEM_DEPS[@]}" -- \
"${PYTHON_DEPS[@]}"
rc=$?
if [ $rc -ne 0 ]; then
    die "missing dependencies"
fi

# -------------------------------
# helper functions
# -------------------------------
usage() {
    printf "usage: %s [-i <primary_image_dir>] [-s <secondary_image_dir>] [-o <output_dir>] [-v]\n" "$0" >&2
    exit 1
}

status() {
    printf "%s: " "$PROGRAM"
    printf "\e[32mstatus\e[0m"
    printf ": $1\n" "${@:2}" >&2 
}

die() {
    printf "%s: " "$PROGRAM"
    printf "\e[31merror\e[0m"
    printf ": $1\n" "${@:2}" >&2 
    usage
    exit 1
}

echo "2"

# -------------------------------
# arg parsing
# -------------------------------
# predefine so not only in scope of while loop
P_IMGDIR=
S_IMGDIR=
EXTFILE=
CAL_DIR=
OUT_DIR=${SCRIPT_DIR}/out
SDIR_IS_NEEDED=
VERBOSE=
while getopts "hi:s:o:v" o; do
    case "${o}" in
        h)
            usage
            ;;
        i)
            [ -z "${OPTARG}" ] && usage 
            P_IMGDIR="${SCRIPT_DIR}/${OPTARG}"
            ;;
        s)
            [ -z "${OPTARG}" ] && usage 
            S_IMGDIR="${SCRIPT_DIR}/${OPTARG}"
            SDIR_IS_NEEDED=true
            ;;
        o)
            [ -z "${OPTARG}" ] && usage 
            OUT_DIR="${SCRIPT_DIR}/${OPTARG}"
            ;;
        v)
            VERBOSE=true
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1)) 

echo "3"

# -------------------------------
# paths
# -------------------------------
# make sure we have a valid image directory
if [ ! -d "$P_IMGDIR" ] || [ -z "${P_IMGDIR}" ]; then
  die "invalid primary image directory \"$P_IMGDIR\""
fi

# make sure we have a valid secondary image directory if we need one
if [ -n "$SDIR_IS_NEEDED" ]; then
  if [ ! -d "$S_IMGDIR" ] || [ -z "${S_IMGDIR}" ]; then
    die "invalid secondary image directory \"$S_IMGDIR\""
  fi
fi

echo "3.1"

# make sure we have a valid output directory
if [ ! -d "$OUT_DIR" ] || [ -z "${OUT_DIR}" ]; then
  mkdir -p $OUT_DIR
fi
OUT_DIR="${OUT_DIR%/}"

echo "3.2"

# strip metadata with exiftool for colmap
exiftool -overwrite_original -all:all= -r $P_IMGDIR >/dev/null

echo "3.3"

rc=$?
if [ $rc -eq 0 ]; then
    status "exiftool metadata stripping succeeded for %s" "$P_IMGDIR"
else
    die "exiftool failed with exit code %d" $rc
fi

echo "3.4"

if [ -n "$SDIR_IS_NEEDED" ]; then
  exiftool -overwrite_original -all:all= -r $S_IMGDIR >/dev/null 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
      status "exiftool metadata stripping succeeded for %s" "$S_IMGDIR"
  else
      die "exiftool failed with exit code %d" $rc
  fi
fi

echo "4"

# Prefer the repo-local CUDA-enabled COLMAP build unless overridden.
if [ -z "${FIPMESH_COLMAP_BIN:-}" ]; then
    LOCAL_COLMAP_BIN="$SCRIPT_DIR/colmap_local"
    if [ -x "$LOCAL_COLMAP_BIN" ]; then
        export FIPMESH_COLMAP_BIN="$LOCAL_COLMAP_BIN"
    fi
fi

# -------------------------------
# COLMAP variables
# -------------------------------
# use all available CPU threads by default for COLMAP stages that support it.
CPU_THREADS_DEFAULT="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
if ! [[ "$CPU_THREADS_DEFAULT" =~ ^[0-9]+$ ]] || [ "$CPU_THREADS_DEFAULT" -lt 1 ]; then
    CPU_THREADS_DEFAULT=4
fi

TOTAL_MEM_KB="$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
if ! [[ "$TOTAL_MEM_KB" =~ ^[0-9]+$ ]] || [ "$TOTAL_MEM_KB" -lt 1 ]; then
    TOTAL_MEM_KB=0
fi
TOTAL_MEM_GB=$(( (TOTAL_MEM_KB + 1048575) / 1048576 ))

EXTRACT_THREADS_DEFAULT="$CPU_THREADS_DEFAULT"
MATCH_THREADS_DEFAULT="$CPU_THREADS_DEFAULT"
MAPPER_THREADS_DEFAULT="$CPU_THREADS_DEFAULT"
FUSION_THREADS_DEFAULT="$CPU_THREADS_DEFAULT"
PATCH_CACHE_DEFAULT=64
FUSION_CACHE_DEFAULT=64

if [ "$TOTAL_MEM_GB" -gt 0 ] && [ "$TOTAL_MEM_GB" -le 18 ]; then
    EXTRACT_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 6 ? CPU_THREADS_DEFAULT : 6 ))
    MATCH_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 6 ? CPU_THREADS_DEFAULT : 6 ))
    MAPPER_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 8 ? CPU_THREADS_DEFAULT : 8 ))
    FUSION_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 8 ? CPU_THREADS_DEFAULT : 8 ))
    PATCH_CACHE_DEFAULT=12
    FUSION_CACHE_DEFAULT=12
elif [ "$TOTAL_MEM_GB" -gt 0 ] && [ "$TOTAL_MEM_GB" -le 24 ]; then
    EXTRACT_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 8 ? CPU_THREADS_DEFAULT : 8 ))
    MATCH_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 8 ? CPU_THREADS_DEFAULT : 8 ))
    MAPPER_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 10 ? CPU_THREADS_DEFAULT : 10 ))
    FUSION_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 10 ? CPU_THREADS_DEFAULT : 10 ))
    PATCH_CACHE_DEFAULT=16
    FUSION_CACHE_DEFAULT=16
elif [ "$TOTAL_MEM_GB" -gt 0 ] && [ "$TOTAL_MEM_GB" -le 32 ]; then
    EXTRACT_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 10 ? CPU_THREADS_DEFAULT : 10 ))
    MATCH_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 10 ? CPU_THREADS_DEFAULT : 10 ))
    MAPPER_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 12 ? CPU_THREADS_DEFAULT : 12 ))
    FUSION_THREADS_DEFAULT=$(( CPU_THREADS_DEFAULT < 12 ? CPU_THREADS_DEFAULT : 12 ))
    PATCH_CACHE_DEFAULT=24
    FUSION_CACHE_DEFAULT=24
fi

echo "5"

# -------------------------------
# COLMAP defaults
# -------------------------------
# extreme, typically high
export FIPMESH_COLMAP_QUALITY="${FIPMESH_COLMAP_QUALITY:-high}"
export FIPMESH_COLMAP_USE_GPU="${FIPMESH_COLMAP_USE_GPU:-1}"
export FIPMESH_COLMAP_GPU_INDEX="${FIPMESH_COLMAP_GPU_INDEX:--1}"
export FIPMESH_COLMAP_EXTRACT_THREADS="${FIPMESH_COLMAP_EXTRACT_THREADS:-$EXTRACT_THREADS_DEFAULT}"
export FIPMESH_COLMAP_MATCH_THREADS="${FIPMESH_COLMAP_MATCH_THREADS:-$MATCH_THREADS_DEFAULT}"
export FIPMESH_COLMAP_MAPPER_THREADS="${FIPMESH_COLMAP_MAPPER_THREADS:-$MAPPER_THREADS_DEFAULT}"
export FIPMESH_COLMAP_FUSION_THREADS="${FIPMESH_COLMAP_FUSION_THREADS:-$FUSION_THREADS_DEFAULT}"
export FIPMESH_COLMAP_PATCH_CACHE_SIZE="${FIPMESH_COLMAP_PATCH_CACHE_SIZE:-$PATCH_CACHE_DEFAULT}"
export FIPMESH_COLMAP_FUSION_CACHE_SIZE="${FIPMESH_COLMAP_FUSION_CACHE_SIZE:-$FUSION_CACHE_DEFAULT}"
export FIPMESH_COLMAP_FUSION_USE_CACHE="${FIPMESH_COLMAP_FUSION_USE_CACHE:-1}"
# num pixels, 0-1, default .5
export FIPMESH_COLMAP_IMAGE_SCALE="${FIPMESH_COLMAP_IMAGE_SCALE:-1}"
# 1 default, 2 = skip every other, default 2
export FIPMESH_COLMAP_IMAGE_STRIDE="${FIPMESH_COLMAP_IMAGE_STRIDE:-3}"
export FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES="${FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES:-16000}"
export FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD="${FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD:-0.0045}"
export FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD="${FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD:-12}"
export FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING="${FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING:-1}"
export FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE="${FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE:-0}"
export FIPMESH_COLMAP_MATCH_GUIDED="${FIPMESH_COLMAP_MATCH_GUIDED:-1}"
export FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES="${FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES:-65536}"
if [ -n "$S_IMGDIR" ]; then
    export FIPMESH_COLMAP_IMAGES_SECONDARY="$S_IMGDIR"
    export FIPMESH_COLMAP_SECONDARY_ROTATE_DEG="${FIPMESH_COLMAP_SECONDARY_ROTATE_DEG:-180}"
    export FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS="${FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS:-primary_frame_x}"
    export FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X="${FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X:-0}"
    export FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y="${FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y:-0}"
    export FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z="${FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z:-0}"
    export FIPMESH_COLMAP_SECONDARY_TRANSLATE_X="${FIPMESH_COLMAP_SECONDARY_TRANSLATE_X:-0}"
    export FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y="${FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y:-0}"
    export FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z="${FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z:-0}"
    export FIPMESH_COLMAP_SECONDARY_ALIGN_MODE="${FIPMESH_COLMAP_SECONDARY_ALIGN_MODE:-auto}"
fi
# -------------------------------
# status printing
# -------------------------------
if [ "$VERBOSE" ]; then
    status "using COLMAP: %s" "${FIPMESH_COLMAP_BIN:-$(command -v colmap 2>/dev/null || echo 'colmap')}"
    status "COLMAP quality: %s (use_gpu=%s)" "${FIPMESH_COLMAP_QUALITY}" "${FIPMESH_COLMAP_USE_GPU}"
    status "COLMAP input_image_scale: %s" "${FIPMESH_COLMAP_IMAGE_SCALE}"
    status "COLMAP input_image_stride: %s" "${FIPMESH_COLMAP_IMAGE_STRIDE}"
    status "COLMAP gpu_index: %s" "${FIPMESH_COLMAP_GPU_INDEX}"

  if [ "$TOTAL_MEM_GB" -gt 0 ]; then
    status "WSL memory detected: %sGB" "${TOTAL_MEM_GB}"
  fi

    status "COLMAP threads: extract=%s match=%s mapper=%s fusion=%s" \
        "${FIPMESH_COLMAP_EXTRACT_THREADS}" \
        "${FIPMESH_COLMAP_MATCH_THREADS}" \
        "${FIPMESH_COLMAP_MAPPER_THREADS}" \
        "${FIPMESH_COLMAP_FUSION_THREADS}"

    status "COLMAP cache: patch=%sGB fusion=%sGB (fusion_use_cache=%s)" \
        "${FIPMESH_COLMAP_PATCH_CACHE_SIZE}" \
        "${FIPMESH_COLMAP_FUSION_CACHE_SIZE}" \
        "${FIPMESH_COLMAP_FUSION_USE_CACHE}"

    status "COLMAP sparse detect: sift_max_features=%s peak=%s edge=%s dsp=%s affine=%s" \
        "${FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES}" \
        "${FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD}" \
        "${FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD}" \
        "${FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING}" \
        "${FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE}"

    status "COLMAP sparse match: guided=%s max_num_matches=%s" \
    "${FIPMESH_COLMAP_MATCH_GUIDED}" \
    "${FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES}"

    if [ -n "${FIPMESH_COLMAP_IMAGES_SECONDARY:-}" ]; then
        status "secondary image directory: %s" "${FIPMESH_COLMAP_IMAGES_SECONDARY}"
        status "secondary pre-rotate: %s deg around %s-axis" \
          "${FIPMESH_COLMAP_SECONDARY_ROTATE_DEG:-180}" \
          "${FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS:-y}"
        status "secondary extra rotate xyz: (%s, %s, %s)" \
          "${FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X:-0}" \
          "${FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y:-0}" \
          "${FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z:-0}"
        status "secondary pre-translate: (%s, %s, %s)" \
          "${FIPMESH_COLMAP_SECONDARY_TRANSLATE_X:-0}" \
          "${FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y:-0}" \
          "${FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z:-0}"
        status "secondary align mode: %s" "${FIPMESH_COLMAP_SECONDARY_ALIGN_MODE:-auto}"
    fi
fi
status "primary image directory: %s" "$P_IMGDIR"
if [ -n "${FIPMESH_COLMAP_IMAGES_SECONDARY:-}" ]; then
    status "secondary image directory: %s" "${FIPMESH_COLMAP_IMAGES_SECONDARY}"
fi
status "output directory: %s" "$OUT_DIR"

# create intitial cloud 
src/main.sh "$P_IMGDIR" "${S_IMGDIR:-}" "$OUT_DIR"
rc=$?
if [ $rc -eq 0 ]; then
    if [ -n "$SDIR_IS_NEEDED" ]; then
        status "successfully created initial cloud from %s and %s" "$P_IMGDIR" "$S_IMGDIR"
    else
        status "successfully created initial cloud from %s" "$P_IMGDIR"
    fi
else
    if [ -n "$SDIR_IS_NEEDED" ]; then
        die "failed to create initial cloud from %s and %s" "$P_IMGDIR" "$S_IMGDIR"
    fi
    die "failed to create initial cloud from %s" "$P_IMGDIR"
fi



#./remesh_dense_cloud.sh $OUT_DIR
