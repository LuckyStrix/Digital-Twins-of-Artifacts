set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRAM=$0
# we never run this directly so there's no reason to have nice argument parsing.

# -------------------------------
# helper functions
# -------------------------------
status() {
    printf "%s: " "$PROGRAM"
    printf "\e[32mstatus\e[0m"
    printf ": $1\n" "${@:2}" >&2 
}

warn() {
    printf "%s: " "$PROGRAM"
    printf "\e[33mwarning\e[0m"
    printf ": $1\n" "$PROGRAM" "${@:2}" >&2 
}

die() {
    printf "%s: " "$PROGRAM"
    printf "\e[31merror\e[0m"
    printf ": $1\n" "${@:2}" >&2
    exit 1
}


# -------------------------------
# COLMAP MVS pipeline
# -------------------------------
cv_run_colmap_mvs_pipeline() {
    local p_imgdir="$1"
    local s_imgdir="$2"

    local workspace_dir="$3"
    local dense_cloud_out="$4"
    local script="src/run_colmap_mvs.py"

    if [ -z "$p_imgdir" ] || [ -z "$workspace_dir" ] || [ -z "$dense_cloud_out" ]; then
        warn "MVS pipeline skipped: invalid path args"
        return 1
    fi

    if [ "${FIPMESH_SKIP_MVS:-0}" = "1" ]; then
        status "skipping MVS pipeline (FIPMESH_SKIP_MVS=1)"
        return 2
    fi

    if [ ! -r "$script" ]; then
        warn "MVS script not found/readable at $script"
        return 3
    fi

    if ! python3 -c "print(0)" >/dev/null 2>&1; then
        warn "python3 not available, cannot run COLMAP MVS pipeline"
        return 4
    fi

    status "running COLMAP MVS pipeline to densify cloud"
    python3 "$script" \
        --images "$p_imgdir" \
        ${s_imgdir:+--images-secondary "$s_imgdir"} \
        --workspace "$workspace_dir" \
        --dense-cloud-out "$dense_cloud_out" \
        --single-camera-per-folder 1 \
        --clean-workspace || {
        warn "COLMAP MVS pipeline failed"
        return 6
    }

    status "MVS dense cloud ready: $dense_cloud_out"
    return 0
}

# -------------------------------
# Open3D reconstruction pipeline
# -------------------------------
cv_reconstruct_mesh_pipeline() {
    local input_cloud="$1"
    local output_mesh="$2"
    local output_clean_cloud="$3"
    local output_decimated_mesh="$4"
    local script="src/reconstruct_mesh.py"
    if [ -z "$input_cloud" ] || [ -z "$output_mesh" ] || [ -z "$output_clean_cloud" ] || [ -z "$output_decimated_mesh" ]; then
        warn "reconstruction pipeline skipped: invalid path args"
        return 1
    fi

    if [ "${FIPMESH_SKIP_RECON:-0}" = "1" ]; then
        status "skipping reconstruction pipeline (FIPMESH_SKIP_RECON=1)"
        return 0
    fi

    if [ ! -r "$script" ]; then
        warn "reconstruction script not found/readable at $script"
        return 2
    fi

    if ! python3 -c "import open3d" >/dev/null 2>&1; then
        warn "Open3D not available for mesh reconstruction. Install \`open3d\` to enable pipeline."
        return 3
    fi

    status "running Open3D reconstruction pipeline"
    python3 "$script" \
        --input "$input_cloud" \
        --output "$output_mesh" \
        --output-clean-cloud "$output_clean_cloud" \
        --output-decimated "$output_decimated_mesh" \
        --max-input-points 0 \
        --dbscan-max-points 0 \
        --outlier-nb-neighbors 32 \
        --outlier-std-ratio 0 \
        --radius-outlier-nb-points 0 \
        --radius-outlier-radius-factor 2.2 \
        --dbscan-min-points 0 \
        --dbscan-eps-factor 2.2 \
        --dbscan-keep-largest 1 \
        --dbscan-min-cluster-ratio 0.02 \
        --normal-max-nn 96 \
        --normal-orient-k 64 \
        --poisson-depth 10 \
        --poisson-linear-fit \
        --density-trim-quantile 0.03 \
        --component-min-ratio 0.01 \
        --component-min-triangles 1000 \
        --component-max-count 2 \
        --smooth-iters 1 \
        --decimate-target-triangles 300000 || {
        warn "reconstruction pipeline failed"
        return 5
    }

    status "reconstruction pipeline complete: $output_mesh"
    return 0
}

# -------------------------------
# main script
# -------------------------------
if [ "$#" -lt 2 ]; then
    die "Usage: $0 <primary_image_dir> [secondary_image_dir] <out_dir>"
fi

P_IMGDIR="$1"
if [ "$#" -ge 3 ]; then
    S_IMGDIR="$2"
    OUT_DIR="$3"
else
    S_IMGDIR=""
    OUT_DIR="$2"
fi


# paths (default output)
RAW_CLOUD_OBJ="mesh.obj"
MVS_WORKSPACE_DIR=$OUT_DIR
MVS_DENSE_CLOUD_PLY="${OUT_DIR}/fused.ply"
RECON_MESH_OBJ="${OUT_DIR}/recon_mesh_recon.obj"
CLEAN_CLOUD_PLY="${OUT_DIR}/clean_cloud.ply"
DECIMATED_MESH_OBJ="${OUT_DIR}/recon_mesh_recon_decimated.obj"


touch "$MVS_DENSE_CLOUD_PLY"
warn "created file: $MVS_DENSE_CLOUD_PLY"

cv_run_colmap_mvs_pipeline "$P_IMGDIR" "$S_IMGDIR" "$MVS_WORKSPACE_DIR" "$MVS_DENSE_CLOUD_PLY" || {
    die "COLMAP MVS failed; legacy sparse triangulation disabled"
}

status "using dense MVS cloud for meshing: $MVS_DENSE_CLOUD_PLY"

cv_reconstruct_mesh_pipeline "$MVS_DENSE_CLOUD_PLY" \
                             "$RECON_MESH_OBJ" \
                             "$CLEAN_CLOUD_PLY" \
                             "$DECIMATED_MESH_OBJ" || {
    die "dense-cloud meshing pipeline failed"
}
# if no errors in pipeline, exit success
exit 0
