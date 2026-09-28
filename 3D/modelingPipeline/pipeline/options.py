"""Choice lists shared by the settings registry and the UI."""

STAGE_NAMES = [
    "1. Background Removal",
    "2. COLMAP MVS",
    "3. FPFH Alignment",
    "4. Mesh Reconstruction",
]

VIEW_LABELS = [
    "View Photos",
    "View Cloud",
    "View Merged",
    "View Mesh",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

ARTIFACT_TYPES = ["tablet", "papyrus"]

BACKGROUNDS = ["white", "black", "transparent"]

REMBG_MODELS = [
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
    "isnet-general-use",
    "isnet-anime",
    "silueta",
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "bria-rmbg",
    "sam",
    "u2net_custom",
    "dis_custom",
    "ben_custom",
]

COLMAP_QUALITIES = ["extreme", "high", "medium", "low"]

# "(COLMAP default)" means: don't set FIPMESH_COLMAP_CAMERA_MODEL at all.
CAMERA_MODEL_DEFAULT = "(COLMAP default)"
CAMERA_MODELS = [CAMERA_MODEL_DEFAULT, "SIMPLE_RADIAL", "RADIAL", "OPENCV", "PINHOLE"]

SEC_ALIGN_MODES = ["auto", "manual"]

ALIGN_METHODS = ["opening", "fpfh", "collapse", "all"]

SEAM_MODES = ["point", "patch"]
