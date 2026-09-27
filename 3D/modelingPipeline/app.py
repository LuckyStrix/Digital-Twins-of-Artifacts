#!/usr/bin/env python3
"""
Automatic Tablet Reconstruction Pipeline — terminal UI

Stages:
  1. Background removal     (process_photos.py)
  2. COLMAP MVS             (run.sh, once per side with FIPMESH_SKIP_RECON=1)
  3. FPFH alignment         (alignment/run.py → output/aligned_cloud/merged_fpfh.ply)
  4. Mesh reconstruction    (src/reconstruct_mesh.py → output/recon/ + output/model.gltf)

Run:  python3 app.py   (in any terminal: Windows Terminal under WSL, Linux, SSH)

The UI lives in tui/, the pipeline logic in pipeline/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tui.app import main
except ModuleNotFoundError as e:
    if e.name != "textual":
        raise
    sys.exit("This app needs Textual:  python3 -m pip install --break-system-packages textual")

if __name__ == "__main__":
    main()
