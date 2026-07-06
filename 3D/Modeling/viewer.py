#!/usr/bin/env python3
"""
viewer.py -- Spawned as a subprocess by app.py to open 3D files in Open3D.

Usage:
    python viewer.py <path.ply|path.obj|path.gltf>
"""
import sys
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("error: open3d not available", file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 2:
    print("usage: viewer.py <file>", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
ext = Path(path).suffix.lower()

mesh_exts = {".obj", ".gltf", ".glb", ".stl", ".ply"}

if ext in {".obj", ".gltf", ".glb", ".stl"}:
    geom = o3d.io.read_triangle_mesh(path)
    if len(geom.triangles) == 0 and ext == ".ply":
        geom = o3d.io.read_point_cloud(path)
    else:
        geom.compute_vertex_normals()
elif ext == ".ply":
    # Try mesh first, fall back to point cloud
    geom = o3d.io.read_triangle_mesh(path)
    if len(geom.triangles) > 0:
        geom.compute_vertex_normals()
    else:
        geom = o3d.io.read_point_cloud(path)
else:
    geom = o3d.io.read_point_cloud(path)

if geom is None or (hasattr(geom, "points") and len(geom.points) == 0
                    and hasattr(geom, "triangles") and len(geom.triangles) == 0):
    print(f"error: could not load geometry from {path}", file=sys.stderr)
    sys.exit(1)

o3d.visualization.draw_geometries(
    [geom],
    window_name=f"View: {Path(path).name}",
    width=1024,
    height=768,
)
