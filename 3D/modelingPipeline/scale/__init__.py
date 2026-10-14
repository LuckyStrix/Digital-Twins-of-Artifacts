"""Absolute scale from a reference ring fixed to the turntable.

colmap_model  read COLMAP TXT models, rescale models / workspaces, undistort pixels
ring          find the ring in the raw photos and fit one 3D circle to it
apply         Stage 2 hook: fit every sparse model, scale all or none, scale.json

No UI imports in here; numpy + OpenCV only (no scipy, no open3d).
"""
