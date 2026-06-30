#!/usr/bin/env python3
"""Scan artifacts/ and backgrounds/, writing a manifest.json for each.

The site is static and can't list directories itself, so it reads
artifacts/manifest.json and backgrounds/manifest.json instead. Run:

    python3 tools/build_manifest.py

any time you add, remove, or edit an artifact or background folder, and
before you deploy. Wire it into the end of the modeling pipeline if you
want the manifests to stay current automatically.

== artifacts/ ==

Each artifact folder must contain exactly one model file and one .txt
file. The model can be any of:

  - a single .glb (preferred — everything packed into one file)
  - a loose .gltf, alongside whatever .bin/texture files it references
  - a Wavefront .obj, alongside its companion .mtl (and textures) if it
    has one — OBJ has no materials of its own, so without a .mtl the
    model loads with a flat default material

If a folder has more than one of these, .glb wins, then .gltf, then .obj.

The .txt file is written like this (field order doesn't matter):

    Name: Cuneiform Tablet 12
    Type: tablet
    Description: A clay tablet bearing an
    administrative record from the
    Third Dynasty of Ur.

Everything after "Description:" up to end of file is captured as the
description. A blank line starts a new paragraph.

== backgrounds/ ==

Each background folder must contain exactly six images, named for the
cube map faces: px, nx, py, ny, pz, nz (any of .jpg/.jpeg/.png/.webp).
The folder name becomes the option's id and display name — no .txt
file needed. A folder missing any face is skipped. See
backgrounds/README.md for details.
"""
import json
import re
import sys
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
FIELD_RE = re.compile(r"^\s*(name|type|description)\s*:\s*(.*)$", re.IGNORECASE)
MODEL_PATTERNS = ["*.glb", "*.gltf", "*.obj"]

BACKGROUNDS_DIR = Path(__file__).resolve().parent.parent / "backgrounds"
BACKGROUNDS_MANIFEST_PATH = BACKGROUNDS_DIR / "manifest.json"
CUBE_FACES = ["px", "nx", "py", "ny", "pz", "nz"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def find_model_file(folder):
    """Return the folder's model file, preferring .glb, then .gltf, then .obj."""
    for pattern in MODEL_PATTERNS:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches
    return []


def parse_info_txt(text):
    fields = {"name": [], "type": [], "description": []}
    current = None
    for raw_line in text.splitlines():
        match = FIELD_RE.match(raw_line)
        if match:
            current = match.group(1).lower()
            rest = match.group(2).strip()
            if rest:
                fields[current].append(rest)
        elif current:
            fields[current].append(raw_line.strip())

    def join(key):
        paragraphs, buf = [], []
        for line in fields[key]:
            if line == "":
                if buf:
                    paragraphs.append(" ".join(buf))
                    buf = []
            else:
                buf.append(line)
        if buf:
            paragraphs.append(" ".join(buf))
        return "\n\n".join(paragraphs)

    return {"name": join("name"), "type": join("type"), "description": join("description")}


def build():
    if not ARTIFACTS_DIR.exists():
        print(f"No artifacts/ directory found at {ARTIFACTS_DIR}", file=sys.stderr)
        return 1

    artifacts = []
    for folder in sorted(ARTIFACTS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        models = find_model_file(folder)
        txts = sorted(folder.glob("*.txt"))

        if not models:
            print(f"skip '{folder.name}': no .glb/.gltf/.obj model file found", file=sys.stderr)
            continue
        if not txts:
            print(f"skip '{folder.name}': no .txt file found", file=sys.stderr)
            continue
        if len(models) > 1:
            print(f"warn '{folder.name}': multiple model files found, using '{models[0].name}'", file=sys.stderr)
        if len(txts) > 1:
            print(f"warn '{folder.name}': multiple .txt files found, using '{txts[0].name}'", file=sys.stderr)

        model_file = models[0]

        info = parse_info_txt(txts[0].read_text(encoding="utf-8"))
        if not info["name"]:
            info["name"] = folder.name
        if not info["type"]:
            info["type"] = "other"

        entry = {
            "id": folder.name,
            "name": info["name"],
            "type": info["type"],
            "description": info["description"],
            "model": f"{folder.name}/{model_file.name}",
        }

        if model_file.suffix.lower() == ".obj":
            mtls = sorted(folder.glob("*.mtl"))
            if mtls:
                entry["mtl"] = f"{folder.name}/{mtls[0].name}"
                if len(mtls) > 1:
                    print(f"warn '{folder.name}': multiple .mtl files found, using '{mtls[0].name}'", file=sys.stderr)
            else:
                print(f"note '{folder.name}': .obj with no .mtl — will load with a default material", file=sys.stderr)

        artifacts.append(entry)

    artifacts.sort(key=lambda a: a["name"].lower())
    MANIFEST_PATH.write_text(json.dumps({"artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(artifacts)} artifact(s) to {MANIFEST_PATH}")
    return 0


def find_face_file(folder, face):
    for ext in IMAGE_EXTENSIONS:
        matches = sorted(folder.glob(f"{face}{ext}")) + sorted(folder.glob(f"{face}{ext.upper()}"))
        if matches:
            return matches[0]
    return None


def titleize(folder_name):
    return re.sub(r"[-_]+", " ", folder_name).strip().title()


def build_backgrounds():
    if not BACKGROUNDS_DIR.exists():
        return 0

    backgrounds = []
    for folder in sorted(BACKGROUNDS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        faces = {}
        missing = []
        for face in CUBE_FACES:
            file = find_face_file(folder, face)
            if file:
                faces[face] = f"{folder.name}/{file.name}"
            else:
                missing.append(face)

        if missing:
            print(f"skip background '{folder.name}': missing face(s) {', '.join(missing)}", file=sys.stderr)
            continue

        backgrounds.append({"id": folder.name, "name": titleize(folder.name), "faces": faces})

    backgrounds.sort(key=lambda b: b["name"].lower())
    BACKGROUNDS_MANIFEST_PATH.write_text(
        json.dumps({"backgrounds": backgrounds}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(backgrounds)} background(s) to {BACKGROUNDS_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(build() or build_backgrounds())
