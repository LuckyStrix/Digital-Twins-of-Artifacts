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

Two-sided artifacts (a papyrus scanned front and back) are the one
exception: put both .glb files in the same folder and they become a
single artifact with a "Flip to other side" button in the viewer. The
alphabetically-first file is the front, so name them in the order you
want them seen (e.g. side1.glb / side2.glb). This applies only to
exactly two .glb files — three or more, or loose .gltf/.obj, still
resolve to one model, since their companion .bin/.mtl files can't be
told apart from a second side.

The .txt file is written like this (field order doesn't matter):

    Name: Cuneiform Tablet 12
    Type: tablet
    Description: A clay tablet bearing an
    administrative record from the
    Third Dynasty of Ur.
    Link: https://example.edu/collection/tablet-12
    Link Label: Cary Collection Link

Everything after "Description:" up to end of file is captured as the
description. A blank line starts a new paragraph.

"Link" and "Link Label" are both optional. If "Link" is present, the
viewer shows it as a clickable button in the info panel, labeled with
"Link Label" (or a generic default if omitted).

== msi/ ==

An artifact folder may also (or instead — see below) contain an msi/
subfolder holding multispectral band data. This data comes from MISHA
(Multispectral Imaging System for Historical Artifacts), an EXTERNAL,
independent imaging project at RIT's Cultural Heritage Imaging,
Preservation, and Research program (https://www.rit.edu/chipr/misha) — not
one of this repo's own 2D/3D capture systems. See website/tools/
build_msi_assets.py, which generates this subfolder from a raw MISHA
export:

    msi/bands/<wavelength>nm.webp   one downsampled, contrast-stretched
                                     image per captured wavelength
    msi/msi_manifest.json           band list + wavelengths, machine-written
                                     by build_msi_assets.py (never hand-edit)
    msi/recipes.json                optional, hand-authored curator presets
                                     for the web equalizer viewer (never
                                     touched by build_msi_assets.py)

If msi/msi_manifest.json exists, this script adds an "msi" pointer (a path
to that file) to the artifact's manifest entry, the same way "model"/
"back"/"mtl" point at model files. An artifact with msi/ data and no
.glb/.gltf/.obj at all is valid — it's MSI-only, and the homepage/viewer
route it to the "MISHA-Imaged Artifacts" section instead of the regular
model viewer.

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
FIELD_RE = re.compile(r"^\s*(name|type|description|link\s*label|link)\s*:\s*(.*)$", re.IGNORECASE)
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


def find_msi_manifest(folder):
    """Return the folder's msi/msi_manifest.json path if it exists, else None."""
    path = folder / "msi" / "msi_manifest.json"
    return path if path.exists() else None


def is_two_sided(models):
    """True if these model files are a front/back pair rather than one model.

    Exactly two .glb files in one folder means one artifact scanned from both
    sides (the capture pipeline writes a render.glb per side). Anything else —
    one model, three or more, or loose .gltf/.obj whose companion .bin/.mtl
    files can't be told apart from a second side — is a single model.
    """
    return len(models) == 2 and all(m.suffix.lower() == ".glb" for m in models)


def parse_info_txt(text):
    fields = {"name": [], "type": [], "description": [], "link": [], "linklabel": []}
    current = None
    for raw_line in text.splitlines():
        match = FIELD_RE.match(raw_line)
        if match:
            current = re.sub(r"\s+", "", match.group(1).lower())
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

    return {
        "name": join("name"),
        "type": join("type"),
        "description": join("description"),
        "link": join("link"),
        "linklabel": join("linklabel"),
    }


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
        msi_manifest = find_msi_manifest(folder)

        if not models and not msi_manifest:
            print(f"skip '{folder.name}': no model file and no msi/ data found", file=sys.stderr)
            continue
        if not txts:
            print(f"skip '{folder.name}': no .txt file found", file=sys.stderr)
            continue
        two_sided = is_two_sided(models) if models else False
        if len(models) > 1 and not two_sided:
            print(f"warn '{folder.name}': multiple model files found, using '{models[0].name}'", file=sys.stderr)
        if len(txts) > 1:
            print(f"warn '{folder.name}': multiple .txt files found, using '{txts[0].name}'", file=sys.stderr)

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
        }

        if models:
            model_file = models[0]
            entry["model"] = f"{folder.name}/{model_file.name}"

            if two_sided:
                entry["back"] = f"{folder.name}/{models[1].name}"

            if model_file.suffix.lower() == ".obj":
                mtls = sorted(folder.glob("*.mtl"))
                if mtls:
                    entry["mtl"] = f"{folder.name}/{mtls[0].name}"
                    if len(mtls) > 1:
                        print(f"warn '{folder.name}': multiple .mtl files found, using '{mtls[0].name}'", file=sys.stderr)
                else:
                    print(f"note '{folder.name}': .obj with no .mtl — will load with a default material", file=sys.stderr)

        if info["link"]:
            entry["link"] = info["link"]
            if info["linklabel"]:
                entry["linkLabel"] = info["linklabel"]

        # msi is an EXTERNAL data source (MISHA, RIT CHIPR — see the == msi/ ==
        # section above) — this pointer just lets the site find it, it doesn't
        # imply the data was captured by this project's own pipelines.
        if msi_manifest:
            entry["msi"] = f"{folder.name}/msi/msi_manifest.json"

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
