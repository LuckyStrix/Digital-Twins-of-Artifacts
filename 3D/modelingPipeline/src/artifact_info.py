#!/usr/bin/env python3
"""
Write an info.txt metadata file alongside an exported model, in the format the
Digital-Twins-of-Artifacts website expects (website/artifacts/<name>/info.txt):

    Name: ...
    Type: ...
    Description: ...
    Link: ...          (optional)
    Link Label: ...    (optional)

Usable as a library (write_info_txt) or standalone:

    python -m src.artifact_info --output-dir path/to/session \
        --name "Sumerian Cuneiform tablet" --type tablet \
        --description "Clay tablet ..." \
        --link "https://example.com" --link-label "Source"
"""

from __future__ import annotations

import argparse
from pathlib import Path


def write_info_txt(
    dest_dir: Path,
    name: str,
    type_: str,
    description: str,
    link: str = "",
    link_label: str = "",
) -> Path:
    """Write dest_dir/info.txt and return its path. link/link_label are optional."""
    name = name.strip()
    type_ = type_.strip()
    description = description.strip()
    link = link.strip()
    link_label = link_label.strip()

    if not name:
        raise ValueError("name is required")
    if not type_:
        raise ValueError("type_ is required")
    if not description:
        raise ValueError("description is required")

    lines = [
        f"Name: {name}",
        f"Type: {type_}",
        f"Description: {description}",
    ]
    if link:
        lines.append(f"Link: {link}")
        if link_label:
            lines.append(f"Link Label: {link_label}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    info_path = dest_dir / "info.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory to write info.txt into.")
    parser.add_argument("--name", required=True, help="Artifact name.")
    parser.add_argument("--type", required=True, help="Artifact type (e.g. tablet, papyrus).")
    parser.add_argument("--description", required=True, help="Artifact description.")
    parser.add_argument("--link", default="", help="Optional source link URL.")
    parser.add_argument("--link-label", default="", help="Optional label for the link.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    info_path = write_info_txt(
        Path(args.output_dir),
        name=args.name,
        type_=args.type,
        description=args.description,
        link=args.link,
        link_label=args.link_label,
    )
    print(f"wrote {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
