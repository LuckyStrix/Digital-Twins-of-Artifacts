#!/usr/bin/env python3
"""Generate poster-styled SVG pipeline diagrams for the repo READMEs.

Regenerate after changing pipeline stages, rig descriptions, or colors:

    python3 docs/diagrams/generate.py docs/diagrams

Styling mirrors Research_Symposium_Documents/Poster/poster_landscape.html
(step cards with a colored top bar, serif step numbers, small line-art rig
icons) so the GitHub docs and the poster read as one visual system.
"""
import os

SERIF = "Palatino Linotype, Palatino, 'Book Antiqua', Georgia, serif"
SANS = "Segoe UI, 'Helvetica Neue', Helvetica, Arial, sans-serif"

INK = "#241e19"
INK_SOFT = "#5c5348"
INK_FAINT = "#8a8073"
RULE = "#d8d2c6"
PAPER = "#ffffff"
PAPER_2 = "#f2f1ef"

ORANGE = "#f76902"
ORANGE_DEEP = "#b44a00"
TEAL = "#1e5f74"
TEAL_DEEP = "#164a5b"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rig_icon_2d(cx, cy, s=0.8):
    """Poster's compass-light icon (viewBox 0 0 200 150), centered at cx,cy."""
    # original center ~ (100, 75)
    ox, oy = 100, 75
    def T(x, y):
        return (cx + (x - ox) * s, cy + (y - oy) * s)
    px, py = T(68, 52)
    parts = [f'<g>']
    parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{64*s:.1f}" height="{46*s:.1f}" rx="2" '
                  f'fill="#e4d6b8" stroke="#b7a271" stroke-width="1.5"/>')
    tx, ty = T(100, 79)
    parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-family="{SERIF}" '
                  f'font-size="{9*s:.1f}" fill="#7a6a44">papyrus</text>')
    lines = [(100, 26, 100, 50), (174, 75, 134, 75), (100, 124, 100, 100), (26, 75, 66, 75)]
    parts.append(f'<g stroke="{ORANGE}" stroke-width="1.4" stroke-dasharray="3 3">')
    for x1, y1, x2, y2 in lines:
        ax1, ay1 = T(x1, y1)
        ax2, ay2 = T(x2, y2)
        parts.append(f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}"/>')
    parts.append('</g>')
    dots = [(100, 22), (178, 75), (100, 128), (22, 75)]
    parts.append(f'<g fill="{ORANGE}">')
    for x, y in dots:
        dx, dy = T(x, y)
        parts.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="{6*s:.1f}"/>')
    parts.append('</g>')
    labels = [(100, 12, "N"), (192, 79, "E"), (100, 146, "S"), (8, 79, "W")]
    parts.append(f'<g font-family="{SANS}" font-size="{9*s:.1f}" font-weight="700" fill="{INK}">')
    for x, y, t in labels:
        lx, ly = T(x, y)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">{t}</text>')
    parts.append('</g>')
    parts.append('</g>')
    return "\n".join(parts)


def rig_icon_3d(cx, cy, s=0.8, uid="ar"):
    ox, oy = 100, 75
    def T(x, y):
        return (cx + (x - ox) * s, cy + (y - oy) * s)
    parts = ['<g>']
    ex, ey = T(100, 118)
    parts.append(f'<ellipse cx="{ex:.1f}" cy="{ey:.1f}" rx="{58*s:.1f}" ry="{16*s:.1f}" '
                  f'fill="#cfe0e6" stroke="{TEAL}" stroke-width="1.5"/>')
    rx, ry = T(86, 92)
    parts.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{28*s:.1f}" height="{22*s:.1f}" rx="2" '
                  f'fill="#b98d5a" stroke="#7a5a34" stroke-width="1.5"/>')
    p1 = T(60, 120)
    p2 = T(140, 120)
    rxx = 40 * s
    ryy = 12 * s
    parts.append(f'<defs><marker id="{uid}" markerWidth="7" markerHeight="7" refX="5" refY="3" '
                  f'orient="auto"><path d="M0 0l6 3-6 3z" fill="{TEAL}"/></marker></defs>')
    parts.append(f'<path d="M{p1[0]:.1f} {p1[1]:.1f} A {rxx:.1f} {ryy:.1f} 0 0 0 {p2[0]:.1f} {p2[1]:.1f}" '
                  f'fill="none" stroke="{TEAL}" stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#{uid})"/>')
    cams = [(24, 40), (90, 20), (156, 40)]
    parts.append(f'<g fill="{TEAL}">')
    for x, y in cams:
        cxp, cyp = T(x, y)
        parts.append(f'<rect x="{cxp:.1f}" y="{cyp:.1f}" width="{20*s:.1f}" height="{14*s:.1f}" rx="2"/>')
    parts.append('</g>')
    rays = [(34, 54, 92, 98), (100, 34, 100, 90), (166, 54, 108, 98)]
    parts.append(f'<g stroke="{TEAL}" stroke-width="1.1" stroke-dasharray="2 3">')
    for x1, y1, x2, y2 in rays:
        ax1, ay1 = T(x1, y1)
        ax2, ay2 = T(x2, y2)
        parts.append(f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}"/>')
    parts.append('</g>')
    parts.append('</g>')
    return "\n".join(parts)


def wrapped_lines(x, y, lines, size, fill, weight="400", family=SANS, anchor="start", lh=None):
    lh = lh or size * 1.35
    out = []
    for i, line in enumerate(lines):
        out.append(f'<text x="{x:.1f}" y="{y + i*lh:.1f}" text-anchor="{anchor}" '
                    f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
                    f'fill="{fill}">{esc(line)}</text>')
    return "\n".join(out)


def step_card(x, y, w, h, num, title, desc_lines, color):
    parts = [f'<g>']
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="7" fill="{PAPER_2}"/>')
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="4.5" rx="2.2" fill="{color}"/>')
    pad = 13
    tx = x + pad
    ny = y + 34
    parts.append(f'<text x="{tx:.1f}" y="{ny:.1f}" font-family="{SERIF}" font-weight="700" '
                  f'font-size="24" fill="{color}">{num}</text>')
    ty = ny + 22
    parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-family="{SANS}" font-weight="800" '
                  f'font-size="14.5" fill="{INK}">{esc(title)}</text>')
    dy = ty + 20
    parts.append(wrapped_lines(tx, dy, desc_lines, 11, INK_SOFT, lh=15))
    parts.append('</g>')
    return "\n".join(parts)


def arrow(x, y, color, size=22):
    return (f'<text x="{x:.1f}" y="{y+size*0.35:.1f}" text-anchor="middle" font-family="{SANS}" '
            f'font-weight="700" font-size="{size}" fill="{color}">&#8594;</text>')


def eyebrow_tag(x, y, text, color):
    tw = 8.2 * len(text) + 18
    parts = [f'<rect x="{x:.1f}" y="{y-15:.1f}" width="{tw:.1f}" height="20" rx="10" fill="{color}"/>']
    parts.append(f'<text x="{x+tw/2:.1f}" y="{y-1:.1f}" text-anchor="middle" font-family="{SANS}" '
                  f'font-size="10.5" font-weight="700" letter-spacing="1.2" fill="{PAPER}">{esc(text.upper())}</text>')
    return "\n".join(parts)


def build_pipeline_svg(*, tag, tag_color, title, subtitle, rig_icon_fn, rig_title, rig_lines,
                        steps, step_color, out_label, filename):
    n = len(steps)
    card_w, card_h, arrow_w = 172, 122, 30
    margin = 22
    flow_w = n * card_w + (n - 1) * arrow_w
    rig_panel_w = 190
    rig_panel_h = 122
    content_w = max(flow_w, rig_panel_w + 26 + 560)
    W = content_w + margin * 2
    H = 40 + 20 + rig_panel_h + 46 + 20 + card_h + 34

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
                f'font-family="{SANS}">')
    svg.append(f'<rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="{PAPER}"/>')

    x0 = margin
    y = 34
    svg.append(eyebrow_tag(x0, y, tag, tag_color))
    svg.append(f'<text x="{x0 + 8.2*len(tag)+30:.1f}" y="{y-1:.1f}" font-family="{SANS}" '
                f'font-size="15.5" font-weight="800" fill="{INK}">{esc(title)}</text>')
    svg.append(f'<text x="{x0 + 8.2*len(tag)+30:.1f}" y="{y+16:.1f}" font-family="{SANS}" '
                f'font-size="11" font-weight="600" fill="{INK_FAINT}">{esc(subtitle)}</text>')

    ry = y + 26
    svg.append(f'<rect x="{x0:.1f}" y="{ry:.1f}" width="{rig_panel_w}" height="{rig_panel_h}" rx="8" '
                f'fill="{PAPER_2}"/>')
    icon_cx = x0 + rig_panel_w / 2
    icon_cy = ry + rig_panel_h / 2 - 8
    svg.append(rig_icon_fn(icon_cx, icon_cy, 0.62))
    svg.append(f'<text x="{icon_cx:.1f}" y="{ry+rig_panel_h-12:.1f}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="10.5" font-weight="700" fill="{tag_color}">CAPTURE RIG</text>')

    cap_x = x0 + rig_panel_w + 26
    cap_y = ry + 22
    svg.append(f'<text x="{cap_x:.1f}" y="{cap_y:.1f}" font-family="{SANS}" font-size="13.5" '
                f'font-weight="800" fill="{INK}">{esc(rig_title)}</text>')
    svg.append(wrapped_lines(cap_x, cap_y + 21, rig_lines, 11.5, INK_SOFT, lh=17))

    # connector from rig row down into the pipeline row
    conn_x = x0 + rig_panel_w / 2
    conn_y1 = ry + rig_panel_h
    conn_y2 = conn_y1 + 40
    svg.append(f'<line x1="{conn_x:.1f}" y1="{conn_y1:.1f}" x2="{conn_x:.1f}" y2="{conn_y2-10:.1f}" '
                f'stroke="{tag_color}" stroke-width="1.6" stroke-dasharray="3 3"/>')
    svg.append(f'<path d="M{conn_x-5:.1f} {conn_y2-10:.1f} L{conn_x+5:.1f} {conn_y2-10:.1f} '
                f'L{conn_x:.1f} {conn_y2:.1f} Z" fill="{tag_color}"/>')

    py = conn_y2 + 14
    svg.append(eyebrow_tag(x0, py, out_label, step_color))
    svg.append(f'<text x="{x0 + 8.2*len(out_label)+30:.1f}" y="{py-1:.1f}" font-family="{SANS}" '
                f'font-size="12.5" font-weight="700" fill="{INK_FAINT}">'
                f'{n} stages, {steps[0][0]}–{steps[-1][0]}</text>')

    fy = py + 20
    fx = x0
    for i, (num, stitle, lines) in enumerate(steps):
        svg.append(step_card(fx, fy, card_w, card_h, num, stitle, lines, step_color))
        fx += card_w
        if i != len(steps) - 1:
            svg.append(arrow(fx + arrow_w / 2, fy + card_h / 2, step_color))
            fx += arrow_w

    svg.append('</svg>')
    out = "\n".join(svg)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote", path, f"{W:.0f}x{H:.0f}")


def pill_box(x, y, w, h, label, sublabel, color):
    parts = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="9" '
             f'fill="{PAPER_2}" stroke="{color}" stroke-width="1.6"/>']
    parts.append(f'<text x="{x+w/2:.1f}" y="{y+h/2-3:.1f}" text-anchor="middle" '
                 f'font-family="{SANS}" font-size="13" font-weight="800" fill="{INK}">{esc(label)}</text>')
    parts.append(f'<text x="{x+w/2:.1f}" y="{y+h/2+15:.1f}" text-anchor="middle" '
                 f'font-family="{SANS}" font-size="10.5" font-weight="600" fill="{INK_FAINT}">{esc(sublabel)}</text>')
    return "\n".join(parts)


def output_chip(x, y, w, h, label, color):
    parts = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h/2:.1f}" fill="{color}"/>']
    parts.append(f'<text x="{x+w/2:.1f}" y="{y+h/2+4.5:.1f}" text-anchor="middle" '
                 f'font-family="{SANS}" font-size="12" font-weight="700" fill="{PAPER}">{esc(label)}</text>')
    return "\n".join(parts)


def build_overview_svg(filename):
    row_h = 150
    top_pad = 26
    row_gap = 18
    H = top_pad + row_h * 2 + row_gap + 34

    icon_w, icon_h = 130, 100
    pill_w, pill_h = 230, 58
    out_w, out_h = 118, 34
    gap = 34

    x0 = 30
    icon_x = x0
    pill_x = icon_x + icon_w + gap
    out_x = pill_x + pill_w + gap
    viewer_x = out_x + out_w + gap + 46
    viewer_w = 150
    W = viewer_x + viewer_w + 30

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{SANS}">']
    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAPER}"/>')
    svg.append('<defs>'
                f'<marker id="ov-ar-a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
                f'<path d="M0 0l7 3-7 3z" fill="{ORANGE_DEEP}"/></marker>'
                f'<marker id="ov-ar-b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
                f'<path d="M0 0l7 3-7 3z" fill="{TEAL_DEEP}"/></marker>'
                '</defs>')

    rows = [
        dict(tag="2D · PHOTOMETRIC STEREO", tag_color=ORANGE_DEEP, icon_fn=rig_icon_2d,
             icon_label="Papyrus & flat manuscripts",
             pill="5-stage photometric-\nstereo pipeline", out_label="render.glb"),
        dict(tag="3D · PHOTOGRAMMETRY", tag_color=TEAL_DEEP, icon_fn=rig_icon_3d,
             icon_label="Cuneiform tablets & objects",
             pill="4-stage COLMAP\nreconstruction pipeline", out_label="model.gltf"),
    ]

    row_centers = []
    for i, row in enumerate(rows):
        ry = top_pad + i * (row_h + row_gap)
        cy = ry + row_h / 2
        row_centers.append(cy)

        svg.append(f'<text x="{icon_x:.1f}" y="{ry+13:.1f}" font-family="{SANS}" font-size="11" '
                    f'font-weight="800" letter-spacing="0.8" fill="{row["tag_color"]}">{esc(row["tag"])}</text>')

        icon_y = ry + 22
        svg.append(f'<rect x="{icon_x:.1f}" y="{icon_y:.1f}" width="{icon_w}" height="{icon_h}" rx="8" '
                    f'fill="{PAPER_2}"/>')
        svg.append(row["icon_fn"](icon_x + icon_w/2, icon_y + icon_h/2, 0.5))

        a1x = icon_x + icon_w
        a1y = icon_y + icon_h/2
        svg.append(arrow((a1x + pill_x)/2, a1y, row["tag_color"], size=18))

        pill_y = icon_y + icon_h/2 - pill_h/2
        p1, p2 = row["pill"].split("\n")
        svg.append(pill_box(pill_x, pill_y, pill_w, pill_h, p1, p2, row["tag_color"]))

        a2x = pill_x + pill_w
        svg.append(arrow((a2x + out_x)/2, a1y, row["tag_color"], size=18))

        out_y = icon_y + icon_h/2 - out_h/2
        svg.append(output_chip(out_x, out_y, out_w, out_h, row["out_label"], row["tag_color"]))

    viewer_y = top_pad + (row_h*2 + row_gap)/2 - 46
    viewer_h = 92

    for i, row in enumerate(rows):
        ry = top_pad + i * (row_h + row_gap)
        icon_y = ry + 22
        cy = icon_y + icon_h/2
        lx1 = out_x + out_w + 4
        ly1 = cy
        lx2 = viewer_x - 8
        ly2 = viewer_y + (viewer_h * (0.32 if i == 0 else 0.68))
        mk = "ov-ar-a" if i == 0 else "ov-ar-b"
        svg.append(f'<path d="M{lx1:.1f} {ly1:.1f} L{lx2-14:.1f} {ly1:.1f} L{lx2:.1f} {ly2:.1f}" '
                    f'fill="none" stroke="{row["tag_color"]}" stroke-width="1.8" '
                    f'marker-end="url(#{mk})"/>')

    svg.append(f'<rect x="{viewer_x-6:.1f}" y="{viewer_y:.1f}" width="{viewer_w}" height="{viewer_h}" rx="10" '
                f'fill="{INK}"/>')
    svg.append(f'<text x="{viewer_x-6+viewer_w/2:.1f}" y="{viewer_y+viewer_h/2-2:.1f}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="13.5" font-weight="800" fill="{PAPER}">Web</text>')
    svg.append(f'<text x="{viewer_x-6+viewer_w/2:.1f}" y="{viewer_y+viewer_h/2+16:.1f}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="13.5" font-weight="800" fill="{PAPER}">viewer</text>')

    svg.append('</svg>')
    out = "\n".join(svg)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote", path, f"{W}x{H}")


OUT_DIR = None


def main(out_dir):
    global OUT_DIR
    OUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)

    build_pipeline_svg(
        tag="Papyrus", tag_color=ORANGE_DEEP,
        title="2D Photometric-Stereo Pipeline", subtitle="Python · papyrus & flat manuscripts",
        rig_icon_fn=rig_icon_2d,
        rig_title="DSLR + Arduino lighting rig",
        rig_lines=["Tethered DSLR; lights fire from N/E/S/W", "(θ≈ 37°) with a polarizer stepper for",
                   "cross-/co-polarized pairs."],
        steps=[
            ("0", "Segment", ["ML segmentation", "isolates the fragment."]),
            ("1", "Calibrate", ["Copy-paper shots remove", "light falloff."]),
            ("2", "Core maps", ["Normal, albedo, specular,", "roughness."]),
            ("3", "Height", ["Frankot–Chellappa", "integration."]),
            ("4", "Bake", ["Maps baked into a", "three.js GLB."]),
        ],
        step_color=ORANGE_DEEP, out_label="Modeling pipeline",
        filename="2d-pipeline.svg",
    )

    build_pipeline_svg(
        tag="Tablets", tag_color=TEAL_DEEP,
        title="3D Photogrammetry Pipeline", subtitle="COLMAP / Open3D · cuneiform tablets & objects",
        rig_icon_fn=rig_icon_3d,
        rig_title="Multi-camera + turntable rig",
        rig_lines=["Cameras coordinated by an Arduino-driven", "turntable photograph the artifact from",
                   "many angles at once."],
        steps=[
            ("1", "Clean", ["Background removed from", "every photo."]),
            ("2", "Reconstruct", ["CUDA COLMAP SfM +", "dense stereo."]),
            ("3", "Align", ["FPFH + RANSAC + ICP", "merge both sides."]),
            ("4", "Mesh", ["Open3D Poisson →", "watertight model."]),
        ],
        step_color=TEAL_DEEP, out_label="Reconstruction pipeline",
        filename="3d-pipeline.svg",
    )

    build_overview_svg("system-overview.svg")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
