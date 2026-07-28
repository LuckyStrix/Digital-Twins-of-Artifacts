"""Generate the project QR codes and a printable 9x11 in sheet.

Outputs (next to this script):
  github_qr.png            - QR for the GitHub repository
  efip2026_qr.png          - QR for the EFIP 2026 site
  printable_qr_codes.pdf   - 9 x 11 in page, each QR printed at 3 x 3 in

Requires: qrcode, pillow
"""

from pathlib import Path

import qrcode
from PIL import Image, ImageDraw, ImageFont

DPI = 300
PAGE_W_IN, PAGE_H_IN = 9, 11
QR_IN = 3.0  # printed size of each QR module area

OUT_DIR = Path(__file__).resolve().parent

CODES = [
    ("Project GitHub", "https://github.com/LuckyStrix/Digital-Twins-of-Artifacts", "github_qr.png"),
    ("EFIP 2026", "https://www.cis.rit.edu/fip/efip2026/index.html", "efip2026_qr.png"),
]

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def make_qr(data, px):
    """Render `data` as a QR image exactly `px` pixels square."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=0,  # quiet zone is added by the page layout below
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    # Nearest-neighbour keeps module edges crisp for print.
    return img.resize((px, px), Image.NEAREST)


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def text_center(draw, cx, y, text, font, fill="black"):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (r - l) / 2 - l, y - t), text, font=font, fill=fill)
    return b - t


def main():
    qr_px = int(QR_IN * DPI)
    page = Image.new("RGB", (PAGE_W_IN * DPI, PAGE_H_IN * DPI), "white")
    draw = ImageDraw.Draw(page)

    title_font = load_font(FONT_BOLD, 84)
    label_font = load_font(FONT_BOLD, 62)
    url_font = load_font(FONT_REG, 34)

    text_center(draw, page.width // 2, int(0.75 * DPI), "Digital Twins of Artifacts", title_font)

    # Two blocks stacked vertically, each: label, QR (3x3 in), url.
    block_top = [int(1.9 * DPI), int(6.2 * DPI)]

    for (label, url, filename), top in zip(CODES, block_top):
        qr = make_qr(url, qr_px)
        qr.save(OUT_DIR / filename, dpi=(DPI, DPI))

        text_center(draw, page.width // 2, top, label, label_font)

        qr_y = top + int(0.55 * DPI)
        qr_x = (page.width - qr_px) // 2
        page.paste(qr, (qr_x, qr_y))
        # Thin crop guide showing the exact 3x3 in footprint.
        draw.rectangle(
            [qr_x - 1, qr_y - 1, qr_x + qr_px, qr_y + qr_px],
            outline=(200, 200, 200),
            width=2,
        )

        text_center(draw, page.width // 2, qr_y + qr_px + int(0.22 * DPI), url, url_font)

    pdf_path = OUT_DIR / "printable_qr_codes.pdf"
    page.save(pdf_path, "PDF", resolution=DPI)
    print(f"wrote {pdf_path} ({PAGE_W_IN}x{PAGE_H_IN} in @ {DPI} dpi, QR {QR_IN}x{QR_IN} in)")


if __name__ == "__main__":
    main()
