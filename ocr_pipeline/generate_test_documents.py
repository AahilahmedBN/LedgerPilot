"""
LedgerPilot — Phase 2: Synthetic Test Document Generator

"""

import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size):
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    # Last resort: PIL's built-in font, but sized up so it's still legible
    # once degradation (blur/noise/rotation) is applied on top. Pillow
    # >=9.2 supports a size argument here; older Pillow silently ignores it.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        raise RuntimeError(
            "No usable TTF font found on this system, and this Pillow "
            "version's default font can't be resized. Install a font "
            "(e.g. DejaVu Sans) or upgrade Pillow with: "
            "pip install --upgrade pillow"
        )


def render_clean_receipt(lines, width=600):
    font = _load_font(22)
    line_height = 30
    height = 60 + line_height * len(lines)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 30
    for line in lines:
        draw.text((30, y), line, fill=(10, 10, 10), font=font)
        y += line_height
    return img


def _add_gaussian_noise(cv_img, sigma=12):
    noise = np.random.normal(0, sigma, cv_img.shape).astype(np.float32)
    noisy = cv_img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _add_uneven_lighting(cv_img):
    h, w = cv_img.shape[:2]
    gradient = np.tile(np.linspace(0.7, 1.15, w), (h, 1)).astype(np.float32)
    gradient = np.dstack([gradient] * 3) if len(cv_img.shape) == 3 else gradient
    lit = cv_img.astype(np.float32) * gradient
    return np.clip(lit, 0, 255).astype(np.uint8)


def _place_on_cluttered_background(receipt_cv, pad=120):
    h, w = receipt_cv.shape[:2]
    bg = np.full((h + pad * 2, w + pad * 2, 3), (120, 110, 95), dtype=np.uint8)  # wood-table-ish color
    bg = _add_gaussian_noise(bg, sigma=6)
    y0, x0 = pad, pad
    bg[y0:y0 + h, x0:x0 + w] = receipt_cv
    return bg


def make_degraded_photo(lines, angle_deg=None, blur=True, out_path="synthetic_receipt.png"):
    if angle_deg is None:
        angle_deg = random.uniform(-8, 8)

    clean = render_clean_receipt(lines)
    cv_img = cv2.cvtColor(np.array(clean), cv2.COLOR_RGB2BGR)

    placed = _place_on_cluttered_background(cv_img)

    h, w = placed.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(placed, m, (w, h), borderValue=(120, 110, 95))

    lit = _add_uneven_lighting(rotated)
    noisy = _add_gaussian_noise(lit, sigma=10)

    if blur:
        noisy = cv2.GaussianBlur(noisy, (3, 3), 0)

    cv2.imwrite(out_path, noisy)
    return out_path


SAMPLE_RECEIPTS = [
    {
        "name": "receipt_rent.png",
        "angle": 4.5,
        "lines": [
            "SHREE PROPERTIES",
            "Shop No. 4, MG Road, Kolhapur",
            "GSTIN: 27ABCDE1234F1Z5",
            "",
            "Receipt No: RCT-2026-0417",
            "Date: 03-08-2026",
            "",
            "Monthly Rent - Shop Premises",
            "Amount: Rs. 18,500.00",
            "CGST @9%: Rs. 1,665.00",
            "SGST @9%: Rs. 1,665.00",
            "Total: Rs. 21,830.00",
        ],
    },
    {
        "name": "receipt_catering.png",
        "angle": -6.0,
        "lines": [
            "ANNAPURNA CATERERS",
            "Rajarampuri, Kolhapur",
            "GSTIN: 27PQRSX5678G1Z2",
            "",
            "Invoice No: AC-8842",
            "Date: 28-07-2026",
            "",
            "Staff Lunch Catering - 25 plates",
            "Amount: Rs. 4,500.00",
            "CGST @2.5%: Rs. 112.50",
            "SGST @2.5%: Rs. 112.50",
            "Total: Rs. 4,725.00",
        ],
    },
    {
        "name": "receipt_purchase.png",
        "angle": 2.0,
        "lines": [
            "KOLHAPUR STEEL TRADERS",
            "Shivaji Udyamnagar, Kolhapur",
            "GSTIN: 27LMNOP9012H1Z8",
            "",
            "Tax Invoice No: KST-3301",
            "Date: 01-08-2026",
            "",
            "Steel Sheets - 200 units",
            "Amount: Rs. 62,000.00",
            "CGST @9%: Rs. 5,580.00",
            "SGST @9%: Rs. 5,580.00",
            "Total: Rs. 73,160.00",
        ],
    },
]


def generate_all(out_dir="sample_docs"):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for spec in SAMPLE_RECEIPTS:
        path = os.path.join(out_dir, spec["name"])
        make_degraded_photo(spec["lines"], angle_deg=spec["angle"], out_path=path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    generated = generate_all()
    print(f"Generated {len(generated)} synthetic test documents:")
    for p in generated:
        print(f"  {p}")
