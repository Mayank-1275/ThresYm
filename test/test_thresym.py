"""
ThresYm - Automated Test & Sample Runner
=========================================
Ye script Test.png par saare 6 thresholding algorithms automatically
run karta hai, result save karta hai aur analytics print karta hai.

Usage:
    python test_thresym.py
    python test_thresym.py --image path/to/your_image.png

Output:
    thresym_results/  (folder mein saare processed images + comparison grid)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io

# ─── Optional: scikit-image ────────────────────────────────────────────────
try:
    from skimage.filters import threshold_niblack, threshold_sauvola
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
    print("[WARN] scikit-image not found → Niblack & Sauvola will be skipped.")
    print("       Install it: pip install scikit-image\n")

# ─── Default test-image path ───────────────────────────────────────────────
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "Test.png")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "thresym_results")

# ─── Default parameters (same as app.py) ──────────────────────────────────
PARAMS = {
    "Simple Threshold":  {"threshold": 127},
    "Otsu's Method":     {},
    "Adaptive Mean":     {"block_size": 11, "C": 2},
    "Adaptive Gaussian": {"block_size": 11, "C": 2},
    "Niblack":           {"window_size": 15, "k": -0.2},
    "Sauvola":           {"window_size": 15, "k": 0.2, "R": 128},
}

COLORS = {
    "Simple Threshold":  (0,   229, 255),   # cyan
    "Otsu's Method":     (57,  255, 138),   # green
    "Adaptive Mean":     (255, 193,   7),   # amber
    "Adaptive Gaussian": (182, 109, 255),   # purple
    "Niblack":           (255,  77, 109),   # red
    "Sauvola":           (255, 165,   0),   # orange
}


# ══════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSOR  (mirrors app.py logic — Streamlit dependency removed)
# ══════════════════════════════════════════════════════════════════════════
class ImageProcessor:
    def __init__(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        self.original  = arr
        self.gray      = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        self.h, self.w = self.gray.shape

    # ── algorithms ────────────────────────────────────────────────────────
    def simple_threshold(self, threshold=127):
        _, binary = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def otsu_threshold(self):
        _, binary = cv2.threshold(self.gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def adaptive_threshold(self, block_size=11, C=2, method="mean"):
        m = (cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean"
             else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        return cv2.adaptiveThreshold(self.gray, 255, m,
                                     cv2.THRESH_BINARY, block_size, C)

    def niblack_threshold(self, window_size=15, k=-0.2):
        if not SKIMAGE_OK:
            return None
        thresh = threshold_niblack(self.gray, window_size=window_size, k=k)
        return (self.gray > thresh).astype(np.uint8) * 255

    def sauvola_threshold(self, window_size=15, k=0.2, R=128):
        if not SKIMAGE_OK:
            return None
        thresh = threshold_sauvola(self.gray, window_size=window_size,
                                   k=k, r=R)
        return (self.gray > thresh).astype(np.uint8) * 255

    # ── run all ───────────────────────────────────────────────────────────
    def run_all(self):
        p = PARAMS
        return {
            "Simple Threshold":  self.simple_threshold(**p["Simple Threshold"]),
            "Otsu's Method":     self.otsu_threshold(),
            "Adaptive Mean":     self.adaptive_threshold(**p["Adaptive Mean"],     method="mean"),
            "Adaptive Gaussian": self.adaptive_threshold(**p["Adaptive Gaussian"], method="gaussian"),
            "Niblack":           self.niblack_threshold(**p["Niblack"]),
            "Sauvola":           self.sauvola_threshold(**p["Sauvola"]),
        }


# ══════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
def compute_stats(binary: np.ndarray) -> dict:
    total  = binary.size
    white  = int(np.sum(binary == 255))
    black  = total - white
    return {
        "total":      total,
        "white_px":   white,
        "black_px":   black,
        "white_pct":  round(white / total * 100, 2),
        "black_pct":  round(black / total * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════
#  COMPARISON GRID  (dark cyberpunk style — pure PIL, no Streamlit needed)
# ══════════════════════════════════════════════════════════════════════════
def make_comparison_grid(gray: np.ndarray,
                         results: dict,
                         image_name: str) -> Image.Image:
    CELL_W, CELL_H = 320, 320
    PAD            = 20
    HEADER_H       = 70
    FOOTER_H       = 55
    BAR_H          = 8
    INFO_H         = 45

    valid = {k: v for k, v in results.items() if v is not None}
    n_methods = len(valid)
    cols      = n_methods + 1          # +1 for original
    grid_w    = cols * CELL_W + (cols + 1) * PAD
    grid_h    = HEADER_H + PAD + CELL_H + INFO_H + BAR_H + FOOTER_H + PAD * 3

    # ── canvas ────────────────────────────────────────────────────────────
    canvas = Image.new("RGB", (grid_w, grid_h), (10, 12, 16))
    draw   = ImageDraw.Draw(canvas)

    # ── title bar ────────────────────────────────────────────────────────
    draw.rectangle([0, 0, grid_w, HEADER_H], fill=(15, 17, 23))
    draw.line([(0, HEADER_H), (grid_w, HEADER_H)], fill=(30, 37, 53), width=1)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sub   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  9)
    except Exception:
        font_title = font_sub = font_label = font_small = ImageFont.load_default()

    draw.text((PAD + 6, 14), "ThresYm",  fill=(0, 229, 255),  font=font_title)
    draw.text((PAD + 6, 42), f"Advanced Image Thresholding Workbench  ·  {image_name}",
              fill=(74, 85, 104), font=font_sub)

    badge_x = grid_w - 10
    for label, color in [("PNG EXPORT", (0, 229, 255)),
                          ("REAL-TIME PARAMS", (182, 109, 255)),
                          ("6 METHODS", (57, 255, 138))]:
        tw = draw.textlength(label, font=font_small)
        bx = int(badge_x - tw - 16)
        draw.rounded_rectangle([bx, 20, int(badge_x), 38],
                                radius=3, fill=(20, 24, 32),
                                outline=(*color, 80))
        draw.text((bx + 8, 23), label, fill=color, font=font_small)
        badge_x = bx - 8

    # ── helper: place one cell ─────────────────────────────────────────────
    def place_cell(col_idx, image_arr, label, color, stats=None, badge=""):
        cx = PAD + col_idx * (CELL_W + PAD)
        cy = HEADER_H + PAD * 2

        # card background
        draw.rounded_rectangle([cx, cy, cx + CELL_W, cy + CELL_H + INFO_H + BAR_H + PAD],
                                radius=8, fill=(19, 23, 32),
                                outline=(30, 37, 53))
        draw.line([(cx, cy), (cx + CELL_W, cy)], fill=color, width=2)

        # thumbnail
        thumb = Image.fromarray(image_arr).resize(
            (CELL_W - 16, CELL_H - 16), Image.LANCZOS)
        canvas.paste(thumb, (cx + 8, cy + 8))

        # label row
        ly = cy + CELL_H + 6
        draw.text((cx + 8, ly), label, fill=(232, 237, 245), font=font_label)
        if badge:
            bw = int(draw.textlength(badge, font=font_small))
            bx2 = cx + CELL_W - bw - 16
            draw.rounded_rectangle([bx2, ly, bx2 + bw + 10, ly + 15],
                                    radius=3,
                                    fill=(*color, 20),
                                    outline=(*color, 60))
            draw.text((bx2 + 5, ly + 2), badge, fill=color, font=font_small)

        # pixel bar
        if stats:
            by = ly + INFO_H - 6
            bar_full = CELL_W - 16
            white_w  = int(bar_full * stats["white_pct"] / 100)

            draw.rounded_rectangle([cx + 8, by, cx + 8 + bar_full, by + BAR_H],
                                    radius=3, fill=(10, 12, 16))
            if white_w > 0:
                draw.rounded_rectangle([cx + 8, by, cx + 8 + white_w, by + BAR_H],
                                        radius=3, fill=(0, 229, 255))

            draw.text((cx + 8,              by + BAR_H + 5),
                      f"W {stats['white_pct']}%", fill=(0, 229, 255), font=font_small)
            draw.text((cx + CELL_W - 60,    by + BAR_H + 5),
                      f"B {stats['black_pct']}%", fill=(182, 109, 255), font=font_small)

    # ── original grayscale ────────────────────────────────────────────────
    place_cell(0, gray, "SOURCE IMAGE", (74, 85, 104), badge="GRAYSCALE")

    # ── thresholded results ───────────────────────────────────────────────
    BADGES = {
        "Simple Threshold":  "GLOBAL",
        "Otsu's Method":     "AUTO",
        "Adaptive Mean":     "LOCAL",
        "Adaptive Gaussian": "LOCAL",
        "Niblack":           "ADVANCED",
        "Sauvola":           "ADVANCED",
    }

    for i, (method, binary) in enumerate(valid.items(), start=1):
        color = COLORS[method]
        stats = compute_stats(binary)
        place_cell(i, binary, method, color, stats, BADGES.get(method, ""))

    # ── footer ────────────────────────────────────────────────────────────
    fy = grid_h - FOOTER_H + 10
    draw.line([(0, fy - 4), (grid_w, fy - 4)], fill=(30, 37, 53), width=1)
    draw.text((PAD + 6, fy + 8),
              "ThresYm v2.0  ·  opencv · scikit-image · numpy",
              fill=(42, 53, 72), font=font_small)
    draw.text((grid_w - 200, fy + 8),
              "Computer Vision Lab",
              fill=(42, 53, 72), font=font_small)

    return canvas


# ══════════════════════════════════════════════════════════════════════════
#  SAVE INDIVIDUAL PNGs
# ══════════════════════════════════════════════════════════════════════════
def save_individual(results: dict, output_dir: str):
    saved = []
    for method, binary in results.items():
        if binary is None:
            continue
        safe = method.lower().replace(" ", "_").replace("'", "")
        path = os.path.join(output_dir, f"thresym_{safe}.png")
        Image.fromarray(binary).save(path)
        saved.append((method, path))
    return saved


# ══════════════════════════════════════════════════════════════════════════
#  PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════
def print_report(gray, results, image_path, saved_files, grid_path):
    sep = "─" * 62

    print(f"\n{'═'*62}")
    print(f"  🔬  ThresYm  ·  Automated Test Report")
    print(f"{'═'*62}")
    print(f"  Image  : {os.path.abspath(image_path)}")
    print(f"  Size   : {gray.shape[1]} x {gray.shape[0]} px")
    print(f"  Pixels : {gray.size:,}")
    print(sep)

    print(f"\n  {'METHOD':<22}  {'WHITE %':>8}  {'BLACK %':>8}  {'STATUS'}")
    print(f"  {sep}")

    for method, binary in results.items():
        if binary is None:
            print(f"  {method:<22}  {'—':>8}  {'—':>8}  ⚠  SKIPPED (scikit-image missing)")
        else:
            s = compute_stats(binary)
            bar_w   = 20
            filled  = int(bar_w * s["white_pct"] / 100)
            bar     = "█" * filled + "░" * (bar_w - filled)
            print(f"  {method:<22}  {s['white_pct']:>7.1f}%  {s['black_pct']:>7.1f}%  ✓  [{bar}]")

    print(f"\n{sep}")
    print(f"  📁  Output files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print()
    for method, path in saved_files:
        print(f"     ✓  {os.path.basename(path):<40}  ({method})")
    print(f"     ✓  {os.path.basename(grid_path):<40}  (Comparison Grid)")
    print(f"{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ThresYm — Automated Image Thresholding Test Runner"
    )
    parser.add_argument(
        "--image", "-i",
        default=DEFAULT_IMAGE,
        help="Path to input image (default: Test.png)"
    )
    args = parser.parse_args()

    image_path = args.image

    # ── Validate input ────────────────────────────────────────────────────
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image not found: {image_path}")
        print("        Put Test.png in the same folder as this script, or")
        print("        pass a custom path:  python test_thresym.py --image your_image.jpg\n")
        sys.exit(1)

    print(f"\n  🔬  ThresYm Test Runner starting …")
    print(f"      Input : {image_path}")

    # ── Output folder ─────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Process ───────────────────────────────────────────────────────────
    print("  ⚙   Loading image …")
    proc    = ImageProcessor(image_path)

    print("  ⚙   Running all 6 thresholding algorithms …")
    results = proc.run_all()

    # ── Save individual PNGs ──────────────────────────────────────────────
    print("  💾  Saving individual results …")
    saved = save_individual(results, OUTPUT_DIR)

    # ── Save grayscale source ─────────────────────────────────────────────
    gray_path = os.path.join(OUTPUT_DIR, "thresym_source_gray.png")
    Image.fromarray(proc.gray).save(gray_path)

    # ── Build comparison grid ─────────────────────────────────────────────
    print("  🖼   Building comparison grid …")
    grid       = make_comparison_grid(proc.gray, results,
                                      os.path.basename(image_path))
    grid_path  = os.path.join(OUTPUT_DIR, "thresym_comparison_grid.png")
    grid.save(grid_path, dpi=(150, 150))

    # ── Print report ──────────────────────────────────────────────────────
    print_report(proc.gray, results, image_path, saved, grid_path)

    print("  ✅  Done!  Open thresym_results/ to see all outputs.\n")


if __name__ == "__main__":
    main()
