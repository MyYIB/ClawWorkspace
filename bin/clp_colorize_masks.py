from __future__ import annotations

import re
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("data/CLP_dataset/clp2k")
MASK_ROOT = ROOT / "annotations"
IMG_ROOT = ROOT / "images"
COLOR_MASK_ROOT = ROOT / "annotations_color"
OVERLAY_ROOT = ROOT / "annotations_overlay"
LABEL_FILE = ROOT / "autosave_label.txt"


def load_palette() -> dict[int, tuple[int, int, int]]:
    # id 0 background
    palette: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    text = LABEL_FILE.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s+([A-Za-z_]+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            idx = int(m.group(1))
            palette[idx] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return palette


def colorize(mask: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in palette.items():
        out[mask == idx] = rgb
    return out


def blend(img: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(img * (1 - alpha) + color_mask * alpha, 0, 255).astype(np.uint8)


def main() -> None:
    palette = load_palette()
    total = 0
    for split in ["training", "validation"]:
        (COLOR_MASK_ROOT / split).mkdir(parents=True, exist_ok=True)
        (OVERLAY_ROOT / split).mkdir(parents=True, exist_ok=True)
        for mp in (MASK_ROOT / split).glob("*.png"):
            stem = mp.stem
            ip = IMG_ROOT / split / f"{stem}.jpg"
            if not ip.exists():
                continue

            mask = np.array(Image.open(mp).convert("L"))
            img = np.array(Image.open(ip).convert("RGB"))
            c = colorize(mask, palette)
            o = blend(img, c, alpha=0.45)

            Image.fromarray(c).save(COLOR_MASK_ROOT / split / f"{stem}.png")
            Image.fromarray(o).save(OVERLAY_ROOT / split / f"{stem}.jpg", quality=95)
            total += 1

    print(f"done colorized+overlay for {total} samples")
    print("color masks:", COLOR_MASK_ROOT)
    print("overlay:", OVERLAY_ROOT)


if __name__ == "__main__":
    main()
