from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def pick_polyline(ax, title: str):
    ax.set_title(title)
    print(f"\n{title}")
    print("- 左键添加点，回车结束当前线")
    pts = plt.ginput(n=-1, timeout=0)
    pts = [[int(round(x)), int(round(y))] for x, y in pts]
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="label json path")
    parser.add_argument("--overlay", default="", help="optional overlay image path")
    args = parser.parse_args()

    jp = Path(args.json)
    data = json.loads(jp.read_text(encoding="utf-8"))

    img_path = Path(data["image_path"])
    if args.overlay:
        show_path = Path(args.overlay)
    else:
        show_path = img_path

    if not show_path.exists():
        raise FileNotFoundError(f"image not found: {show_path}")

    img = np.array(Image.open(show_path).convert("RGB"))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("on")

    near_mid = pick_polyline(ax, "Step1: 标注 near_mid_boundary")
    if near_mid:
        xs = [p[0] for p in near_mid]
        ys = [p[1] for p in near_mid]
        ax.plot(xs, ys, "y-o", linewidth=2)
        plt.draw()

    mid_far = pick_polyline(ax, "Step2: 标注 mid_far_boundary")
    if mid_far:
        xs = [p[0] for p in mid_far]
        ys = [p[1] for p in mid_far]
        ax.plot(xs, ys, "c-o", linewidth=2)
        plt.draw()

    data.setdefault("anchors", {})
    data["anchors"]["near_mid_boundary"] = near_mid
    data["anchors"]["mid_far_boundary"] = mid_far

    jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved anchors -> {jp}")

    plt.title("Saved. close window to finish")
    plt.show()


if __name__ == "__main__":
    main()
