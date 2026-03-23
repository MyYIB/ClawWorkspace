from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/CLP_dataset/clp2k")
OVERLAY_ROOT = ROOT / "annotations_overlay"
OUT_OVERLAY = ROOT / "annotations_overlay_id"
LABEL_JSON_ROOT = ROOT / "labels_v2_auto"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS = 1
MIN_INSTANCE_AREA = 120


def put_text_with_outline(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    x, y = org
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, (0, 0, 0), THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)


def pick_text_anchor(instance: dict, img_shape: tuple[int, int, int]) -> tuple[int, int] | None:
    h, w = img_shape[:2]

    # 优先使用 instance mask 真值中心，保证和 labels_v2_auto 的 id 一一对应
    mask_path = instance.get("mask_path")
    if isinstance(mask_path, str):
        mp = Path(mask_path)
        if not mp.is_absolute():
            mp = Path(mp.as_posix())
        if mp.exists():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                ys, xs = np.where(m > 0)
                if xs.size >= MIN_INSTANCE_AREA:
                    cx = int(xs.mean())
                    cy = int(ys.mean())
                    tx = max(4, min(cx - 10, w - 180))
                    ty = max(16, min(cy + 5, h - 6))
                    return tx, ty

    center = instance.get("center_xy")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        cx, cy = int(center[0]), int(center[1])
        tx = max(4, min(cx - 10, w - 180))
        ty = max(16, min(cy + 5, h - 6))
        return tx, ty

    return None


def draw_instance_ids(overlay: np.ndarray, instances: list[dict]) -> np.ndarray:
    out = overlay.copy()

    for ins in instances:
        ins_id = ins.get("id")
        ins_class = ins.get("class", "unknown")
        if ins_id is None:
            continue

        anchor = pick_text_anchor(ins, out.shape)
        if anchor is None:
            continue

        text = f"{ins_id}:{ins_class}"
        put_text_with_outline(out, text, anchor)

    return out


def main() -> None:
    total = 0

    for split in ["training", "validation"]:
        (OUT_OVERLAY / split).mkdir(parents=True, exist_ok=True)

        json_files = sorted((LABEL_JSON_ROOT / split).glob("*.json"))
        for jp in json_files:
            data = json.loads(jp.read_text(encoding="utf-8"))
            image_id = data.get("image_id") or jp.stem
            instances = data.get("instances", [])

            op = OVERLAY_ROOT / split / f"{image_id}.jpg"
            if not op.exists():
                continue

            overlay = cv2.imread(str(op), cv2.IMREAD_COLOR)
            if overlay is None:
                continue

            overlay_id = draw_instance_ids(overlay, instances)
            cv2.imwrite(str(OUT_OVERLAY / split / f"{image_id}.jpg"), overlay_id, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            total += 1

    print(f"done instance-id overlay render for {total} samples")
    print("out overlay:", OUT_OVERLAY)


if __name__ == "__main__":
    main()
