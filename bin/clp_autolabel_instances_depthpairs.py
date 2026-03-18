from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path("data/CLP_dataset/clp2k")
ANN_ROOT = ROOT / "annotations"
IMG_ROOT = ROOT / "images"
OUT_ROOT = ROOT / "labels_v2_auto"
INST_MASK_ROOT = ROOT / "instance_masks_auto"
LABEL_FILE = ROOT / "autosave_label.txt"

# 过滤过小连通域（按图像面积比例）
DEFAULT_MIN_AREA_RATIO = 0.0008
CLASS_MIN_AREA_RATIO = {
    "person": 0.00015,
    "birds": 0.00010,
    "animals": 0.00020,
    "boat": 0.00020,
}

# 类别ID -> 名称

def load_class_map() -> Dict[int, str]:
    text = LABEL_FILE.read_text(encoding="utf-8", errors="ignore")
    out: Dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"^(\d+)\s+([A-Za-z_]+)", line.strip())
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def sanitize_class(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def level_from_center_y(cy: float, h: int) -> str:
    y = cy / max(h, 1)
    if y >= 0.66:
        return "near"
    if y >= 0.40:
        return "mid"
    return "far"


def make_depth_score(cy: float, area: float, h: int, img_area: float) -> float:
    y_term = cy / max(h, 1)
    a_term = math.sqrt(max(area, 1.0) / max(img_area, 1.0))
    return 0.7 * y_term + 0.3 * a_term


def build_pairs(instances: List[dict], h: int, w: int) -> List[dict]:
    if len(instances) < 2:
        return []

    img_area = h * w
    scored = []
    for ins in instances:
        x1, y1, x2, y2 = ins["bbox_xyxy"]
        area = max((x2 - x1 + 1) * (y2 - y1 + 1), 1)
        cx, cy = ins["center_xy"]
        s = make_depth_score(cy, area, h, img_area)
        scored.append((ins["id"], s))

    id2score = dict(scored)

    pairs = []
    ids = [ins["id"] for ins in instances]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            diff = id2score[a] - id2score[b]
            if abs(diff) < 0.08:
                continue
            # 分数更高的更靠前
            if diff > 0:
                rel = {"a_id": a, "b_id": b, "relation": "A_front_B", "confidence": round(min(abs(diff) / 0.35, 1.0), 3)}
            else:
                rel = {"a_id": b, "b_id": a, "relation": "A_front_B", "confidence": round(min(abs(diff) / 0.35, 1.0), 3)}
            pairs.append((abs(diff), rel))

    # 限制数量，优先保留差异大的
    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = [p[1] for p in pairs[:60]]
    return pairs


def process_one(split: str, ann_path: Path, class_map: Dict[int, str]) -> None:
    stem = ann_path.stem
    img_path = IMG_ROOT / split / f"{stem}.jpg"
    if not img_path.exists():
        return

    mask = np.array(Image.open(ann_path).convert("L"))
    h, w = mask.shape
    img_area = h * w

    instances = []
    inst_id = 1

    # 为该图创建实例mask目录
    out_inst_dir = INST_MASK_ROOT / split
    out_inst_dir.mkdir(parents=True, exist_ok=True)

    for cls_id in sorted(np.unique(mask).tolist()):
        if cls_id == 0:
            continue
        cls_name = sanitize_class(class_map.get(cls_id, f"class_{cls_id}"))

        # 可选：先跳过 sky（大面积背景，不利于实例化）
        if cls_name == "sky":
            continue

        bin_mask = (mask == cls_id).astype(np.uint8)
        n, labels, stats, cent = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        min_area = int(CLASS_MIN_AREA_RATIO.get(cls_name, DEFAULT_MIN_AREA_RATIO) * img_area)

        for k in range(1, n):
            x, y, ww, hh, area = stats[k]
            if area < max(min_area, 8):
                continue

            x1, y1, x2, y2 = int(x), int(y), int(x + ww - 1), int(y + hh - 1)
            cx, cy = float(cent[k][0]), float(cent[k][1])
            san_level = level_from_center_y(cy, h)

            inst_mask = (labels == k).astype(np.uint8) * 255
            inst_mask_name = f"{stem}_inst_{inst_id}.png"
            inst_mask_path = out_inst_dir / inst_mask_name
            Image.fromarray(inst_mask).save(inst_mask_path)

            instances.append(
                {
                    "id": inst_id,
                    "class": cls_name,
                    "mask_path": str(inst_mask_path).replace("\\", "/"),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_xy": [round(cx, 2), round(cy, 2)],
                    "san_yuan_level": san_level,
                }
            )
            inst_id += 1

    depth_pairs = build_pairs(instances, h, w)

    out = {
        "image_id": stem,
        "image_path": str(img_path).replace("\\", "/"),
        "width": int(w),
        "height": int(h),
        "instances": instances,
        "depth_pairs": depth_pairs,
        "anchors": {"near_mid_boundary": [], "mid_far_boundary": []},
    }

    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    class_map = load_class_map()
    total = 0
    for split in ["training", "validation"]:
        ann_dir = ANN_ROOT / split
        for ann_path in ann_dir.glob("*.png"):
            process_one(split, ann_path, class_map)
            total += 1
    print(f"done. auto labels generated for {total} masks")
    print("labels:", OUT_ROOT)
    print("instance masks:", INST_MASK_ROOT)


if __name__ == "__main__":
    main()
