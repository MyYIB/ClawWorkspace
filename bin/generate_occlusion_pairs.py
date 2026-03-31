from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


LEVEL_ORDER = {"near": 0, "mid": 1, "far": 2}


def parse_depth_rel(rel: str):
    r = (rel or "").upper()
    if "A_FRONT_B" in r:
        return "A_FRONT_B"
    if "A_BEHIND_B" in r or "A_BACK_B" in r:
        return "A_BEHIND_B"
    return None


def load_mask(path: str):
    p = Path(path)
    if not p.exists():
        return None
    m = Image.open(p).convert("L")
    arr = np.array(m)
    return arr > 127


def infer_front(a: dict, b: dict, depth_map: dict):
    key = (int(a.get("id", -1)), int(b.get("id", -1)))
    rev = (key[1], key[0])
    if key in depth_map:
        rel = depth_map[key]
        if rel == "A_FRONT_B":
            return "A"
        if rel == "A_BEHIND_B":
            return "B"
    if rev in depth_map:
        rel = depth_map[rev]
        if rel == "A_FRONT_B":
            return "B"
        if rel == "A_BEHIND_B":
            return "A"

    la = LEVEL_ORDER.get(str(a.get("san_yuan_level", "")).lower(), 1)
    lb = LEVEL_ORDER.get(str(b.get("san_yuan_level", "")).lower(), 1)
    if la < lb:
        return "A"
    if lb < la:
        return "B"
    return None


def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    ba = max(0.0, (bx2 - bx1) * (by2 - by1))
    den = aa + ba - inter
    return 0.0 if den <= 0 else inter / den


def get_bbox(obj):
    b = obj.get("bbox_xyxy")
    if isinstance(b, list) and len(b) >= 4:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    c = obj.get("center_xy")
    if isinstance(c, list) and len(c) >= 2:
        x, y = float(c[0]), float(c[1])
        return [x - 32, y - 32, x + 32, y + 32]
    return [0.0, 0.0, 1.0, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source label root, e.g. data/.../labels_v2_auto")
    ap.add_argument("--dst", required=True, help="dest label root")
    ap.add_argument("--overlap-thr", type=float, default=0.01, help="min overlap ratio wrt smaller mask")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    files = sorted(src.rglob("*.json"))

    total = 0
    touched = 0
    for fp in files:
        total += 1
        data = json.loads(fp.read_text(encoding="utf-8"))
        inst = data.get("instances", [])
        depth_pairs = data.get("depth_pairs", [])

        depth_map = {}
        for dp in depth_pairs:
            try:
                a_id = int(dp.get("a_id", -1))
                b_id = int(dp.get("b_id", -1))
            except Exception:
                continue
            rel = parse_depth_rel(str(dp.get("relation", "")))
            if rel:
                depth_map[(a_id, b_id)] = rel

        masks = {}
        for obj in inst:
            mid = int(obj.get("id", -1))
            mp = obj.get("mask_path")
            if isinstance(mp, str):
                masks[mid] = load_mask(mp)
            else:
                masks[mid] = None

        occ_pairs = []
        n = len(inst)
        for i in range(n):
            for j in range(i + 1, n):
                a = inst[i]
                b = inst[j]
                a_id = int(a.get("id", -1))
                b_id = int(b.get("id", -1))
                ma = masks.get(a_id)
                mb = masks.get(b_id)
                if ma is None or mb is None:
                    continue

                inter = np.logical_and(ma, mb).sum()
                denom = min(ma.sum(), mb.sum()) if (ma is not None and mb is not None) else 0
                overlap_ratio = float(inter / denom) if denom > 0 else 0.0

                ba = get_bbox(a)
                bb = get_bbox(b)
                iou = bbox_iou_xyxy(ba, bb)

                ca = a.get("center_xy", [0.0, 0.0])
                cb = b.get("center_xy", [0.0, 0.0])
                if len(ca) < 2:
                    ca = [(ba[0] + ba[2]) * 0.5, (ba[1] + ba[3]) * 0.5]
                if len(cb) < 2:
                    cb = [(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5]
                dist = float(np.hypot(float(ca[0]) - float(cb[0]), float(ca[1]) - float(cb[1])))

                has_contact = (overlap_ratio >= args.overlap_thr) or (iou >= 0.01) or (dist < 80)
                if not has_contact:
                    continue

                front = infer_front(a, b, depth_map)
                conf = min(1.0, 0.45 + 0.4 * max(overlap_ratio, iou) + (0.15 if dist < 80 else 0.0))
                if front == "A":
                    occ_pairs.append({
                        "a_id": a_id,
                        "b_id": b_id,
                        "relation": "A_occludes_B",
                        "confidence": round(conf, 3),
                        "source": "heuristic(depth_pairs+mask/bbox proximity)"
                    })
                elif front == "B":
                    occ_pairs.append({
                        "a_id": b_id,
                        "b_id": a_id,
                        "relation": "A_occludes_B",
                        "confidence": round(conf, 3),
                        "source": "heuristic(depth_pairs+mask/bbox proximity)"
                    })

        data["occlusion_pairs"] = occ_pairs
        if occ_pairs:
            touched += 1

        out = dst / fp.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done: total={total}, with_occlusion_pairs={touched}, dst={dst}")


if __name__ == "__main__":
    main()
