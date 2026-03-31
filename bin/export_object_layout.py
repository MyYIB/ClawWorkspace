from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.append("projects/SanYuan-LayoutNet")

from src.models.layoutnet import Object3DLayoutNet
from src.data.dataset import PlanADataset


def map_to_scene(center3d, size3d):
    # 将网络[0,1]输出映射到一个更直观的场景坐标
    # x: [-5, 5], y: [0, 5] (高度), z: [0, 10] (纵深)
    x = float((center3d[0] - 0.5) * 10.0)
    y = float((1.0 - center3d[1]) * 5.0)
    z = float(center3d[2] * 10.0)

    sx = float(max(0.1, size3d[0] * 4.0))
    sy = float(max(0.1, size3d[1] * 4.0))
    sz = float(max(0.1, size3d[2] * 4.0))
    return [x, y, z], [sx, sy, sz]


def load_one(data: dict, image_size=(512, 512), max_objects=24):
    img = Image.open(data["image_path"]).convert("RGB").resize(image_size)
    arr = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(arr).permute(2, 0, 1)

    ds = PlanADataset(label_root=".", image_size=image_size, max_objects=max_objects)
    objs = ds._extract_objects(data)

    obj_feat = np.zeros((max_objects, 16), dtype=np.float32)
    obj_valid = np.zeros((max_objects,), dtype=np.float32)
    meta = []

    n = min(len(objs), max_objects)
    for i in range(n):
        feat, _, _, _, obj_id = ds._object_to_features_and_target(objs[i])
        obj_feat[i] = feat
        obj_valid[i] = 1.0
        meta.append({
            "id": int(obj_id),
            "class": str(objs[i].get("class", "unknown")),
        })

    return img_t, torch.from_numpy(obj_feat), torch.from_numpy(obj_valid), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", default="data/CLP_dataset/clp2k/labels_v2_auto")
    ap.add_argument("--out", default="outputs/layout_exports")
    ap.add_argument("--num-samples", type=int, default=5)
    args = ap.parse_args()

    label_root = Path(args.labels)
    files = sorted(label_root.rglob("*.json"))[: args.num_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Object3DLayoutNet(obj_feat_dim=16, d_model=128, num_layers=2, num_heads=4, dropout=0.1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for fp in files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            img_t, obj_feat, obj_valid, meta = load_one(data)
            pred = model(
                img_t.unsqueeze(0).to(device),
                obj_feat.unsqueeze(0).to(device),
                obj_valid.unsqueeze(0).to(device),
            )

            n = len(meta)
            center3d = pred["center3d"][0, :n].cpu().numpy()
            size3d = pred["size3d"][0, :n].cpu().numpy()
            order_score = torch.sigmoid(pred["order_logit"][0, :n]).cpu().numpy()

            rank = np.argsort(order_score)[::-1]
            rank_map = {int(idx): int(r + 1) for r, idx in enumerate(rank)}

            objects = []
            for i in range(n):
                pos, scl = map_to_scene(center3d[i], size3d[i])
                objects.append({
                    "id": meta[i]["id"],
                    "class": meta[i]["class"],
                    "pred_center_01": [float(v) for v in center3d[i]],
                    "pred_size_01": [float(v) for v in size3d[i]],
                    "scene_position_xyz": pos,
                    "scene_scale_xyz": scl,
                    "depth_order_score": float(order_score[i]),
                    "depth_rank_far_to_near": rank_map[i],
                })

            out = {
                "image_id": data.get("image_id", fp.stem),
                "image_path": data.get("image_path", ""),
                "ckpt": args.ckpt,
                "coordinate_note": {
                    "scene_position_xyz": "x:[-5,5], y:[0,5], z:[0,10], z越大越远",
                    "scene_scale_xyz": "相对尺度，可按你的3D资产进一步归一化"
                },
                "objects": objects,
            }

            rel = fp.relative_to(label_root)
            out_fp = out_root / rel
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            out_fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done, exported {len(files)} files to {out_root}")


if __name__ == "__main__":
    main()
