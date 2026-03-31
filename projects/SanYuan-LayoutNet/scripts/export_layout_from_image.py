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


# 说明：当前模型是“对象级”输入，因此除了山水画图像，还需要对象实例信息（instances）。
# 最简输入：
# {
#   "image_id": "xxx",
#   "image_path": "...jpg",
#   "instances": [
#     {"id":1, "class":"mountain", "bbox_xyxy":[x1,y1,x2,y2], "center_xy":[x,y], "san_yuan_level":"near"}
#   ],
#   "depth_pairs": []
# }


def map_to_scene(center3d, size3d):
    # 可按你Unity场景再调
    x = float((center3d[0] - 0.5) * 10.0)   # [-5,5]
    y = float((1.0 - center3d[1]) * 5.0)    # [0,5]
    z = float(center3d[2] * 10.0)           # [0,10], 越大越远

    sx = float(max(0.1, size3d[0] * 4.0))
    sy = float(max(0.1, size3d[1] * 4.0))
    sz = float(max(0.1, size3d[2] * 4.0))
    return [x, y, z], [sx, sy, sz]


def build_model(ckpt_path: str, device: str):
    model = Object3DLayoutNet(obj_feat_dim=16, d_model=128, num_layers=2, num_heads=4, dropout=0.1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def load_input(input_json: Path, image_size=(512, 512), max_objects=24):
    data = json.loads(input_json.read_text(encoding="utf-8"))
    image_path = data["image_path"]

    img = Image.open(image_path).convert("RGB").resize(image_size)
    arr = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(arr).permute(2, 0, 1)

    ds = PlanADataset(label_root=".", image_size=image_size, max_objects=max_objects)
    instances = ds._extract_objects(data)

    obj_feat = np.zeros((max_objects, 16), dtype=np.float32)
    obj_valid = np.zeros((max_objects,), dtype=np.float32)
    meta = []

    n = min(len(instances), max_objects)
    for i in range(n):
        feat, _, _, _, obj_id = ds._object_to_features_and_target(instances[i])
        obj_feat[i] = feat
        obj_valid[i] = 1.0
        meta.append({
            "id": int(obj_id),
            "class": str(instances[i].get("class", "unknown")),
        })

    return data, img_t, torch.from_numpy(obj_feat), torch.from_numpy(obj_valid), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True, help="包含 image_path + instances 的json")
    ap.add_argument("--ckpt", required=True, help="模型权重 .pt")
    ap.add_argument("--output-json", required=True, help="输出摆放json")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--max-objects", type=int, default=24)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.ckpt, device)

    data, img_t, obj_feat, obj_valid, meta = load_input(
        Path(args.input_json),
        image_size=(args.image_size, args.image_size),
        max_objects=args.max_objects,
    )

    with torch.no_grad():
        pred = model(
            img_t.unsqueeze(0).to(device),
            obj_feat.unsqueeze(0).to(device),
            obj_valid.unsqueeze(0).to(device),
        )

    n = len(meta)
    center3d = pred["center3d"][0, :n].cpu().numpy()
    size3d = pred["size3d"][0, :n].cpu().numpy()
    order_score = torch.sigmoid(pred["order_logit"][0, :n]).cpu().numpy()

    rank = np.argsort(order_score)[::-1]  # far -> near
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
        "image_id": data.get("image_id", Path(args.input_json).stem),
        "image_path": data.get("image_path", ""),
        "ckpt": args.ckpt,
        "coordinate_note": {
            "scene_position_xyz": "x:[-5,5], y:[0,5], z:[0,10], z越大越远",
            "scene_scale_xyz": "相对尺度，可按Unity场景缩放"
        },
        "objects": objects,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
