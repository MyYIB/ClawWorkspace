from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import sys
sys.path.append("projects/SanYuan-LayoutNet")

from src.models.layoutnet import Object3DLayoutNet
from src.data.dataset import PlanADataset


def load_sample(json_path: Path, image_size=(512, 512), max_objects=24):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = data["image_path"]
    img = Image.open(image_path).convert("RGB").resize(image_size)
    arr = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(arr).permute(2, 0, 1)

    ds = PlanADataset(label_root=str(json_path.parent), image_size=image_size, max_objects=max_objects)
    objs = ds._extract_objects(data)

    obj_feat = np.zeros((max_objects, 16), dtype=np.float32)
    obj_valid = np.zeros((max_objects,), dtype=np.float32)
    obj_ids = []
    centers_2d = []

    n = min(len(objs), max_objects)
    for i in range(n):
        feat, center3d, _, _, obj_id = ds._object_to_features_and_target(objs[i])
        obj_feat[i] = feat
        obj_valid[i] = 1.0
        obj_ids.append(obj_id)
        centers_2d.append((center3d[0] * image_size[0], center3d[1] * image_size[1]))

    return data, img, img_t, torch.from_numpy(obj_feat), torch.from_numpy(obj_valid), obj_ids, centers_2d


def draw_prediction(img: Image.Image, obj_ids, centers_2d, pred_center3d, pred_order, out_path: Path):
    draw = ImageDraw.Draw(img)
    # sort far->near by predicted z
    order_idx = np.argsort(pred_center3d[:, 2])[::-1]
    rank = {int(idx): r + 1 for r, idx in enumerate(order_idx)}

    for i, oid in enumerate(obj_ids):
        x, y = centers_2d[i]
        z = float(pred_center3d[i, 2])
        o = float(pred_order[i])
        color = (255, int(255 * (1 - z)), 0)
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        draw.text((x + 8, y - 8), f"id{oid} z={z:.2f} rk={rank[i]}", fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def run_one_ckpt(ckpt_path: Path, sample_jsons: list[Path], out_dir: Path, image_size=(512, 512), max_objects=24):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Object3DLayoutNet(obj_feat_dim=16, d_model=128, num_layers=2, num_heads=4, dropout=0.1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    summary = []
    with torch.no_grad():
        for jp in sample_jsons:
            data, img, img_t, obj_feat, obj_valid, obj_ids, centers_2d = load_sample(jp, image_size=image_size, max_objects=max_objects)
            if len(obj_ids) == 0:
                continue
            pred = model(
                img_t.unsqueeze(0).to(device),
                obj_feat.unsqueeze(0).to(device),
                obj_valid.unsqueeze(0).to(device),
            )
            n = len(obj_ids)
            center3d = pred["center3d"][0, :n].cpu().numpy()
            order = torch.sigmoid(pred["order_logit"][0, :n]).cpu().numpy()

            rel = jp.relative_to(Path("data/CLP_dataset/clp2k/labels_v2_auto"))
            stem = rel.with_suffix("").as_posix().replace("/", "__")
            out_img = out_dir / ckpt_path.parent.name / f"{ckpt_path.stem}__{stem}.png"
            draw_prediction(img.copy(), obj_ids, centers_2d, center3d, order, out_img)

            summary.append({
                "ckpt": str(ckpt_path),
                "sample": str(jp),
                "num_objects": n,
                "pred_z_mean": float(center3d[:, 2].mean()),
                "pred_z_std": float(center3d[:, 2].std()),
            })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/CLP_dataset/clp2k/labels_v2_auto")
    ap.add_argument("--out", default="outputs/infer_compare")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--ckpts", nargs="+", required=True)
    args = ap.parse_args()

    label_root = Path(args.labels)
    sample_jsons = sorted(label_root.rglob("*.json"))[: args.num_samples]
    out_dir = Path(args.out)

    all_summary = []
    for c in args.ckpts:
        all_summary.extend(run_one_ckpt(Path(c), sample_jsons, out_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done, outputs in: {out_dir}")


if __name__ == "__main__":
    main()
