from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import PlanADataset
from src.engine.trainer import train_one_epoch
from src.models.layoutnet import Object3DLayoutNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PlanADataset(
        label_root=cfg["data"]["label_root"],
        image_size=tuple(cfg["data"]["image_size"]),
        max_objects=int(cfg["data"].get("max_objects", 32)),
        min_objects=int(cfg["data"].get("min_objects", 1)),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = Object3DLayoutNet(
        obj_feat_dim=int(cfg["model"].get("obj_feat_dim", 16)),
        d_model=int(cfg["model"].get("d_model", 256)),
        num_layers=int(cfg["model"].get("num_layers", 4)),
        num_heads=int(cfg["model"].get("num_heads", 8)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        metric = train_one_epoch(
            model=model,
            loader=dl,
            optimizer=optim,
            device=device,
            loss_cfg=cfg["loss"],
            grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
        )
        print(
            f"epoch={epoch} total={metric['total']:.5f} proj={metric['proj']:.5f} "
            f"depth={metric['depth_rank']:.5f} occ={metric['occ']:.5f}"
        )

        if epoch % int(cfg["output"].get("save_every", 1)) == 0:
            ckpt = out_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, ckpt)


if __name__ == "__main__":
    main()
