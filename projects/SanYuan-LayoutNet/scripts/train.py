from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import PlanADataset
from src.engine.trainer import train_one_epoch
from src.models.layoutnet import PolylineLayoutNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PlanADataset(
        label_root=cfg["data"]["label_root"],
        image_size=tuple(cfg["data"]["image_size"]),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )

    model = PolylineLayoutNet(num_curve_points=cfg["model"]["num_samples_per_curve"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        metric = train_one_epoch(model, dl, optim, device, cfg["loss"])
        print(f"epoch={epoch} train_total={metric['total']:.6f}")

        ckpt = out_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)


if __name__ == "__main__":
    main()
