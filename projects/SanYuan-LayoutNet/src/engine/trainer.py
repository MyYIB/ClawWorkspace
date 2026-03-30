from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.planA_loss import compute_loss


def train_one_epoch(model, loader: DataLoader, optimizer, device, loss_cfg):
    model.train()
    meter = {"total": 0.0}

    for batch in tqdm(loader, desc="train", leave=False):
        for k in ["image", "near_mid_curve", "mid_far_curve", "mask_target"]:
            batch[k] = batch[k].to(device)

        pred = model(batch["image"])
        losses = compute_loss(pred, batch, loss_cfg)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        meter["total"] += losses["total"].item()

    meter["total"] /= max(len(loader), 1)
    return meter
