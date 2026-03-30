from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.planA_loss import compute_loss


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def train_one_epoch(model, loader: DataLoader, optimizer, device, loss_cfg, grad_clip: float = 1.0):
    model.train()
    meter = {"total": 0.0, "proj": 0.0, "depth_rank": 0.0, "occ": 0.0}

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _to_device(batch, device)

        pred = model(batch["image"], batch["obj_feat"], batch["obj_valid"])
        losses = compute_loss(pred, batch, loss_cfg)

        optimizer.zero_grad()
        losses["total"].backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in meter:
            meter[k] += losses[k].item()

    n = max(len(loader), 1)
    for k in meter:
        meter[k] /= n
    return meter
