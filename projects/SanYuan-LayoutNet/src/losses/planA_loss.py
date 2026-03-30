from __future__ import annotations

import torch
import torch.nn.functional as F


def smoothness_loss(curve: torch.Tensor) -> torch.Tensor:
    # curve: [B, K, 2]
    d1 = curve[:, 1:, :] - curve[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return d2.abs().mean()


def order_loss(near_mid: torch.Tensor, mid_far: torch.Tensor) -> torch.Tensor:
    # 假设 y 越大越远（可按数据定义调整）
    # 约束 near_mid 不应整体“远于” mid_far
    diff = near_mid[..., 1] - mid_far[..., 1]
    return F.relu(diff).mean()


def compute_loss(pred: dict, batch: dict, w: dict) -> dict:
    l_curve_1 = F.l1_loss(pred["near_mid_curve"], batch["near_mid_curve"])
    l_curve_2 = F.l1_loss(pred["mid_far_curve"], batch["mid_far_curve"])
    l_curve = l_curve_1 + l_curve_2

    l_smooth = smoothness_loss(pred["near_mid_curve"]) + smoothness_loss(pred["mid_far_curve"])
    l_order = order_loss(pred["near_mid_curve"], pred["mid_far_curve"])

    l_mask_ce = F.cross_entropy(pred["mask_logits"], batch["mask_target"])

    total = (
        w.get("w_curve_l1", 1.0) * l_curve
        + w.get("w_curve_smooth", 0.1) * l_smooth
        + w.get("w_order", 0.5) * l_order
        + w.get("w_mask_ce", 1.0) * l_mask_ce
    )

    return {
        "total": total,
        "curve": l_curve,
        "smooth": l_smooth,
        "order": l_order,
        "mask_ce": l_mask_ce,
    }
