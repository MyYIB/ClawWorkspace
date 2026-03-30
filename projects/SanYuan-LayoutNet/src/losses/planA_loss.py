from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # pred/target: [B,N,C], valid:[B,N]
    m = valid.unsqueeze(-1)
    diff = (pred - target).abs() * m
    denom = m.sum() * pred.shape[-1] + 1e-6
    return diff.sum() / denom


def _pair_bce_with_ignore(logit: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # logit/target: [B,N,N], target=-1 ignore
    B, N, _ = target.shape
    vv = (valid.unsqueeze(2) * valid.unsqueeze(1)) > 0.5
    mask = (target >= 0.0) & vv
    if mask.sum() == 0:
        return logit.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logit[mask], target[mask])


def compute_loss(pred: dict, batch: dict, w: dict) -> dict:
    valid = batch["obj_valid"]

    # reprojection proxy: 3D center x,y should align object 2D centroid
    l_proj = _masked_l1(pred["center3d"][..., :2], batch["target_xy"], valid)

    # size consistency
    l_size = _masked_l1(pred["size3d"], batch["target_size3d"], valid)

    # depth rank / occlusion
    l_depth_rank = _pair_bce_with_ignore(pred["depth_pair_logit"], batch["target_depth_pair"], valid)
    l_occ = _pair_bce_with_ignore(pred["occ_pair_logit"], batch["target_occ_pair"], valid)

    # relation consistency proxy: center z should correlate with order target
    order_pred = torch.sigmoid(pred["order_logit"])
    l_rel = _masked_l1(order_pred.unsqueeze(-1), batch["target_order"].unsqueeze(-1), valid)

    # sanyuan prior
    B, N, _ = pred["sanyuan_logit"].shape
    logits = pred["sanyuan_logit"].reshape(B * N, 3)
    target_cls = batch["target_sanyuan_bin"].argmax(dim=-1).reshape(B * N)
    valid_flat = valid.reshape(B * N) > 0.5
    if valid_flat.sum() > 0:
        l_sanyuan = F.cross_entropy(logits[valid_flat], target_cls[valid_flat])
    else:
        l_sanyuan = logits.new_tensor(0.0)

    total = (
        w.get("w_proj", 1.0) * (l_proj + 0.5 * l_size)
        + w.get("w_depth_rank", 0.7) * l_depth_rank
        + w.get("w_occ", 0.5) * l_occ
        + w.get("w_rel", 0.3) * l_rel
        + w.get("w_sanyuan_prior", 0.1) * l_sanyuan
    )

    return {
        "total": total,
        "proj": l_proj,
        "size": l_size,
        "depth_rank": l_depth_rank,
        "occ": l_occ,
        "rel": l_rel,
        "sanyuan": l_sanyuan,
    }
