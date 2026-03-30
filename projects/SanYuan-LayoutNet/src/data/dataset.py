from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PlanADataset(Dataset):
    def __init__(self, label_root: str, image_size=(768, 768)):
        self.label_files = sorted(Path(label_root).rglob("*.json"))
        self.image_size = image_size

    def __len__(self):
        return len(self.label_files)

    def _resample_curve(self, pts, k=64):
        if not pts:
            return np.zeros((k, 2), dtype=np.float32)
        arr = np.array(pts, dtype=np.float32)
        idx = np.linspace(0, len(arr) - 1, k)
        left = np.floor(idx).astype(int)
        right = np.clip(left + 1, 0, len(arr) - 1)
        alpha = (idx - left)[:, None]
        return arr[left] * (1 - alpha) + arr[right] * alpha

    def __getitem__(self, i):
        jp = self.label_files[i]
        data = json.loads(jp.read_text(encoding="utf-8"))

        img = Image.open(data["image_path"]).convert("RGB").resize(self.image_size)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        anchors = data.get("anchors", {})
        near_mid = self._resample_curve(anchors.get("near_mid_boundary", []))
        mid_far = self._resample_curve(anchors.get("mid_far_boundary", []))

        near_mid = torch.from_numpy(near_mid / np.array(self.image_size[::-1], dtype=np.float32))
        mid_far = torch.from_numpy(mid_far / np.array(self.image_size[::-1], dtype=np.float32))

        # TODO: 用真实几何逻辑构建 mask_target
        mask_target = torch.zeros(self.image_size, dtype=torch.long)

        return {
            "image": img,
            "near_mid_curve": near_mid,
            "mid_far_curve": mid_far,
            "mask_target": mask_target,
        }
