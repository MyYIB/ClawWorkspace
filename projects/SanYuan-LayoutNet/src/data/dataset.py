from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PlanADataset(Dataset):
    """Object-to-3D layout dataset (weakly supervised).

    Notes:
    - 优先读取对象级字段（objects/relations）。
    - 若缺失，回退到可运行的弱监督构造，保证训练流程不中断。
    """

    def __init__(self, label_root: str, image_size=(768, 768), max_objects: int = 32, min_objects: int = 1):
        self.label_files = sorted(Path(label_root).rglob("*.json"))
        self.image_size = tuple(image_size)
        self.max_objects = max_objects
        self.min_objects = min_objects

    def __len__(self):
        return len(self.label_files)

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _normalize_xy(self, x: float, y: float) -> tuple[float, float]:
        w, h = self.image_size[0], self.image_size[1]
        x = np.clip(x / max(w, 1), 0.0, 1.0)
        y = np.clip(y / max(h, 1), 0.0, 1.0)
        return float(x), float(y)

    def _extract_objects(self, data: dict) -> list[dict]:
        objs = data.get("objects", [])
        if isinstance(objs, dict):
            objs = list(objs.values())
        if not isinstance(objs, list):
            objs = []
        return objs

    def _object_to_features_and_target(self, obj: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        # 输入特征（16维）
        # [cx,cy,bw,bh,area,aspect,cls_id_norm,stroke,texture,shape, + zeros]
        feat = np.zeros((16,), dtype=np.float32)

        # 支持 centroid / bbox 多种字段名
        c = obj.get("centroid", obj.get("center", {})) or {}
        b = obj.get("bbox", {}) or {}

        cx = self._safe_float(c.get("x", b.get("x", 0.5 * self.image_size[0])))
        cy = self._safe_float(c.get("y", b.get("y", 0.5 * self.image_size[1])))
        bw = self._safe_float(b.get("w", b.get("width", 0.15 * self.image_size[0])), 0.15 * self.image_size[0])
        bh = self._safe_float(b.get("h", b.get("height", 0.15 * self.image_size[1])), 0.15 * self.image_size[1])

        nx, ny = self._normalize_xy(cx, cy)
        nw = float(np.clip(bw / max(self.image_size[0], 1), 1e-4, 1.0))
        nh = float(np.clip(bh / max(self.image_size[1], 1), 1e-4, 1.0))
        area = float(np.clip(nw * nh, 1e-6, 1.0))
        aspect = float(np.clip(nw / max(nh, 1e-6), 0.05, 20.0))

        cls_id = self._safe_float(obj.get("class_id", 0), 0.0)
        cls_id_norm = float(np.clip(cls_id / 100.0, 0.0, 1.0))

        stroke = self._safe_float(obj.get("stroke_density", 0.5), 0.5)
        texture = self._safe_float(obj.get("texture_score", 0.5), 0.5)
        shape = self._safe_float(obj.get("shape_complexity", 0.5), 0.5)

        feat[:10] = np.array([nx, ny, nw, nh, area, aspect / 20.0, cls_id_norm, stroke, texture, shape], dtype=np.float32)

        # 弱监督3D target
        # x,y: 来自2D中心; z: 先验=1-y（越上方越远，可后续替换为关系标注）
        center3d = np.array([nx, ny, float(np.clip(1.0 - ny, 0.0, 1.0))], dtype=np.float32)
        size3d = np.array([nw, nh, float(np.clip((nw + nh) * 0.5, 1e-4, 1.0))], dtype=np.float32)
        order = float(center3d[2])
        return feat, center3d, size3d, order

    def __getitem__(self, i):
        jp = self.label_files[i]
        data = json.loads(jp.read_text(encoding="utf-8"))

        image_path = data.get("image_path")
        if image_path is None:
            # fallback: 尝试同名图片
            image_path = str(jp.with_suffix(".jpg"))
        img = Image.open(image_path).convert("RGB").resize(self.image_size)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        objects = self._extract_objects(data)
        if len(objects) < self.min_objects:
            objects = [{"centroid": {"x": self.image_size[0] * 0.5, "y": self.image_size[1] * 0.5}, "bbox": {"w": self.image_size[0] * 0.2, "h": self.image_size[1] * 0.2}}]

        n = min(len(objects), self.max_objects)

        obj_feat = np.zeros((self.max_objects, 16), dtype=np.float32)
        obj_valid = np.zeros((self.max_objects,), dtype=np.float32)
        target_center3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        target_size3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        target_order = np.zeros((self.max_objects,), dtype=np.float32)
        target_xy = np.zeros((self.max_objects, 2), dtype=np.float32)

        for idx in range(n):
            feat, center3d, size3d, order = self._object_to_features_and_target(objects[idx])
            obj_feat[idx] = feat
            obj_valid[idx] = 1.0
            target_center3d[idx] = center3d
            target_size3d[idx] = size3d
            target_order[idx] = order
            target_xy[idx] = center3d[:2]

        # pair-wise labels（-1 忽略）
        target_depth_pair = np.full((self.max_objects, self.max_objects), -1.0, dtype=np.float32)
        target_occ_pair = np.full((self.max_objects, self.max_objects), -1.0, dtype=np.float32)

        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                # depth: 1 表示 a 比 b 更远
                target_depth_pair[a, b] = 1.0 if target_order[a] > target_order[b] else 0.0
                # occ weak label: 若x/y接近且a更近，a可能遮挡b
                dx = abs(target_xy[a, 0] - target_xy[b, 0])
                dy = abs(target_xy[a, 1] - target_xy[b, 1])
                close_2d = (dx < 0.2) and (dy < 0.2)
                target_occ_pair[a, b] = 1.0 if close_2d and (target_order[a] < target_order[b]) else 0.0

        # 三远先验组（近/中/远）: 按order分桶
        target_sanyuan_bin = np.zeros((self.max_objects, 3), dtype=np.float32)
        for idx in range(n):
            z = target_order[idx]
            if z < 0.33:
                target_sanyuan_bin[idx, 0] = 1.0
            elif z < 0.66:
                target_sanyuan_bin[idx, 1] = 1.0
            else:
                target_sanyuan_bin[idx, 2] = 1.0

        return {
            "image": img,
            "obj_feat": torch.from_numpy(obj_feat),
            "obj_valid": torch.from_numpy(obj_valid),
            "target_center3d": torch.from_numpy(target_center3d),
            "target_size3d": torch.from_numpy(target_size3d),
            "target_order": torch.from_numpy(target_order),
            "target_xy": torch.from_numpy(target_xy),
            "target_depth_pair": torch.from_numpy(target_depth_pair),
            "target_occ_pair": torch.from_numpy(target_occ_pair),
            "target_sanyuan_bin": torch.from_numpy(target_sanyuan_bin),
        }
