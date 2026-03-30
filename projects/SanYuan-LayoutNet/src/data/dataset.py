from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PlanADataset(Dataset):
    """Object-to-3D layout dataset.

    优先读取自动标注格式：
    - instances
    - depth_pairs
    且明确不使用 anchors 作为训练输入。
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
        # v2_auto 格式
        if isinstance(data.get("instances"), list):
            return data["instances"]

        # 兼容旧格式
        objs = data.get("objects", [])
        if isinstance(objs, dict):
            objs = list(objs.values())
        if not isinstance(objs, list):
            objs = []
        return objs

    @staticmethod
    def _class_to_id(cls_name: str) -> float:
        if not cls_name:
            return 0.0
        # 稳定映射到[0,99]
        return float((abs(hash(cls_name)) % 100))

    def _object_to_features_and_target(self, obj: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        feat = np.zeros((16,), dtype=np.float32)

        # id
        obj_id = int(obj.get("id", 0))

        # center
        if isinstance(obj.get("center_xy"), list) and len(obj.get("center_xy")) >= 2:
            cx = self._safe_float(obj["center_xy"][0], 0.5 * self.image_size[0])
            cy = self._safe_float(obj["center_xy"][1], 0.5 * self.image_size[1])
        else:
            c = obj.get("centroid", obj.get("center", {})) or {}
            cx = self._safe_float(c.get("x", 0.5 * self.image_size[0]))
            cy = self._safe_float(c.get("y", 0.5 * self.image_size[1]))

        # bbox
        if isinstance(obj.get("bbox_xyxy"), list) and len(obj.get("bbox_xyxy")) >= 4:
            x1, y1, x2, y2 = [self._safe_float(v, 0.0) for v in obj["bbox_xyxy"][:4]]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
        else:
            b = obj.get("bbox", {}) or {}
            bw = self._safe_float(b.get("w", b.get("width", 0.15 * self.image_size[0])), 0.15 * self.image_size[0])
            bh = self._safe_float(b.get("h", b.get("height", 0.15 * self.image_size[1])), 0.15 * self.image_size[1])

        nx, ny = self._normalize_xy(cx, cy)
        nw = float(np.clip(bw / max(self.image_size[0], 1), 1e-4, 1.0))
        nh = float(np.clip(bh / max(self.image_size[1], 1), 1e-4, 1.0))
        area = float(np.clip(nw * nh, 1e-6, 1.0))
        aspect = float(np.clip(nw / max(nh, 1e-6), 0.05, 20.0))

        cls_id = self._safe_float(obj.get("class_id", self._class_to_id(str(obj.get("class", "")))), 0.0)
        cls_id_norm = float(np.clip(cls_id / 100.0, 0.0, 1.0))

        stroke = self._safe_float(obj.get("stroke_density", 0.5), 0.5)
        texture = self._safe_float(obj.get("texture_score", 0.5), 0.5)
        shape = self._safe_float(obj.get("shape_complexity", 0.5), 0.5)

        feat[:10] = np.array([nx, ny, nw, nh, area, aspect / 20.0, cls_id_norm, stroke, texture, shape], dtype=np.float32)

        center3d = np.array([nx, ny, float(np.clip(1.0 - ny, 0.0, 1.0))], dtype=np.float32)
        size3d = np.array([nw, nh, float(np.clip((nw + nh) * 0.5, 1e-4, 1.0))], dtype=np.float32)
        order = float(center3d[2])
        return feat, center3d, size3d, order, obj_id

    @staticmethod
    def _parse_depth_relation(rel: str) -> int | None:
        r = (rel or "").upper()
        if "A_FRONT_B" in r:
            return 0  # a closer than b => a not farther
        if "A_BEHIND_B" in r or "A_BACK_B" in r:
            return 1  # a farther than b
        return None

    def __getitem__(self, i):
        jp = self.label_files[i]
        data = json.loads(jp.read_text(encoding="utf-8"))

        image_path = data.get("image_path")
        if image_path is None:
            image_path = str(jp.with_suffix(".jpg"))

        img = Image.open(image_path).convert("RGB").resize(self.image_size)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        objects = self._extract_objects(data)
        if len(objects) < self.min_objects:
            objects = [{"id": 1, "center_xy": [self.image_size[0] * 0.5, self.image_size[1] * 0.5], "bbox_xyxy": [0, 0, self.image_size[0] * 0.2, self.image_size[1] * 0.2]}]

        n = min(len(objects), self.max_objects)

        obj_feat = np.zeros((self.max_objects, 16), dtype=np.float32)
        obj_valid = np.zeros((self.max_objects,), dtype=np.float32)
        target_center3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        target_size3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        target_order = np.zeros((self.max_objects,), dtype=np.float32)
        target_xy = np.zeros((self.max_objects, 2), dtype=np.float32)

        id_to_slot: dict[int, int] = {}

        for idx in range(n):
            feat, center3d, size3d, order, obj_id = self._object_to_features_and_target(objects[idx])
            obj_feat[idx] = feat
            obj_valid[idx] = 1.0
            target_center3d[idx] = center3d
            target_size3d[idx] = size3d
            target_order[idx] = order
            target_xy[idx] = center3d[:2]
            if obj_id != 0:
                id_to_slot[obj_id] = idx

        # pair-wise labels（-1 忽略）
        target_depth_pair = np.full((self.max_objects, self.max_objects), -1.0, dtype=np.float32)
        target_occ_pair = np.full((self.max_objects, self.max_objects), -1.0, dtype=np.float32)

        # 先用几何默认值填充有效对象对
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                target_depth_pair[a, b] = 1.0 if target_order[a] > target_order[b] else 0.0
                dx = abs(target_xy[a, 0] - target_xy[b, 0])
                dy = abs(target_xy[a, 1] - target_xy[b, 1])
                close_2d = (dx < 0.2) and (dy < 0.2)
                target_occ_pair[a, b] = 1.0 if close_2d and (target_order[a] < target_order[b]) else 0.0

        # 再用显式 depth_pairs 覆盖（如果存在）
        depth_pairs = data.get("depth_pairs", [])
        if isinstance(depth_pairs, list):
            for dp in depth_pairs:
                try:
                    a_id = int(dp.get("a_id", -1))
                    b_id = int(dp.get("b_id", -1))
                except Exception:
                    continue
                if a_id in id_to_slot and b_id in id_to_slot:
                    a = id_to_slot[a_id]
                    b = id_to_slot[b_id]
                    rel = self._parse_depth_relation(str(dp.get("relation", "")))
                    if rel is not None:
                        target_depth_pair[a, b] = float(rel)
                        target_depth_pair[b, a] = float(1 - rel)

        # 三远先验组（来自实例字段，若无则按z分桶）
        target_sanyuan_bin = np.zeros((self.max_objects, 3), dtype=np.float32)
        level_map = {"near": 0, "mid": 1, "far": 2}
        for idx in range(n):
            lvl = str(objects[idx].get("san_yuan_level", "")).lower()
            if lvl in level_map:
                target_sanyuan_bin[idx, level_map[lvl]] = 1.0
            else:
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
