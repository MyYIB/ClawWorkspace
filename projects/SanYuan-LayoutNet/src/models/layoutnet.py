from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class PolylineLayoutNet(nn.Module):
    def __init__(self, num_curve_points: int = 64):
        super().__init__()
        self.num_curve_points = num_curve_points

        backbone = tvm.resnet34(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_curve_points * 4),  # two curves, each point has (x,y)
            nn.Sigmoid(),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1),
        )

    def forward(self, x: torch.Tensor):
        feat = self.stem(x)

        pooled = self.pool(feat).flatten(1)
        curves = self.fc(pooled).view(-1, self.num_curve_points, 4)
        near_mid = curves[..., 0:2]
        mid_far = curves[..., 2:4]

        mask_logits = self.mask_head(feat)
        mask_logits = nn.functional.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "near_mid_curve": near_mid,
            "mid_far_curve": mid_far,
            "mask_logits": mask_logits,
        }
