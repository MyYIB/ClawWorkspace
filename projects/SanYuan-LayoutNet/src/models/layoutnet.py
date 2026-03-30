from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class Object3DLayoutNet(nn.Module):
    def __init__(self, obj_feat_dim: int = 16, d_model: int = 256, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        backbone = tvm.resnet18(weights=None)
        self.image_stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_proj = nn.Linear(512, d_model)

        self.obj_proj = nn.Sequential(
            nn.Linear(obj_feat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.relation_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.center_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 3), nn.Sigmoid())
        self.size_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 3), nn.Sigmoid())
        self.order_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True), nn.Linear(d_model // 2, 1))
        self.sanyuan_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True), nn.Linear(d_model // 2, 3))

    def forward(self, image: torch.Tensor, obj_feat: torch.Tensor, obj_valid: torch.Tensor):
        # image global token
        img_feat = self.image_stem(image)
        img_token = self.image_pool(img_feat).flatten(1)
        img_token = self.image_proj(img_token).unsqueeze(1)  # [B,1,D]

        # object tokens
        obj_token = self.obj_proj(obj_feat)  # [B,N,D]
        obj_token = obj_token + img_token  # global conditioning

        # key padding mask: True means ignore
        key_padding_mask = obj_valid < 0.5
        h = self.relation_encoder(obj_token, src_key_padding_mask=key_padding_mask)

        center3d = self.center_head(h)
        size3d = self.size_head(h)
        order_logit = self.order_head(h).squeeze(-1)
        sanyuan_logit = self.sanyuan_head(h)

        # pair-wise logits by order difference
        depth_pair_logit = order_logit.unsqueeze(2) - order_logit.unsqueeze(1)

        # occ pair logit by geometry heuristic in latent space
        # 使用xy近邻 + 深度差的可学习近似
        xy = center3d[..., :2]
        z = center3d[..., 2]
        dist = torch.cdist(xy, xy, p=2)
        z_diff = z.unsqueeze(2) - z.unsqueeze(1)
        occ_pair_logit = -3.0 * dist + 2.0 * z_diff

        return {
            "center3d": center3d,
            "size3d": size3d,
            "order_logit": order_logit,
            "sanyuan_logit": sanyuan_logit,
            "depth_pair_logit": depth_pair_logit,
            "occ_pair_logit": occ_pair_logit,
        }
