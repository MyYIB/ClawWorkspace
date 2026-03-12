"""
3D 重建模块 - Reconstruction Module

支持多种 3D 重建方法：
- NeRF (Neural Radiance Fields)
- 3D Gaussian Splatting (待实现)

本模块负责从单张/多张山水画重建 3D 场景。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ReconstructionConfig:
    """重建配置"""
    method: str = "nerf"  # nerf, 3dgs
    num_layers: int = 8
    hidden_dim: int = 256
    num_freqs: int = 10
    aabb_scale: float = 2.0


class PositionalEncoding(nn.Module):
    """位置编码（NeRF 风格）"""
    
    def __init__(self, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入坐标 [B, 3]
        Returns:
            编码后的特征 [B, 3 + 3*2*num_freqs]
        """
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    """
    基础 NeRF 模型
    
    输入：3D 坐标 + 视角方向
    输出：RGB 颜色 + 体密度
    """
    
    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 256,
        num_freqs_pos: int = 10,
        num_freqs_dir: int = 4,
    ):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(num_freqs_pos)
        self.dir_encoder = PositionalEncoding(num_freqs_dir)
        
        pos_dim = 3 + 3 * 2 * num_freqs_pos
        dir_dim = 3 + 3 * 2 * num_freqs_dir
        
        # 密度网络（只依赖位置）
        density_layers = []
        for i in range(num_layers):
            density_layers.extend([
                nn.Linear(pos_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.density_net = nn.Sequential(*density_layers)
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # 颜色网络（依赖位置 + 方向）
        color_layers = []
        for i in range(num_layers // 2):
            color_layers.extend([
                nn.Linear(pos_dim + dir_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.color_net = nn.Sequential(*color_layers)
        self.color_head = nn.Linear(hidden_dim, 3)
        self.color_activation = nn.Sigmoid()
    
    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            positions: 3D 坐标 [B, 3]
            directions: 视角方向 [B, 3]
            
        Returns:
            包含 rgb 和 density 的字典
        """
        # 编码
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        
        # 密度
        density_feat = self.density_net(pos_encoded)
        density = self.density_head(density_feat)
        density = torch.relu(density)  # 密度非负
        
        # 颜色
        color_feat = self.color_net(torch.cat([pos_encoded, dir_encoded], dim=-1))
        rgb = self.color_head(color_feat)
        rgb = self.color_activation(rgb)
        
        return {
            "rgb": rgb,
            "density": density,
        }


class ReconstructionModule(nn.Module):
    """
    重建模块 - 统一接口
    
    支持 NeRF 和 3DGS 等多种方法
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__()
        self.config = config
        
        if config.method == "nerf":
            self.model = NeRF(
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                num_freqs_pos=config.num_freqs,
            )
        else:
            raise NotImplementedError(f"Method {config.method} not implemented")
    
    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        san_yuan_constraints: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            positions: 3D 采样点坐标 [B, N, 3]
            directions: 视角方向 [B, N, 3]
            san_yuan_constraints: 三远法约束（可选）
            
        Returns:
            渲染结果
        """
        # 基础 NeRF 输出
        B, N, _ = positions.shape
        positions_flat = positions.view(-1, 3)
        directions_flat = directions.view(-1, 3)
        
        output = self.model(positions_flat, directions_flat)
        
        # 恢复形状
        rgb = output["rgb"].view(B, N, 3)
        density = output["density"].view(B, N, 1)
        
        # 应用三远法约束（如果提供）
        if san_yuan_constraints is not None:
            # 这里可以应用深度约束、注意力权重等
            # TODO: 实现三远法约束的具体应用
            attention = san_yuan_constraints.get("attention_weights")
            if attention is not None:
                # 示例：用注意力权重调制密度
                density = density * attention[:, 1:2]  # 使用深远权重
        
        return {
            "rgb": rgb,
            "density": density,
            "san_yuan_applied": san_yuan_constraints is not None,
        }
    
    def render(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        num_samples: int = 64,
        near: float = 0.1,
        far: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """
        体渲染
        
        Args:
            rays_o: 射线原点 [B, 3]
            rays_d: 射线方向 [B, 3]
            num_samples: 每条射线的采样点数
            near: 近裁剪面
            far: 远裁剪面
            
        Returns:
            渲染的 RGB 图像和深度图
        """
        B, _ = rays_o.shape
        
        # 沿射线采样
        t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
        t_vals = t_vals.view(1, -1, 1).repeat(B, 1, 1)  # [B, num_samples, 1]
        
        positions = rays_o.unsqueeze(1) + t_vals * rays_d.unsqueeze(1)  # [B, num_samples, 3]
        directions = rays_d.unsqueeze(1).expand(-1, num_samples, -1)
        
        # 前向传播
        output = self.forward(positions, directions)
        
        # 体渲染（简化版）
        rgb, density = output["rgb"], output["density"]
        
        # 计算透射率
        delta = t_vals[:, 1:, :] - t_vals[:, :-1, :]  # [B, num_samples-1, 1]
        alpha = 1 - torch.exp(-density[:, :-1, :] * delta)  # [B, num_samples-1, 1]
        
        # 权重
        transmittance = torch.cumprod(
            torch.cat([torch.ones(B, 1, 1, device=alpha.device), 1 - alpha + 1e-10], dim=1),
            dim=1
        )[:, :-1, :]
        weights = alpha * transmittance
        
        # 加权求和
        rgb_rendered = torch.sum(weights * rgb[:, :-1, :], dim=1)
        depth_rendered = torch.sum(weights * t_vals[:, :-1, :], dim=1)
        
        return {
            "rgb": rgb_rendered,
            "depth": depth_rendered,
            "weights": weights,
        }


# 测试代码
if __name__ == "__main__":
    print("Testing ReconstructionModule...")
    
    config = ReconstructionConfig(method="nerf")
    model = ReconstructionModule(config)
    
    # 测试前向传播
    B, N = 2, 100
    positions = torch.randn(B, N, 3)
    directions = torch.randn(B, N, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    
    output = model(positions, directions)
    print(f"Output rgb shape: {output['rgb'].shape}")
    print(f"Output density shape: {output['density'].shape}")
    
    # 测试渲染
    rays_o = torch.randn(B, 3)
    rays_d = torch.randn(B, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    
    render_output = model.render(rays_o, rays_d, num_samples=64)
    print(f"Rendered rgb shape: {render_output['rgb'].shape}")
    print(f"Rendered depth shape: {render_output['depth'].shape}")
    
    print("\nReconstructionModule test passed!")
