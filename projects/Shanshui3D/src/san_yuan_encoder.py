"""
三远法编码器 - SanYuan Encoder

将中国传统山水画的"三远法"理论形式化为可计算的空间约束条件。

三远法定义（郭熙《林泉高致》）:
- 高远：自山下而仰山巅（仰视视角，强调山势高耸）
- 深远：自山前而窥山后（纵深透视，强调层次深远）
- 平远：自近山而望远山（平视远眺，强调开阔平缓）

本模块将三远法编码为：
1. 相机姿态约束
2. 深度分布约束
3. 空间注意力权重
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SanYuanPreset:
    """三远法意境预设"""
    gaoyuan: float = 0.5   # 高远强度 [0, 1]
    shenyuan: float = 0.5  # 深远强度 [0, 1]
    pingyuan: float = 0.5  # 平远强度 [0, 1]
    
    def validate(self):
        """验证参数范围"""
        for name, value in [("gaoyuan", self.gaoyuan), 
                           ("shenyuan", self.shenyuan), 
                           ("pingyuan", self.pingyuan)]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
    
    def normalize(self):
        """归一化使总和为 1（可选）"""
        total = self.gaoyuan + self.shenyuan + self.pingyuan
        if total > 0:
            self.gaoyuan /= total
            self.shenyuan /= total
            self.pingyuan /= total
        return self


class SanYuanEncoder(nn.Module):
    """
    三远法编码器
    
    输入：三远法参数 (gaoyuan, shenyuan, pingyuan)
    输出：空间约束条件，用于指导 3D 重建和渲染
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_layers: int = 3,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 三远法参数编码网络
        layers = []
        input_dim = 3  # (gaoyuan, shenyuan, pingyuan)
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            ])
        
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出投影：相机约束、深度约束、注意力权重
        self.camera_head = nn.Linear(embedding_dim, 6)  # [yaw, pitch, roll, fov, dist_x, dist_y]
        self.depth_head = nn.Linear(embedding_dim, 3)   # [depth_scale, depth_bias, depth_var]
        self.attention_head = nn.Linear(embedding_dim, 3)  # [vertical_weight, horizontal_weight, depth_weight]
    
    def forward(
        self,
        preset: SanYuanPreset,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        编码三远法参数为空间约束
        
        Args:
            preset: 三远法预设参数
            batch_size: 批次大小
            
        Returns:
            包含相机约束、深度约束、注意力权重的字典
        """
        preset.validate()
        
        # 构建输入张量 [B, 3]
        san_yuan_input = torch.tensor(
            [[preset.gaoyuan, preset.shenyuan, preset.pingyuan]],
            dtype=torch.float32,
        ).repeat(batch_size, 1)
        
        # 编码
        embedding = self.encoder(san_yuan_input)  # [B, embedding_dim]
        
        # 解码为不同约束
        camera_constraints = self.camera_head(embedding)
        depth_constraints = self.depth_head(embedding)
        attention_weights = torch.softmax(self.attention_head(embedding), dim=-1)
        
        return {
            "embedding": embedding,
            "camera_constraints": camera_constraints,
            "depth_constraints": depth_constraints,
            "attention_weights": attention_weights,
            "preset": preset,
        }
    
    def get_preset_from_name(self, name: str) -> SanYuanPreset:
        """
        根据预设名称获取三远法参数
        
        预设定义参考传统山水画的典型构图：
        - "monumental": 纪念碑式高远（如范宽《溪山行旅图》）
        - "layered": 层峦叠嶂深远（如王蒙《青卞隐居图》）
        - "expansive": 开阔平远（如倪瓒《容膝斋图》）
        - "balanced": 三远平衡
        """
        presets = {
            "monumental": SanYuanPreset(gaoyuan=0.8, shenyuan=0.3, pingyuan=0.2),
            "layered": SanYuanPreset(gaoyuan=0.3, shenyuan=0.8, pingyuan=0.3),
            "expansive": SanYuanPreset(gaoyuan=0.2, shenyuan=0.3, pingyuan=0.8),
            "balanced": SanYuanPreset(gaoyuan=0.5, shenyuan=0.5, pingyuan=0.5),
        }
        
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        
        return presets[name]


class SanYuanLoss(nn.Module):
    """
    三远法约束损失
    
    用于在训练过程中强制模型遵循三远法空间约束
    """
    
    def __init__(
        self,
        camera_weight: float = 1.0,
        depth_weight: float = 1.0,
        attention_weight: float = 0.5,
    ):
        super().__init__()
        self.camera_weight = camera_weight
        self.depth_weight = depth_weight
        self.attention_weight = attention_weight
        
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predicted_constraints: Dict[str, torch.Tensor],
        target_constraints: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算三远法约束损失
        
        Args:
            predicted_constraints: 模型预测的约束
            target_constraints: 目标约束（来自 SanYuanEncoder）
            
        Returns:
            包含各项损失和总损失的字典
        """
        camera_loss = self.mse_loss(
            predicted_constraints["camera_constraints"],
            target_constraints["camera_constraints"],
        )
        
        depth_loss = self.mse_loss(
            predicted_constraints["depth_constraints"],
            target_constraints["depth_constraints"],
        )
        
        attention_loss = self.mse_loss(
            predicted_constraints["attention_weights"],
            target_constraints["attention_weights"],
        )
        
        total_loss = (
            self.camera_weight * camera_loss +
            self.depth_weight * depth_loss +
            self.attention_weight * attention_loss
        )
        
        return {
            "total_loss": total_loss,
            "camera_loss": camera_loss,
            "depth_loss": depth_loss,
            "attention_loss": attention_loss,
        }


# 测试代码
if __name__ == "__main__":
    print("Testing SanYuanEncoder...")
    
    encoder = SanYuanEncoder()
    preset = SanYuanPreset(gaoyuan=0.7, shenyuan=0.5, pingyuan=0.3)
    
    output = encoder(preset, batch_size=2)
    
    print(f"Input preset: {preset}")
    print(f"Output embedding shape: {output['embedding'].shape}")
    print(f"Output camera constraints shape: {output['camera_constraints'].shape}")
    print(f"Output depth constraints shape: {output['depth_constraints'].shape}")
    print(f"Output attention weights shape: {output['attention_weights'].shape}")
    
    # 测试预设
    for name in ["monumental", "layered", "expansive", "balanced"]:
        p = encoder.get_preset_from_name(name)
        print(f"Preset '{name}': G={p.gaoyuan:.2f}, S={p.shenyuan:.2f}, P={p.pingyuan:.2f}")
    
    print("\nSanYuanEncoder test passed!")
