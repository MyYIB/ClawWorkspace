#!/usr/bin/env python
"""
Shanshui3D - 推理脚本

用法:
    python scripts/infer.py --config configs/base.yaml --input <painting_image> --output <output_dir>
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from src.san_yuan_encoder import SanYuanEncoder, SanYuanPreset
from src.reconstruction import ReconstructionModule, ReconstructionConfig
from src.style_transfer import StyleTransferModule, StyleConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Shanshui3D Inference")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--input", type=str, required=True, help="输入山水画路径")
    parser.add_argument("--output", type=str, default="outputs/inference", help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--preset", type=str, default="balanced", 
                       choices=["monumental", "layered", "expansive", "balanced"],
                       help="三远法意境预设")
    parser.add_argument("--gaoyuan", type=float, default=None, help="高远强度 (覆盖 preset)")
    parser.add_argument("--shenyuan", type=float, default=None, help="深远强度 (覆盖 preset)")
    parser.add_argument("--pingyuan", type=float, default=None, help="平远强度 (覆盖 preset)")
    parser.add_argument("--render_views", type=int, default=8, help="渲染视角数量")
    return parser.parse_args()


def load_image(path: str, size: tuple = None) -> torch.Tensor:
    """加载图像并转换为 tensor"""
    image = Image.open(path).convert("RGB")
    if size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    
    # 归一化到 [0, 1]
    image_np = np.array(image) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor


def load_checkpoint(model_dict: dict, checkpoint_path: str, device: torch.device):
    """加载检查点"""
    if checkpoint_path is None:
        print("No checkpoint provided, using random initialization")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    for name, model in model_dict.items():
        if name in checkpoint:
            model.load_state_dict(checkpoint[name])
            print(f"Loaded {name} from checkpoint")
        else:
            print(f"Warning: {name} not found in checkpoint")


def main():
    args = parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    print(f"Loaded config: {args.config}")
    
    # 设置设备
    device = torch.device("cuda" if config.device.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载输入图像
    print(f"\nLoading input image: {args.input}")
    input_image = load_image(args.input, size=(512, 512)).to(device)
    
    # 构建三远法预设
    preset = SanYuanPreset()
    if args.gaoyuan is not None:
        preset.gaoyuan = args.gaoyuan
    if args.shenyuan is not None:
        preset.shenyuan = args.shenyuan
    if args.pingyuan is not None:
        preset.pingyuan = args.pingyuan
    
    # 或者使用预设
    if args.gaoyuan is None and args.shenyuan is None and args.pingyuan is None:
        encoder_temp = SanYuanEncoder()
        preset = encoder_temp.get_preset_from_name(args.preset)
    
    print(f"\nSanYuan Preset: G={preset.gaoyuan:.2f}, S={preset.shenyuan:.2f}, P={preset.pingyuan:.2f}")
    
    # 构建模型
    print("\nBuilding models...")
    san_yuan_encoder = SanYuanEncoder().to(device)
    
    recon_config = ReconstructionConfig(
        method=config.reconstruction.method,
        num_layers=config.reconstruction.num_layers,
        hidden_dim=config.reconstruction.hidden_dim,
        num_freqs=config.reconstruction.num_freqs,
    )
    reconstruction = ReconstructionModule(recon_config).to(device)
    
    style_config = StyleConfig(
        method=config.style.method,
        style_weight=config.style.style_weight,
        content_weight=config.style.content_weight,
        ink_strength=config.style.chinese_painting.ink_strength,
    )
    style_transfer = StyleTransferModule(style_config).to(device)
    
    # 加载检查点
    load_checkpoint({
        "san_yuan_encoder": san_yuan_encoder,
        "reconstruction": reconstruction,
        "style_transfer": style_transfer,
    }, args.checkpoint, device)
    
    # 编码三远法约束
    print("\nEncoding SanYuan constraints...")
    san_yuan_constraints = san_yuan_encoder(preset, batch_size=1)
    
    # 推理（占位符 - 需要完整实现）
    print("\n" + "="*50)
    print("Inference (placeholder - model not trained yet)")
    print("="*50)
    
    # 模拟渲染多个视角
    print(f"\nRendering {args.render_views} views...")
    
    for view_idx in tqdm(range(args.render_views)):
        # TODO: 实现完整的渲染逻辑
        # 这里是占位符输出
        
        # 模拟渲染输出
        rendered_rgb = torch.rand(1, 3, 256, 256, device=device)
        rendered_depth = torch.rand(1, 1, 256, 256, device=device)
        
        # 风格迁移
        stylized = style_transfer(rendered_rgb, rendered_depth)
        
        # 保存结果
        output_image = stylized["stylized_rgb"][0].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image_pil = Image.fromarray(output_image)
        
        output_path = output_dir / f"view_{view_idx:03d}_{args.preset}.png"
        output_image_pil.save(output_path)
    
    # 保存三远法参数
    import json
    metadata = {
        "input_image": args.input,
        "preset": args.preset,
        "san_yuan": {
            "gaoyuan": preset.gaoyuan,
            "shenyuan": preset.shenyuan,
            "pingyuan": preset.pingyuan,
        },
        "render_views": args.render_views,
        "config": args.config,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Inference completed!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Rendered {args.render_views} views")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
