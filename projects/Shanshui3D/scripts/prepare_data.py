#!/usr/bin/env python
"""
Shanshui3D - 数据准备脚本

用法:
    python scripts/prepare_data.py --input <raw_data_dir> --output <processed_data_dir>
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Shanshui3D Data Preparation")
    parser.add_argument("--input", type=str, required=True, help="原始数据目录")
    parser.add_argument("--output", type=str, required=True, help="处理后数据输出目录")
    parser.add_argument("--resize", type=int, default=512, help="调整图像大小")
    return parser.parse_args()


def process_image(input_path: Path, output_path: Path, size: int):
    """处理单张图像"""
    image = Image.open(input_path).convert("RGB")
    
    # 调整大小
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    return {
        "input": str(input_path),
        "output": str(output_path),
        "size": (size, size),
    }


def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有图像
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    print(f"Processing to {output_dir}")
    print(f"Resize to: {args.resize}x{args.resize}")
    print("\n" + "="*50)
    
    # 处理图像
    results = []
    for i, img_path in enumerate(image_files, 1):
        output_path = output_dir / f"{img_path.stem}_processed.png"
        
        try:
            result = process_image(img_path, output_path, args.resize)
            results.append(result)
            print(f"[{i}/{len(image_files)}] Processed: {img_path.name}")
        except Exception as e:
            print(f"[{i}/{len(image_files)}] Failed: {img_path.name} - {e}")
    
    # 保存处理记录
    import json
    record_path = output_dir / "processing_record.json"
    with open(record_path, "w") as f:
        json.dump({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "resize": args.resize,
            "processed_count": len(results),
            "files": results,
        }, f, indent=2)
    
    print("\n" + "="*50)
    print(f"✅ Data preparation completed!")
    print(f"Processed: {len(results)}/{len(image_files)} images")
    print(f"Output: {output_dir}")
    print(f"Record: {record_path}")


if __name__ == "__main__":
    main()
