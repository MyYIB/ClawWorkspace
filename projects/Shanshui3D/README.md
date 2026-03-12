# Shanshui3D - 山水画空间意境重现

基于三远法（高远、深远、平远）的中国山水画 3D 空间意境重现系统。

## 项目目标

将中国传统山水画的"三远法"空间理论形式化，并结合现代 3D 重建技术（NeRF/3DGS），实现：
1. 从单张/多张山水画重建 3D 场景
2. 支持三远法空间控制（调整高远/深远/平远的意境强度）
3. 保持中国画笔墨风格的 3D 风格化

## 创新点

- **首次将三远法形式化为 3D 空间控制条件**
- **首次实现山水画意境的 3D 重现**
- **结合传统画论与现代神经渲染**

## 技术路线

```
输入（山水画）→ 深度/法线估计 → 三远法空间编码 → NeRF/3DGS 重建 → 风格化渲染 → 输出（3D 场景）
```

### 核心模块
1. **三远法编码器**：将高远/深远/平远形式化为空间约束
2. **3D 重建模块**：基于 NeRF 或 3D Gaussian Splatting
3. **风格迁移模块**：保持中国画笔墨风格
4. **可控渲染模块**：支持意境参数的实时调整

## 环境要求

- Python >= 3.10
- CUDA >= 11.8
- PyTorch >= 2.0
- Nerfstudio (可选)

## 安装

```bash
# 克隆项目
cd projects/Shanshui3D

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 运行方式

### 1. 数据准备
```bash
python scripts/prepare_data.py --input data/raw --output data/processed
```

### 2. 训练
```bash
python scripts/train.py --config configs/base.yaml
```

### 3. 推理
```bash
python scripts/infer.py --config configs/base.yaml --input <painting_image> --output outputs/
```

### 4. 交互式调整（待实现）
```bash
python scripts/viewer.py --scene outputs/scene_<id>
```

## 当前状态

- [x] 项目框架创建
- [x] 论文调研完成
- [ ] 核心论文深入阅读（CoARF, CCLAP）
- [ ] 技术栈确定
- [ ] 数据集准备
- [ ] 基线方法复现
- [ ] 三远法模块设计
- [ ] 系统集成
- [ ] 实验验证

## 目录结构

```
Shanshui3D/
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml          # 基础配置
├── src/
│   ├── __init__.py
│   ├── san_yuan_encoder.py # 三远法编码器（待实现）
│   ├── reconstruction.py   # 3D 重建模块（待实现）
│   └── style_transfer.py   # 风格迁移（待实现）
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── infer.py
├── data/
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后数据
└── outputs/
    ├── checkpoints/       # 模型检查点
    ├── renders/           # 渲染结果
    └── logs/              # 训练日志
```

## 参考论文

1. **CoARF** (2024): Controllable 3D Artistic Style Transfer for Radiance Fields
2. **CCLAP** (2023): Controllable Chinese Landscape Painting Generation via Latent Diffusion Model
3. **ConCLVD** (2024): Controllable Chinese Landscape Video Generation via Diffusion Model

## 团队成员

- 肖总（PI）
- 小迪（AI 研究助理）

## 许可证

TBD

## 联系方式

TBD

---
*项目创建时间：2026-03-07 | 版本：v0.1*
