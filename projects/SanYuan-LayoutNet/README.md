# SanYuan-LayoutNet (Plan A)

## 项目目标
围绕“空间位置重建”主线，建立一个可训练的三远法锚点与分区预测网络：
1. 预测两条关键边界线：`near_mid_boundary`、`mid_far_boundary`
2. 基于边界构造近/中/远三分区 soft mask
3. 为后续风格生成或布局控制提供可微、可复用的空间条件

## 输入输出
- 输入：
  - 图像 `image`
  - 标注 JSON（含 `anchors`）
- 输出：
  - 两条边界线（折线点集 / 稠密化曲线）
  - 三分区概率图（near/mid/far）
  - 训练日志与可视化结果

## 环境要求
- Python 3.10+
- PyTorch 2.2+
- CUDA（可选）

## 快速开始
```bash
pip install -r requirements.txt
python scripts/train.py --config configs/planA_base.yaml
```

## 目录结构
- `src/models/`：网络结构
- `src/data/`：数据集与预处理
- `src/losses/`：损失函数
- `src/engine/`：训练与评估循环
- `scripts/`：启动脚本
- `configs/`：实验配置
- `docs/`：架构设计与训练手册
- `outputs/`：模型、日志、可视化

## 当前状态
- [x] Plan A 架构设计文档
- [x] 训练手册 v1
- [x] 最小可运行项目骨架
- [ ] 数据接入与基线训练
- [ ] 指标对齐与消融实验
