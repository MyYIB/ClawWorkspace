# SanYuan-LayoutNet (Plan A)

## 项目目标
围绕“空间位置重建”主线，建立一个可训练的对象级 3D 布局网络：
1. 从山水画分割对象中提取对象与关系特征
2. 重建对象在三维空间中的位置、尺度与前后层次
3. 通过可微重投影与关系约束保证2D观测一致性
4. 为后续风格生成或布局控制提供结构化空间条件

## 输入输出
- 输入：
  - 图像 `image`
  - 标注 JSON（含对象分割与关系字段）
- 输出：
  - 对象级 3D 布局参数（位置/尺度/深度顺序）
  - 重投影一致性结果与关系预测结果
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
