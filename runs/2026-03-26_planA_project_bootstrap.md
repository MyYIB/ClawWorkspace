# 2026-03-26 Plan A 项目初始化记录

## 本次目标
1. 给出 Plan A（SanYuan-LayoutNet）网络架构设计
2. 给出训练手册 v1
3. 在 `projects/` 下创建最小可运行实验项目

## 产出文件
- `projects/SanYuan-LayoutNet/docs/architecture_planA.md`
- `projects/SanYuan-LayoutNet/docs/training_manual_planA.md`
- `projects/SanYuan-LayoutNet/README.md`
- `projects/SanYuan-LayoutNet/configs/planA_base.yaml`
- `projects/SanYuan-LayoutNet/scripts/train.py`
- `projects/SanYuan-LayoutNet/src/...`（model/data/loss/engine）

## 项目结构
已包含：
- `README.md`
- `requirements.txt`
- `src/`
- `scripts/`
- `configs/`
- `outputs/`
- `docs/`

## 说明
- 当前版本为“可运行基线骨架”，可直接启动训练流程。
- `mask_target` 仍为占位逻辑，下一步需按两条边界构造真实三分区标签。
