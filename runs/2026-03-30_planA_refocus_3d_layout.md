# 2026-03-30 Plan A 目标纠偏与重设计记录

## 用户目标（已确认）
- 主任务：将山水画中的分割对象在三维空间中重现空间位置关系。
- 非主任务：三远法锚点/分区预测。
- 三远法定位：空间先验与正则约束。

## 本次改动
1. 重写架构文档：
   - `projects/SanYuan-LayoutNet/docs/architecture_planA.md`
   - 从“边界+分区预测”切换到“对象到3D布局重建”。
2. 重写训练手册：
   - `projects/SanYuan-LayoutNet/docs/training_manual_planA.md`
   - 训练流程改为 Stage A/B/C，主指标改为 Reproj-mIoU / Depth-Order Acc / Occlusion F1。

## 新架构摘要
- Object Encoder：对象特征编码
- Scene Relation Transformer：对象关系建模
- 3D Layout Head：回归对象3D中心/尺度/顺序
- Differentiable Projection：重投影监督
- SanYuan Prior Module：三远法软先验

## 损失函数
`L = L_proj + L_depth_rank + L_occ + L_rel + L_sanyuan_prior`

## 实验计划
- Phase 1：小样本可行性验证
- Phase 2：主实验与消融
- Phase 3：可视化与失败案例分析

## 下一步
1. 完成第一轮 smoke training（小样本）。
2. 将弱监督 target（当前由几何启发构造）替换为人工关系标注。
3. 增加验证集与评估脚本，输出 Reproj-mIoU / Depth-Order Acc / Occlusion F1。

## 本轮已落地代码（2026-03-30）
- 已更新 `configs/planA_base.yaml` 为对象级3D布局参数。
- 已重写 `src/data/dataset.py`：
  - 读取对象字段 `objects`（若缺失则安全回退）
  - 输出固定长度对象token、pair关系标签、三远先验桶
- 已重写 `src/models/layoutnet.py`：
  - `Object3DLayoutNet`（对象编码 + 关系Transformer + 3D输出头）
- 已重写 `src/losses/planA_loss.py`：
  - `L_proj/L_depth_rank/L_occ/L_rel/L_sanyuan_prior` 组合损失
- 已重写 `src/engine/trainer.py` 与 `scripts/train.py` 对齐新输入输出。
- 已执行语法检查：`python -m py_compile` 通过。

## Smoke Training（2026-03-30）
### 命令
```bash
$env:PYTHONPATH='projects/SanYuan-LayoutNet'; \
python projects/SanYuan-LayoutNet/scripts/train.py \
  --config projects/SanYuan-LayoutNet/configs/planA_smoke.yaml
```

### 结果（3 epoch）
- epoch1: total=0.01150, proj=0.00535, depth=0.00000, occ=0.00000
- epoch2: total=0.00354, proj=0.00183, depth=0.00000, occ=0.00000
- epoch3: total=0.00210, proj=0.00102, depth=0.00000, occ=0.00000

### 产出
- checkpoint: `outputs/planA_v2_smoke/epoch_001.pt`
- checkpoint: `outputs/planA_v2_smoke/epoch_002.pt`
- checkpoint: `outputs/planA_v2_smoke/epoch_003.pt`

### 观察
- 训练链路已跑通，损失稳定下降。
- `depth/occ` 近似 0：当前弱监督关系标签过于简化，下一步需替换为真实对象关系标注/更难负样本构造。

## Smoke Training v2（labels_v2_auto, 不使用 anchors, 2026-03-30）
### 调整
- 数据集切换为：`data/CLP_dataset/clp2k/labels_v2_auto`
- 训练输入明确不使用 `anchors`
- 数据读取优先 `instances + depth_pairs`

### 命令
```bash
$env:PYTHONPATH='projects/SanYuan-LayoutNet'; \
python projects/SanYuan-LayoutNet/scripts/train.py \
  --config projects/SanYuan-LayoutNet/configs/planA_smoke_auto.yaml
```

### 结果（3 epoch）
- epoch1: total=0.47968, proj=0.06688, depth=0.18460, occ=0.31707
- epoch2: total=0.30293, proj=0.02756, depth=0.08688, occ=0.27306
- epoch3: total=0.28589, proj=0.02134, depth=0.07904, occ=0.27474

### 结论
- 与上一轮（`labels_v2` 空标签）相比，这一轮 `depth/occ` 不再为 0，说明关系监督已真实生效。
- 下一步可优先扩大你已人工核验 `depth_pairs` 的样本比例，提升深度关系学习质量。

## Full Training（全数据集，labels_v2_auto，20 epoch，2026-03-30）
### 配置
- 配置文件：`projects/SanYuan-LayoutNet/configs/planA_full_auto.yaml`
- 数据：`data/CLP_dataset/clp2k/labels_v2_auto`（不使用 anchors）
- 输出：`outputs/planA_v2_full_auto/`
- 日志：`runs/2026-03-30_planA_full_auto_train.log`

### 关键结果
- epoch1: total=0.49395, proj=0.06871, depth=0.19290, occ=0.31951
- epoch10: total≈0.259（中期稳定下降）
- epoch18: total=0.25064, proj=0.00970, depth=0.06147, occ=0.27735（best total/depth）
- epoch20: total=0.25148, proj=0.00901, depth=0.06307, occ=0.27875（best proj）

### 分析
1. `total/proj/depth` 在前 5~8 个 epoch 快速下降，后期趋于平台，说明主干已学到稳定几何映射。
2. `depth` 从 0.1929 → 0.0615，下降明显，说明 `depth_pairs` 信号有效。
3. `occ` 从 0.3195 → 0.277~0.279，改善有限，提示当前遮挡监督仍偏弱（多由几何启发构造）。
4. 最优 epoch 在 18 左右，后续提升不明显，建议以 `epoch_018.pt` 作为当前主checkpoint。

### 下一步建议
- 优先补强人工核验的 `depth_pairs` 与真实 `occlusion` 标注，提升 `occ` 指标可学性。
- 增加验证脚本（Depth-Order Acc / Occlusion F1 / Reproj-mIoU）替代仅看训练 loss。
- 训练策略可尝试：冻结前层 + 调高 `w_occ`（如 0.6~0.8）做 5~10 epoch 微调。

## Occ Boost 微调实跑（labels_v2_auto_occ, 2026-03-31）
### 配置
- 配置文件：`projects/SanYuan-LayoutNet/configs/planA_occ_boost_auto_occ_5e.yaml`
- 数据：`data/CLP_dataset/clp2k/labels_v2_auto_occ`
- 关键参数：`w_occ=0.8`，`epochs=5`，`lr=1.5e-4`
- 日志：`runs/2026-03-31_planA_occ_boost_auto_occ_5e.log`
- 输出：`outputs/planA_v2_occ_boost_auto_occ_5e/`（本轮 `save_every=100`，不落 checkpoint）

### 命令
```bash
$env:PYTHONPATH='projects/SanYuan-LayoutNet'; \
python projects/SanYuan-LayoutNet/scripts/train.py \
  --config projects/SanYuan-LayoutNet/configs/planA_occ_boost_auto_occ_5e.yaml
```

### 结果（5 epoch）
- epoch1: total=0.60651, proj=0.07969, depth=0.20531, occ=0.31350
- epoch2: total=0.39363, proj=0.03769, depth=0.09034, occ=0.26189
- epoch3: total=0.37571, proj=0.03215, depth=0.08099, occ=0.26348
- epoch4: total=0.36666, proj=0.02768, depth=0.07750, occ=0.26631
- epoch5: total=0.35688, proj=0.02442, depth=0.07444, occ=0.26550

### 结论
- `total/proj/depth` 继续稳定下降，训练链路有效。
- `occ` 较基线首轮明显下降，但在 epoch2 后进入平台（约 0.262~0.266）。
- 当前瓶颈仍主要在遮挡监督质量，而非权重不足本身。
