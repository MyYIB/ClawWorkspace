# Plan A 训练手册（v2：对象到3D空间位置重建）

## 1. 训练目标
训练一个对象级布局网络：输入山水画对象分割，输出对象在3D空间的相对位置与层次关系。

主指标：
- Reproj-mIoU
- Depth-Order Acc
- Occlusion F1

---

## 2. 数据准备
### 2.1 必备数据
- 图像：`data/CLP_dataset/clp2k/images`
- 对象分割：`data/CLP_dataset/clp2k/labels_v2`（至少含对象mask/轮廓）
- 对象关系弱标签（可选但推荐）：
  - 深度前后对（i,j）
  - 遮挡关系（谁挡谁）

### 2.2 标注字段建议
每图建议包含：
- `objects[]`: id, class, mask/bbox, centroid
- `relations[]`: pair_id, depth_order, occlusion
- `priors.sanyuan`（可选）: 对象级远近先验分组或分数

### 2.3 划分
- train/val/test = 8:1:1
- 保证不同画家/题材分布尽量均衡

---

## 3. 环境安装
```bash
cd projects/SanYuan-LayoutNet
pip install -r requirements.txt
```

---

## 4. 训练配置建议
主配置：`configs/planA_base.yaml`

建议新增或关注参数：
- `model.max_objects`
- `loss.w_proj`
- `loss.w_depth_rank`
- `loss.w_occ`
- `loss.w_rel`
- `loss.w_sanyuan_prior`
- `train.stage`（A/B/C）

初始权重推荐：
- `w_proj=1.0`
- `w_depth_rank=0.7`
- `w_occ=0.5`
- `w_rel=0.3`
- `w_sanyuan_prior=0.1`

---

## 5. 训练流程
### Stage A（5~10 epoch）
目标：稳定对象编码
- 降低关系与先验损失权重
- 观察对象特征是否可分

### Stage B（30~80 epoch）
目标：主任务收敛
- 打开重投影、深度排序、遮挡损失
- 以 `Depth-Order Acc` 与 `Reproj-mIoU` 为主要 early-stop 依据

### Stage C（10~20 epoch）
目标：结构微调
- 增加先验约束权重（仅小幅）
- 防止出现空间关系漂移

---

## 6. 训练命令
```bash
python scripts/train.py --config configs/planA_base.yaml
```

如需分阶段：
```bash
python scripts/train.py --config configs/planA_base.yaml --stage A
python scripts/train.py --config configs/planA_base.yaml --stage B
python scripts/train.py --config configs/planA_base.yaml --stage C
```

---

## 7. 评估与可视化
每个 epoch 至少记录：
- `loss_total`
- `reproj_mIoU`
- `depth_order_acc`
- `occlusion_f1`
- `rel_consistency`

可视化建议：
1. 原图 + 对象mask
2. 预测3D布局俯视图（scatter + box）
3. 重投影叠加图
4. 错误样本Top-K（排序冲突/遮挡冲突）

---

## 8. 消融实验模板
至少做以下对比：
- Baseline（无关系模块、无先验）
- +Relation
- +Relation +DepthRank
- Full（+SanYuanPrior）

记录项：
- 指标表
- 关键可视化
- 失败案例分析
- 下一步改动

---

## 9. 常见问题
1) **Depth-Order Acc 不升**
- 检查对象对标签是否噪声过高
- 提高 `w_depth_rank`，降低学习率

2) **Reproj-mIoU 高但空间层次差**
- 增加 `w_rel` 与 `w_occ`
- 强化对象关系采样（困难样本挖掘）

3) **先验过强导致模式塌缩**
- 降低 `w_sanyuan_prior`
- 先在 Stage B 收敛后再加先验
