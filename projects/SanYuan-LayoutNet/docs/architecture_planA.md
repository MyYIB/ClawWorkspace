# Plan A 网络架构设计（SanYuan-LayoutNet v2）

## 0. 任务定义（纠偏后）
**核心任务**：将山水画中的分割对象重建到统一三维空间，恢复对象间空间位置关系（远近、前后、层次、遮挡）。

- 输入：单幅山水图 + 对象分割（实例或语义对象掩码）
- 输出：对象级 3D 布局（center/scale/depth/order）
- 三远法角色：**空间先验约束**，不是最终预测目标

---

## 1. 总体结构
`Image + Masks -> Object Encoder -> Scene Relation Transformer -> 3D Layout Head -> Differentiable Projection`

### 1.1 Object Encoder（对象编码器）
对每个对象提取融合特征：
- 局部视觉特征（纹理、笔触、语义）
- 掩码几何特征（面积、长宽比、重心、边界复杂度）
- 2D 位置特征（归一化 bbox / centroid）

建议实现：
- Backbone：ConvNeXt-T / ViT-S（二选一）
- Mask Pooling：RoIAlign + Masked Average Pooling
- 输出：对象 token `f_i ∈ R^D`

### 1.2 Scene Relation Transformer（关系建模）
构建对象级关系图并全局推理：
- 节点：对象 token
- 边：对象对相对关系（上下、左右、接触、重叠、潜在遮挡）
- 模块：Transformer Encoder（含 pair-wise relation bias）

输出：关系增强对象表示 `h_i`

### 1.3 3D Layout Head（核心预测头）
对每个对象回归：
- `center_i = (x_i, y_i, z_i)`
- `size_i = (w_i, h_i, d_i)`
- `order_i`（深度顺序logit）
- `occ_i`（可见性/遮挡强度，可选）

说明：
- `x,y` 对应画面横纵空间映射
- `z` 反映前后纵深
- `order` 用于显式排序监督（比纯回归更稳定）

### 1.4 Differentiable Projection（可微重投影）
将 3D 布局重投影回 2D，得到对象投影 mask / depth map，与输入掩码及关系标签对齐。

作用：
1. 提供无绝对3D GT时的可训练监督通路
2. 保证预测3D结构与原图几何一致

### 1.5 SanYuan Prior Module（先验模块）
将三远法写为软先验正则：
- 高远：纵向抬升与远深耦合趋势
- 深远：层层递进的深度分布
- 平远：横向展开与缓变深度

**只参与约束，不作为主输出标签。**

---

## 2. 监督与损失函数
总损失：

`L = λ_proj L_proj + λ_depth L_depth_rank + λ_occ L_occ + λ_rel L_rel + λ_prior L_sanyuan`

### 2.1 `L_proj`（重投影一致）
- 投影 mask 与 GT mask 的 BCE + Dice / IoU Loss
- 约束：3D 布局在 2D 观测上可解释

### 2.2 `L_depth_rank`（相对深度排序）
- 训练对象对 `(i,j)` 的前后关系（ranking / logistic loss）
- 数据来源：标注规则、人工弱标签、遮挡推断

### 2.3 `L_occ`（遮挡一致）
- 预测遮挡关系与 2D 轮廓重叠一致
- 可用二分类或多级遮挡标签

### 2.4 `L_rel`（关系图一致）
- 与先验空间关系图（邻接、层级）对齐
- 用于提升全局布局稳定性

### 2.5 `L_sanyuan`（三远先验）
- 基于统计分布（对象在 y/z 的分布趋势）构建 KL / MSE 正则
- 权重小于主任务损失，防止先验压制数据

---

## 3. 训练阶段设计
### Stage A：对象编码预热
- 冻结关系模块与3D头
- 先学稳对象特征（分类/掩码重建辅助）

### Stage B：3D布局主训练
- 打开关系模块 + 3D头 + 重投影
- 主优化目标：`L_proj + L_depth_rank + L_occ`

### Stage C：先验微调
- 引入/提升 `L_sanyuan`
- 控制先验权重，改善意境一致性与全局结构

---

## 4. 评估指标（围绕核心目标）
1. **Reproj-mIoU**：3D投影与2D掩码一致性
2. **Depth-Order Acc**：对象对深度排序准确率
3. **Occlusion F1**：遮挡关系预测准确率
4. **Rel-Consistency**：关系图一致性分数
5. （可选）**View-Consistency**：多视角渲染一致性

> 不再把“三远边界/分区预测准确率”作为主指标。

---

## 5. 消融实验设计
- Ablation-1：去掉 Relation Transformer
- Ablation-2：去掉 `L_depth_rank`
- Ablation-3：去掉 `L_sanyuan`
- Ablation-4：无重投影监督（仅直接回归）
- Full：完整模型

目标：验证每个模块对“3D空间位置重建”的贡献。

---

## 6. 与 Plan B（MIDI迁移）衔接
Plan A 输出对象级 3D 布局后，可作为 Plan B 的结构条件：
- 以对象深度与关系图作为条件 token
- 在风格迁移阶段保持空间结构不塌陷
