# Plan A 网络架构设计（SanYuan-LayoutNet）

## 1. 设计目标
在单幅山水图像上学习三远空间结构，输出：
- 边界1：`near_mid_boundary`
- 边界2：`mid_far_boundary`
- 三远分区概率图：`P_near, P_mid, P_far`

## 2. 总体结构
`Image Encoder -> Dual Curve Head + Region Head`

1) **Image Encoder（共享骨干）**
- Backbone: ResNet34/FPN（首版先用 ResNet34）
- 输出多尺度特征：`C3,C4,C5`

2) **Dual Curve Head（双边界曲线头）**
- 每条边界预测 K 个有序控制点（默认 K=64）
- 输出维度：`(K,2)`，坐标归一化到 `[0,1]`
- 解码方式：MLP + positional embedding

3) **Region Head（分区头）**
- 基于共享特征 + 曲线引导特征
- 预测 3 类分割图（near/mid/far）
- 输出大小与输入同尺度（或 1/4 上采样）

## 3. 标注与监督映射
- JSON 中读取 anchors：
  - `anchors.near_mid_boundary`
  - `anchors.mid_far_boundary`
- 折线点重采样为固定 K 点，形成回归目标
- 通过两条曲线 rasterize 生成 near/mid/far 软标签（训练分区头）

## 4. 损失函数
总损失：
`L = λ1*L_curve + λ2*L_smooth + λ3*L_order + λ4*L_mask_ce + λ5*L_mask_dice`

- `L_curve`：预测点与GT点 L1/L2
- `L_smooth`：二阶差分平滑约束，避免锯齿
- `L_order`：约束 near_mid 在 mid_far 之前（纵向/深度顺序）
- `L_mask_ce`：三类分区交叉熵
- `L_mask_dice`：缓解类别不均衡

## 5. 训练与推理接口
- 训练输入：`image, gt_curve1, gt_curve2, gt_mask`
- 推理输出：`pred_curve1, pred_curve2, pred_mask`
- 后处理：曲线平滑 + 可视化叠加

## 6. 指标
- 曲线误差：平均点距（pixel / normalized）
- 区域指标：mIoU / Dice
- 结构一致性：边界顺序违反率（越低越好）

## 7. 与后续模块衔接
该网络输出可直接作为：
- Plan B 的空间先验条件
- 扩散模型条件控制图（layout control map）
- 3D重建阶段的层次先验（near-mid-far depth prior）
