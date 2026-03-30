# Plan A 训练手册（v1）

## 1. 数据准备
1. 确认图片与标注目录：
   - `data/CLP_dataset/clp2k/images`
   - `data/CLP_dataset/clp2k/labels_v2`
2. 标注 JSON 至少包含：
   - `image_path`
   - `anchors.near_mid_boundary`
   - `anchors.mid_far_boundary`
3. 建议先抽样检查 100 条，确认点序与图像方向一致。

## 2. 环境安装
```bash
cd projects/SanYuan-LayoutNet
pip install -r requirements.txt
```

## 3. 配置说明
主配置文件：`configs/planA_base.yaml`
关键参数：
- `num_samples_per_curve`：曲线采样点数
- `batch_size`：显存不足时优先调小
- `w_order`：边界顺序约束强度

## 4. 训练命令
```bash
python scripts/train.py --config configs/planA_base.yaml
```

## 5. 验证与可视化
每个 epoch 输出：
- `train_loss`, `val_loss`
- `curve_error`
- `mIoU`
并在 `outputs/.../vis/` 保存预测叠加图。

## 6. 常见问题
1) **loss 不收敛**
- 检查标注点顺序是否反向
- 将 `lr` 降到 `1e-4`

2) **曲线抖动明显**
- 提高 `w_curve_smooth`
- 增大采样点数并启用曲线后平滑

3) **near/mid/far 混淆**
- 提高 `w_order`
- 在数据增强中减少强几何变换

## 7. 推荐训练流程
- Stage 1：仅训练曲线头（10~20 epoch）
- Stage 2：联合训练曲线+分区头（60+ epoch）
- Stage 3：固定 backbone，微调分区头（10 epoch）

## 8. 实验记录模板
每次实验记录：
- 配置差异
- 关键指标
- 可视化结论
- 下一步改动
