# CLP 数据集标注方案 v1（面向三远法空间意境）

## 1. 数据集现状扫描（已完成）
- 路径：`data/CLP_dataset/clp2k/`
- 图像：`2210` 张 `.jpg`
- 语义标注：`2207` 张 `.png`
- 现有语义类别：`14` 类 + 背景 `0`（共 15 个像素 ID）

### 1.1 已发现的数据完整性问题
缺少标注掩码：
- `images/training/smithsonian_128.jpg`
- `images/validation/Princeton_82.jpg`
- `images/validation/where-truth-lies_11.jpg`

建议：先补齐这 3 张掩码，避免训练/评估脚本报错。

### 1.2 现有类别映射（来自 `autosave_label.txt`）
| id | class |
|---|---|
| 0 | background |
| 1 | river |
| 2 | mountain |
| 3 | trees |
| 4 | clouds |
| 5 | building |
| 6 | animals |
| 7 | birds |
| 8 | boat |
| 9 | person |
| 10 | sky |
| 11 | bridge |
| 12 | waterfall |
| 13 | rock |
| 14 | grass |

---

## 2. 标注目标（新增层）
在不破坏 CLP 原语义分割前提下，增加“空间意境标签层”：

1) 全局三远法标签（图像级）
- `dominant_mode`: `high | deep | level`
- `mode_mix`: `{high, deep, level}`，和为 `1.0`

2) 对象级空间属性（实例/语义区域级）
- `depth_bin`: `near | mid | far`
- `importance`: `main | secondary | minor`
- `region_role`: `subject | transition | background`

3) 关系标签（可选增强）
- `pair_relations`: 前后/上下/遮挡
- `depth_chain`: 近→中→远序列

---

## 3. 推荐实施策略（两阶段）

### 阶段 A（快速可用，1-2 天）
仅加图像级标签：
- `dominant_mode`
- `mode_mix`

优点：成本最低、可立即用于“按三远法分组训练/评估”。

### 阶段 B（研究增强，3-7 天）
对每张图的主要区域（建议前 3~8 个区域）增加：
- `depth_bin`
- `importance`
- 关键 `pair_relations`

优点：能支持空间结构约束与关系损失设计。

---

## 4. 三远法判定标准（操作化）
- **高远 high**：主峰/山体垂向上冲，明显“仰观”趋势。
- **深远 deep**：前中后景层层推进，纵深通道显著。
- **平远 level**：横向铺展显著，视线平推，远山淡出。

标注规则：
1. 先选主导类型 `dominant_mode`。
2. 再给 `mode_mix`（例如 `0.6/0.3/0.1`）。
3. 若两位标注员主导类型不一致，进入复核。

---

## 5. 质检规则
- `mode_mix` 三项和必须为 `1.0 ± 0.01`
- `dominant_mode == argmax(mode_mix)`
- 对象级若标 `far`，一般不应同时标 `main`（特殊构图可放行并加备注）
- 抽检比例：每 200 张至少抽检 30 张

---

## 6. 交付文件建议
- 原语义 mask 继续沿用：`annotations/*.png`
- 新增 sidecar（同名 JSON）：`labels_v2/<split>/<image_stem>.json`

示例：`labels_v2/training/harvard_0.json`

```json
{
  "image": "harvard_0.jpg",
  "dominant_mode": "deep",
  "mode_mix": {"high": 0.2, "deep": 0.6, "level": 0.2},
  "regions": [
    {"semantic_id": 2, "class": "mountain", "depth_bin": "far", "importance": "main", "region_role": "background"},
    {"semantic_id": 1, "class": "river", "depth_bin": "near", "importance": "secondary", "region_role": "transition"}
  ],
  "pair_relations": [
    {"a_class": "river", "b_class": "mountain", "front_back": "a_front", "confidence": 0.85}
  ],
  "notes": "前景水路引导到远山，深远主导。"
}
```

---

## 7. 下一步（我可直接继续）
1. 生成批量待标注清单（training/validation 分开）
2. 生成 `labels_v2` 空模板文件（按图像名批量创建）
3. 提供一个自动校验脚本（检查字段、数值范围、逻辑一致性）
