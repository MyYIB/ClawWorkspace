# 2026-03-31 epoch_018 现场导出与小样本核验

## 目标
1. 用 `outputs/planA_v2_full_auto/epoch_018.pt` 现场跑 1 张图导出摆放 JSON。
2. 再跑 3~5 张图确认效果（本次共 5 张）。
3. 结果入库并推仓。

## 执行命令
```powershell
$ckpt='outputs/planA_v2_full_auto/epoch_018.pt'
$outDir='outputs/layout_exports_epoch018_usercheck'
$files=@(
  'data/CLP_dataset/clp2k/labels_v2_auto_occ/validation/Princeton_127.json',
  'data/CLP_dataset/clp2k/labels_v2_auto_occ/validation/Princeton_134.json',
  'data/CLP_dataset/clp2k/labels_v2_auto_occ/validation/harvard_12.json',
  'data/CLP_dataset/clp2k/labels_v2_auto_occ/validation/met_100.json',
  'data/CLP_dataset/clp2k/labels_v2_auto_occ/validation/smithsonian_118.json'
)
foreach($f in $files){
  $name=[System.IO.Path]::GetFileNameWithoutExtension($f)
  python projects/SanYuan-LayoutNet/scripts/export_layout_from_image.py `
    --input-json $f --ckpt $ckpt --output-json "$outDir/$name.layout.json"
}
```

## 导出文件
- `outputs/layout_exports_epoch018_usercheck/Princeton_127.layout.json`
- `outputs/layout_exports_epoch018_usercheck/Princeton_134.layout.json`
- `outputs/layout_exports_epoch018_usercheck/harvard_12.layout.json`
- `outputs/layout_exports_epoch018_usercheck/met_100.layout.json`
- `outputs/layout_exports_epoch018_usercheck/smithsonian_118.layout.json`

## 快速核验（用 depth_pairs 做粗对齐检查）
- harvard_12: 0/3 = **0.000**
- met_100: 2/25 = **0.080**
- Princeton_127: 3/16 = **0.188**
- Princeton_134: 3/11 = **0.273**
- smithsonian_118: 2/10 = **0.200**
- 均值：**0.148**

## 结论
- 5 张图均可稳定导出 3D 摆放 JSON（流程通）。
- 但在该 5 张样本上，`epoch_018` 的深度顺序与 `depth_pairs` 对齐偏弱（均值 0.148），说明当前模型在“遮挡/前后关系”层面仍需继续优化。
- 可继续沿 `w_occ` + 标注质量提升方向迭代，并加入验证集早停指标。
