from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def pick_polyline(ax, title: str):
    ax.set_title(title)
    print(f"\n{title}")
    print("- 左键添加点")
    print("- 键盘 u: 撤销上一个点")
    print("- 键盘 r: 清空并重画当前线")
    print("- 回车: 完成当前线")

    points: list[list[int]] = []
    line, = ax.plot([], [], "o-", color="yellow", linewidth=2)
    hint = ax.text(
        0.01,
        0.01,
        "左键加点 | u撤销 | r重画 | Enter完成",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.4, pad=4),
    )

    done = {"value": False}

    def redraw():
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            line.set_data(xs, ys)
        else:
            line.set_data([], [])
        ax.figure.canvas.draw_idle()

    def on_click(event):
        if done["value"]:
            return
        if event.inaxes != ax:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            points.append([int(round(event.xdata)), int(round(event.ydata))])
            redraw()

    def on_key(event):
        if done["value"]:
            return
        key = (event.key or "").lower()
        if key in ["enter", "return"]:
            done["value"] = True
        elif key == "u":
            if points:
                points.pop()
                redraw()
        elif key == "r":
            points.clear()
            redraw()

    cid_click = ax.figure.canvas.mpl_connect("button_press_event", on_click)
    cid_key = ax.figure.canvas.mpl_connect("key_press_event", on_key)

    while not done["value"]:
        plt.pause(0.05)

    ax.figure.canvas.mpl_disconnect(cid_click)
    ax.figure.canvas.mpl_disconnect(cid_key)
    hint.remove()

    return points


def find_overlay_for_json(jp: Path, overlay_dir: Path) -> Path | None:
    exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    for ext in exts:
        p = overlay_dir / f"{jp.stem}{ext}"
        if p.exists():
            return p
    return None


def process_one_json(jp: Path, overlay: str = "", force: bool = False) -> bool:
    data = json.loads(jp.read_text(encoding="utf-8"))

    anchors = data.get("anchors", {})
    has_near_mid = bool(anchors.get("near_mid_boundary"))
    has_mid_far = bool(anchors.get("mid_far_boundary"))
    if (not force) and has_near_mid and has_mid_far:
        print(f"[SKIP] anchors 已存在: {jp}")
        return False

    img_path = Path(data["image_path"])

    if overlay:
        overlay_path = Path(overlay)
        if overlay_path.is_dir():
            found = find_overlay_for_json(jp, overlay_path)
            if found is None:
                print(f"[SKIP] overlay_dir 中找不到同名图: {jp.name}")
                return False
            show_path = found
        else:
            show_path = overlay_path
    else:
        show_path = img_path

    if not show_path.exists():
        print(f"[SKIP] image not found: {show_path}")
        return False

    img = np.array(Image.open(show_path).convert("RGB"))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("on")

    print(f"\n=== 标注文件: {jp} ===")
    print("提示：回车可结束当前边界（可为空）")

    near_mid = pick_polyline(ax, "Step1: 标注 near_mid_boundary")
    if near_mid:
        xs = [p[0] for p in near_mid]
        ys = [p[1] for p in near_mid]
        ax.plot(xs, ys, "y-o", linewidth=2)
        plt.draw()

    mid_far = pick_polyline(ax, "Step2: 标注 mid_far_boundary")
    if mid_far:
        xs = [p[0] for p in mid_far]
        ys = [p[1] for p in mid_far]
        ax.plot(xs, ys, "c-o", linewidth=2)
        plt.draw()

    data.setdefault("anchors", {})
    data["anchors"]["near_mid_boundary"] = near_mid
    data["anchors"]["mid_far_boundary"] = mid_far

    jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved anchors -> {jp}")

    plt.title("Saved. close window to continue")
    plt.show()
    return True


def collect_jsons(json_file: str, json_dir: str) -> list[Path]:
    if json_file:
        return [Path(json_file)]

    root = Path(json_dir)
    if not root.exists():
        raise FileNotFoundError(f"json_dir not found: {root}")

    return sorted(root.rglob("*.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="", help="单个 label json path")
    parser.add_argument("--json-dir", default="", help="批量模式：json 文件夹（会递归处理 *.json）")
    parser.add_argument("--overlay", default="", help="可选：单张叠加图路径，或叠加图文件夹")
    parser.add_argument("--force", action="store_true", help="即使已有 anchors 也强制重标")
    args = parser.parse_args()

    if (not args.json) and (not args.json_dir):
        raise ValueError("必须提供 --json 或 --json-dir")
    if args.json and args.json_dir:
        raise ValueError("--json 和 --json-dir 只能二选一")

    json_files = collect_jsons(args.json, args.json_dir)
    if not json_files:
        print("没有找到任何 json 文件")
        return

    print(f"待处理 json 数量: {len(json_files)}")

    done = 0
    skipped = 0
    for i, jp in enumerate(json_files, start=1):
        print(f"\n[{i}/{len(json_files)}] {jp}")
        try:
            changed = process_one_json(jp, overlay=args.overlay, force=args.force)
            if changed:
                done += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"[ERR] {jp}: {e}")

    print("\n==== 完成 ====")
    print(f"成功写入: {done}")
    print(f"跳过/失败: {skipped}")


if __name__ == "__main__":
    main()
