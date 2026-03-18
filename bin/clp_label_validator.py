from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/CLP_dataset/clp2k")
LABEL_ROOT = ROOT / "labels_v2"


def validate_file(path: Path) -> list[str]:
    errs: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"JSON parse error: {e}"]

    for k in ["image", "dominant_mode", "mode_mix", "regions", "pair_relations", "notes"]:
        if k not in data:
            errs.append(f"missing key: {k}")

    mode = data.get("dominant_mode", "")
    allowed_mode = {"", "high", "deep", "level"}
    if mode not in allowed_mode:
        errs.append(f"invalid dominant_mode: {mode}")

    mix = data.get("mode_mix", {})
    if not isinstance(mix, dict):
        errs.append("mode_mix must be an object")
    else:
        keys = ["high", "deep", "level"]
        if all(k in mix for k in keys):
            vals = []
            for k in keys:
                v = mix[k]
                if not isinstance(v, (int, float)):
                    errs.append(f"mode_mix.{k} must be number")
                    continue
                if v < 0 or v > 1:
                    errs.append(f"mode_mix.{k} out of [0,1]: {v}")
                vals.append(float(v))
            if len(vals) == 3:
                s = sum(vals)
                if abs(s - 1.0) > 0.01 and s != 0.0:
                    errs.append(f"mode_mix sum should be 1.0±0.01 (or all-zero draft), got {s:.4f}")
                if mode in {"high", "deep", "level"} and s > 0:
                    argmax = keys[max(range(3), key=lambda i: vals[i])]
                    if argmax != mode:
                        errs.append(f"dominant_mode({mode}) != argmax(mode_mix)({argmax})")
        else:
            errs.append("mode_mix keys must include high/deep/level")

    if not isinstance(data.get("regions", []), list):
        errs.append("regions must be list")
    if not isinstance(data.get("pair_relations", []), list):
        errs.append("pair_relations must be list")

    return errs


def main() -> None:
    files = sorted(LABEL_ROOT.glob("**/*.json"))
    if not files:
        print("No files found under", LABEL_ROOT)
        return

    bad = 0
    for f in files:
        errs = validate_file(f)
        if errs:
            bad += 1
            print(f"\n[ERROR] {f}")
            for e in errs:
                print(" -", e)

    print(f"\nchecked={len(files)} bad={bad} good={len(files)-bad}")


if __name__ == "__main__":
    main()
