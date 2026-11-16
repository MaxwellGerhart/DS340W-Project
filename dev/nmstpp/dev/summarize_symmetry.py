"""
Aggregate the train_symmetry.py runs into one table: mean +- sd over seeds.

Usage:
    python summarize_symmetry.py [--root results/symmetry]
Writes <root>/summary.csv and <root>/summary.md.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

METRICS = ["ACC_action", "F1_action", "F1_global", "MAE_deltaT", "MAE_x", "MAE_y"]
GAPS = ["y_gap_m", "x_gap_m", "action_flip"]
ARM_ORDER = ["baseline", "baseline + FA at inference", "augment", "augment + FA at inference", "frameavg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/symmetry")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the table uses "±"; the Windows console default is cp1252

    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "*_seed*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        base = {"seed": r["seed"], "best_epoch": r["best_epoch"]}
        rows.append({**base, "arm": r["arm"], **{m: r["test"][m] for m in METRICS},
                     **{f"mirrored_{m}": r["test_mirrored"][m] for m in ["ACC_action", "MAE_y"]},
                     **{g: r["equivariance_gap"][g] for g in GAPS}})
        if "test_frameavg_at_inference" in r:
            rows.append({**base, "arm": r["arm"] + " + FA at inference",
                         **{m: r["test_frameavg_at_inference"][m] for m in METRICS}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.root, "summary.csv"), index=False)

    cols = METRICS + ["mirrored_MAE_y"] + GAPS + ["best_epoch"]
    lines = ["| arm | seeds | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for arm in [a for a in ARM_ORDER if a in set(df.arm)]:
        g = df[df.arm == arm]
        cells = []
        for c in cols:
            v = g[c].dropna() if c in g else pd.Series(dtype=float)
            cells.append("-" if v.empty else f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}")
        lines.append(f"| {arm} | {len(g)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)

    # Paired differences against the baseline, seed by seed.
    b = df[df.arm == "baseline"].set_index("seed")
    for arm in ["baseline + FA at inference", "augment", "frameavg"]:
        a = df[df.arm == arm].set_index("seed")
        seeds = sorted(set(a.index) & set(b.index))
        if not seeds:
            continue
        d = {m: (a.loc[seeds, m] - b.loc[seeds, m]) for m in METRICS}
        print(f"\n  {arm} - baseline (paired over seeds {seeds}):")
        print("  " + " | ".join(f"{m} {np.mean(v):+.4f} [{np.min(v):+.4f}, {np.max(v):+.4f}]" for m, v in d.items()))


if __name__ == "__main__":
    main()
