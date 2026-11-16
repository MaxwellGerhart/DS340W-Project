"""
Reproduce the reported results from the trained models saved in results/.

Nothing is trained here. Every checkpoint is loaded and scored again on the
held-out Wyscout test split (sample_data/test.csv.gz: 183 matches, 267,732
events), so the numbers printed are computed on this machine, not read back
from a file. Three parts:

    1. Event prediction     results/full_2epoch, against the row the OpenSTARLab
                            paper publishes for NMSTPP on Wyscout (its Table 6)
    2. Symmetry probe       the same model on the mirror image of the test set,
                            its equivariance gap, and test-time frame averaging
    3. Symmetry experiment  results/symmetry: baseline / augment / frameavg,
                            mean +- sd over seeds

It also reports how far what it computed is from the numbers saved when the
models were trained.

Usage:
    python reproduce_results.py
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from event.sports.soccer.utils.UEID_preprocessing import UEID_preprocessing
from run_training import apply_positional_encoding_fix
from symmetry import FastUEID, FrameAveraged, Mirror, equivariance_gap, package_metrics
from train_symmetry import build_model

# NMSTPP on Wyscout, as published in Table 6 of the OpenSTARLab paper.
PAPER = {"ACC_action": 0.67, "F1_action": 0.17, "MAE_deltaT": 3.34, "MAE_x": 6.94, "MAE_y": 15.08}
NAMES = {"ACC_action": "Action accuracy", "F1_action": "Action F1", "MAE_deltaT": "Time MAE (s)",
         "MAE_x": "X MAE", "MAE_y": "Y MAE"}
SHOWN = ["ACC_action", "F1_action", "F1_global", "MAE_deltaT", "MAE_x", "MAE_y"]
GAPS = ["mirrored_MAE_y", "y_gap_m", "x_gap_m", "action_flip"]
ARM_ORDER = ["baseline", "baseline + FA at inference", "augment", "augment + FA at inference", "frameavg"]


def load_run(run_dir, device):
    """The checkpoint, hyperparameters and normalisation constants of one run."""
    run_1 = os.path.join(run_dir, "run_1")
    model_path = max(glob.glob(os.path.join(run_1, "_model_*.pth")),
                     key=lambda p: int(re.search(r"_model_(\d+)\.pth$", p).group(1)))
    with open(os.path.join(run_1, "hyperparameters.json")) as f:
        config = json.load(f)
    with open(os.path.join(run_dir, "min_max_dict.json")) as f:
        mm = json.load(f)
    model = build_model(config, len(config["features"]), device)
    # map_location: the checkpoint must open on a machine without a GPU too.
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.eval(), config, mm["min_dict"], mm["max_dict"]


def max_diff(new, saved):
    """Largest absolute difference over the numbers two result dicts share."""
    if isinstance(new, dict):
        return max([max_diff(new[k], saved[k]) for k in new if k in saved] or [0.0])
    if isinstance(new, list):
        return max([max_diff(a, b) for a, b in zip(new, saved)] or [0.0])
    return abs(new - saved)


def table(rows, cols, width=13, label=24):
    """Print {row label: {column: value}} as a fixed-width table."""
    print(f"  {'':{label}s}" + "".join(f"{c:>{width}s}" for c in cols))
    for name, r in rows.items():
        print(f"  {name:{label}s}" + "".join(f"{r.get(c, '-'):>{width}}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-path", default="sample_data/test.csv.gz")
    ap.add_argument("--baseline-run", default="results/full_2epoch/out/train/20260904_102542")
    ap.add_argument("--symmetry-root", default="results/symmetry")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = pd.read_csv(args.test_path)
    print(f"  device    : {device}")
    print(f"  test split: {args.test_path}  ({raw['match_id'].nunique()} matches, {len(raw):,} events)")

    def test_split(config, min_dict, max_dict):
        # Normalised with the run's own training-set constants, as at training time.
        df, _, _ = UEID_preprocessing(raw.copy(), min_dict, max_dict)
        return FastUEID(df, config["seq_len"], config["features"], device)

    # ------------------------------------------------------------------ 1
    print("\n=== 1. Event prediction: NMSTPP on the Wyscout test split ===")
    print(f"  checkpoint: {args.baseline_run}\n")
    model, config, min_dict, max_dict = load_run(args.baseline_run, device)
    test = test_split(config, min_dict, max_dict)
    mirror = Mirror(config["features"], min_dict, max_dict, config["num_actions"])
    published = package_metrics(model, test, min_dict, max_dict, config)
    print(f"  {'metric':20s}{'this run':>12s}{'paper':>10s}")
    for m, name in NAMES.items():
        print(f"  {name:20s}{published[m]:12.4f}{PAPER[m]:10.2f}")

    # ------------------------------------------------------------------ 2
    print("\n=== 2. Symmetry probe: does that model respect the left-right mirror? ===\n")
    probe = {
        "as published": published,
        "mirrored test set": package_metrics(model, test, min_dict, max_dict, config, mirror_data=mirror),
        "test-time frame avg": package_metrics(FrameAveraged(model, mirror), test, min_dict, max_dict, config),
    }
    table({k: {m: f"{v[m]:.4f}" for m in SHOWN} for k, v in probe.items()}, SHOWN)
    gap = equivariance_gap(model, test, mirror, min_dict, max_dict)
    gap_fa = equivariance_gap(FrameAveraged(model, mirror), test, mirror, min_dict, max_dict)
    print()
    table({"equivariance gap, model": {k: f"{v:.4f}" for k, v in gap.items()},
           "         frame-averaged": {k: f"{v:.1e}" for k, v in gap_fa.items()}}, list(gap))

    saved = os.path.join(args.baseline_run, "symmetry_probe.json")
    if os.path.exists(saved):
        with open(saved) as f:
            old = json.load(f)
        diff = max(max_diff(probe, old["metrics"]), max_diff(gap, old["equivariance_gap"]))
        print(f"\n  largest difference from the saved results (parts 1-2): {diff:.1e}")

    # ------------------------------------------------------------------ 3
    print("\n=== 3. Symmetry experiment: baseline vs augmentation vs frame averaging ===\n")
    # Every arm was trained with the positional-encoding fix, so it is switched
    # on here; part 1 above scored its checkpoint with the published encoding.
    apply_positional_encoding_fix()
    rows, diff = [], 0.0
    for run_dir in sorted(glob.glob(os.path.join(args.symmetry_root, "*_seed*"))):
        model, config, min_dict, max_dict = load_run(run_dir, device)
        assert config["pe_fix"], f"{run_dir} was trained without the positional-encoding fix"
        arm, seed, bs = config["symmetry_arm"], config["seed"], config["batch_size"]
        test = test_split(config, min_dict, max_dict)
        mirror = Mirror(config["features"], min_dict, max_dict, config["num_actions"])
        averaged = FrameAveraged(model, mirror)
        scored = averaged if arm == "frameavg" else model      # the model as it was trained
        res = {"test": package_metrics(scored, test, min_dict, max_dict, config, batch_size=bs),
               "test_mirrored": package_metrics(scored, test, min_dict, max_dict, config, batch_size=bs,
                                                mirror_data=mirror),
               "equivariance_gap": equivariance_gap(scored, test, mirror, min_dict, max_dict)}
        rows.append({"arm": arm, "seed": seed, **{m: res["test"][m] for m in SHOWN},
                     "mirrored_MAE_y": res["test_mirrored"]["MAE_y"],
                     **{g: res["equivariance_gap"][g] for g in GAPS[1:]}})
        if arm != "frameavg":
            res["test_frameavg_at_inference"] = package_metrics(averaged, test, min_dict, max_dict,
                                                                config, batch_size=bs)
            rows.append({"arm": arm + " + FA at inference", "seed": seed,
                         **{m: res["test_frameavg_at_inference"][m] for m in SHOWN}})
        print(f"  scored {arm:9s} seed {seed}: acc {res['test']['ACC_action']:.4f} | "
              f"F1 {res['test']['F1_action']:.4f} | y {res['test']['MAE_y']:.3f} | "
              f"y gap {res['equivariance_gap']['y_gap_m']:.3f}")
        saved = os.path.join(run_dir, "results.json")
        if os.path.exists(saved):
            with open(saved) as f:
                diff = max(diff, max_diff(res, json.load(f)))

    df = pd.DataFrame(rows)
    arms = [a for a in ARM_ORDER if a in set(df.arm)]

    def cell(arm, col):
        v = df[df.arm == arm][col].dropna()
        return "-" if v.empty else f"{v.mean():.4f} +- {v.std(ddof=1) if len(v) > 1 else 0:.4f}"

    # One column per arm; "+FA" is the same model frame-averaged at inference only.
    short = {a: a.replace(" + FA at inference", "+FA") for a in arms}
    print(f"\n  mean +- sd over {df.seed.nunique()} seeds, on the test split:")
    table({c: {short[a]: cell(a, c) for a in arms} for c in SHOWN + GAPS}, list(short.values()),
          width=18, label=16)
    print("  (the last four rows measure symmetry: zero means exactly mirror-symmetric)")
    print(f"\n  largest difference from the saved results (part 3): {diff:.1e}")

    out = os.path.join("results", "reproduced_results.json")
    with open(out, "w") as f:
        json.dump({"event_prediction": {m: published[m] for m in NAMES}, "paper": PAPER,
                   "symmetry_probe": probe, "equivariance_gap": gap,
                   "equivariance_gap_frame_averaged": gap_fa, "symmetry_experiment": rows}, f, indent=2)
    print(f"\n  written: {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
