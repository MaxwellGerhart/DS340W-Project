"""
Does a trained event model respect the pitch's left-right symmetry?

Loads a checkpoint written by the package's train() and, on the test split:

  1. scores it as published                    (should match REPRODUCTION.md Table 6)
  2. scores it on the mirror image of the test set (inputs and targets mirrored)
  3. measures the equivariance gap: f(M x) against M f(x)
  4. scores it with test-time frame averaging -- the TacticAI construction
     applied to an already-trained model, no retraining

A model that had learnt the symmetry would score the same in 1 and 2 and show a
zero gap in 3.

Usage:
    python probe_equivariance.py --run-dir results/full_2epoch/out/train/20260904_102542
    python probe_equivariance.py --run-dir <run> --check-loader   # also verify FastUEID == UEID_loader
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from event.sports.soccer.inference.UEID_inference import load_model
from event.sports.soccer.utils.UEID_preprocessing import UEID_preprocessing
from symmetry import FastUEID, FrameAveraged, Mirror, equivariance_gap, package_metrics

SHOWN = ["ACC_action", "F1_action", "F1_global", "MAE_deltaT", "MAE_x", "MAE_y"]


def check_loader(df, seq_len, features, device, n=2000):
    """Assert FastUEID reproduces UEID_loader sample-for-sample on the first n rows."""
    from event.sports.soccer.dataloaders.UEID_loader import UEID_loader
    part = df.iloc[:n].reset_index(drop=True)
    slow = UEID_loader(part, seq_len, features)
    fast = FastUEID(part, seq_len, features, device)
    assert list(slow.valid_indices) == list(fast.valid_indices), "valid_indices differ"
    X, Y, E = next(fast.batches(len(fast)))
    for j in range(len(slow)):
        xs, ys, es = slow[j]
        assert es == E[j].item()
        assert torch.equal(xs, X[j].cpu()) and torch.equal(ys, Y[j].cpu()), f"sample {j} differs"
    print(f"  loader check: FastUEID == UEID_loader on {len(slow)} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--test-path", default="sample_data/test.csv.gz")
    ap.add_argument("--out", default=None, help="json file for the results (default: <run-dir>/symmetry_probe.json)")
    ap.add_argument("--check-loader", action="store_true")
    args = ap.parse_args()

    run_1 = os.path.join(args.run_dir, "run_1")
    model_path = max(glob.glob(os.path.join(run_1, "_model_*.pth")),
                     key=lambda p: int(re.search(r"_model_(\d+)\.pth$", p).group(1)))
    cfg_path = os.path.join(run_1, "hyperparameters.json")
    with open(cfg_path) as f:
        config = json.load(f)
    with open(os.path.join(args.run_dir, "min_max_dict.json")) as f:
        mm = json.load(f)
    min_dict, max_dict = mm["min_dict"], mm["max_dict"]
    features = config["features"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, _, _ = UEID_preprocessing(pd.read_csv(args.test_path), min_dict, max_dict)
    if args.check_loader:
        check_loader(df, config["seq_len"], features, device)
    data = FastUEID(df, config["seq_len"], features, device)

    model = load_model(config["model"], model_path, cfg_path).to(device)
    mirror = Mirror(features, min_dict, max_dict, config["num_actions"])
    print(f"  checkpoint : {model_path}")
    print(f"  test events: {len(data):,}   mirrored features: "
          f"{[features[i] for i in mirror.in_idx]}")

    rows = {
        "as published": package_metrics(model, data, min_dict, max_dict, config),
        "mirrored test set": package_metrics(model, data, min_dict, max_dict, config, mirror_data=mirror),
        "test-time frame avg": package_metrics(FrameAveraged(model, mirror), data, min_dict, max_dict, config),
    }
    gap = equivariance_gap(model, data, mirror, min_dict, max_dict)
    gap_fa = equivariance_gap(FrameAveraged(model, mirror), data, mirror, min_dict, max_dict)

    print()
    print(f"  {'':22s}" + "".join(f"{m:>12s}" for m in SHOWN))
    for name, r in rows.items():
        print(f"  {name:22s}" + "".join(f"{r[m]:12.4f}" for m in SHOWN))
    print()
    print("  equivariance gap        " + "".join(f"{k:>13s}" for k in gap))
    print("  model                   " + "".join(f"{v:13.4f}" for v in gap.values()))
    print("  frame-averaged model    " + "".join(f"{v:13.2e}" for v in gap_fa.values()))

    out = args.out or os.path.join(args.run_dir, "symmetry_probe.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": model_path, "metrics": rows,
                   "equivariance_gap": gap, "equivariance_gap_frame_averaged": gap_fa},
                  f, indent=2)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
