"""
Build a smaller train/valid pair for the screen recording.

The full dataset takes ~5 minutes to index and ~17 minutes per epoch, which is
too long to record. This keeps a subset of whole matches so the model sees the
exact same code path, just less of it.

Subsetting by whole match_id matters: UEID_loader validates that a sequence
starts and ends inside one match and one period, so slicing arbitrary rows
would silently drop or corrupt sequences at the boundaries.

Usage:
    python make_demo_subset.py

sample_data/ holds the train_demo.csv and valid_demo.csv this writes, gzipped.
"""

import argparse
import os

import pandas as pd


def subset(src, dst, n_matches):
    df = pd.read_csv(src)
    keep = df["match_id"].unique()[:n_matches]
    out = df[df["match_id"].isin(keep)]
    out.to_csv(dst, index=False)
    print(
        f"{os.path.basename(dst):16s} {len(out):>9,} rows "
        f"({out['match_id'].nunique()} matches, from {df['match_id'].nunique()})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--train-matches", type=int, default=60)
    parser.add_argument("--valid-matches", type=int, default=10)
    # Simulation is the expensive downstream stage: it runs up to max_iter
    # autoregressive steps for every possession, which costs roughly 8.5 hours
    # over the full 183-match test split. Fig 5 reports the shape of the
    # per-timestep decay, which is stable well below that, so the simulation
    # stage runs on a subset of whole matches.
    parser.add_argument("--test-matches", type=int, default=40)
    args = parser.parse_args()

    d = os.path.abspath(args.data_dir)
    subset(os.path.join(d, "train.csv"), os.path.join(d, "train_demo.csv"), args.train_matches)
    subset(os.path.join(d, "valid.csv"), os.path.join(d, "valid_demo.csv"), args.valid_matches)
    subset(os.path.join(d, "test.csv"), os.path.join(d, "test_sim.csv"), args.test_matches)
    print("done")


if __name__ == "__main__":
    main()
