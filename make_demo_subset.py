"""
Build a smaller train/valid pair for the screen recording.

The full dataset takes ~5 minutes to index and ~17 minutes per epoch, which is
too long to record. This keeps a subset of whole matches so the model sees the
exact same code path, just less of it.

Subsetting by whole match_id matters: UEID_loader validates that a sequence
starts and ends inside one match and one period, so slicing arbitrary rows
would silently drop or corrupt sequences at the boundaries.

Usage:
    python make_demo_subset.py --data-dir C:/Users/maxwe/ds340w-data
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
    parser.add_argument("--data-dir", default="C:/Users/maxwe/ds340w-data")
    parser.add_argument("--train-matches", type=int, default=60)
    parser.add_argument("--valid-matches", type=int, default=10)
    args = parser.parse_args()

    d = os.path.abspath(args.data_dir)
    subset(os.path.join(d, "train.csv"), os.path.join(d, "train_demo.csv"), args.train_matches)
    subset(os.path.join(d, "valid.csv"), os.path.join(d, "valid_demo.csv"), args.valid_matches)
    print("done")


if __name__ == "__main__":
    main()
