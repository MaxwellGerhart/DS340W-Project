#!/usr/bin/env python3
"""Recombine the training/test/validation match files into one file per competition.

Inverts split_matches.py: replays its seeded shuffle so every record returns to
its original position in the source archive.
"""
import argparse
import json
import random
from pathlib import Path

from split_matches import split_counts

# Flattened filenames: "-1" = training, "-2" = test, unsuffixed = validation.
SUBSET_PATTERNS = {
    "training": ("matches_{}-1.json", "matches_{}-1 (1).json"),
    "test": ("matches_{}-2.json",),
    "validation": ("matches_{}.json",),
}


def find(src: Path, patterns, name: str) -> Path:
    for pattern in patterns:
        path = src / pattern.format(name)
        if path.exists():
            return path
    raise FileNotFoundError(f"no file matching {patterns} for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recombine split Wyscout match JSON files.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.source_dir / "split_manifest.json").read_text(encoding="utf-8"))
    rng = random.Random(manifest["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Same file order as split_matches.py, so the shared RNG replays identically.
    for filename in sorted(manifest["files"]):
        name = filename[len("matches_"):-len(".json")]
        parts = {
            subset: json.loads(find(args.source_dir, patterns, name).read_text(encoding="utf-8"))
            for subset, patterns in SUBSET_PATTERNS.items()
        }
        n = manifest["files"][filename]["source"]
        indices = list(range(n))
        rng.shuffle(indices)
        train_n, test_n, _ = split_counts(n)
        ordered = parts["training"] + parts["test"] + parts["validation"]
        assert len(ordered) == n, f"{filename}: expected {n} records, got {len(ordered)}"

        records = [None] * n
        for position, record in zip(indices, ordered):
            records[position] = record
        assert len({r["wyId"] for r in records}) == n, f"{filename}: duplicate wyId"

        (args.output_dir / filename).write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{filename}: {n} records")


if __name__ == "__main__":
    main()
