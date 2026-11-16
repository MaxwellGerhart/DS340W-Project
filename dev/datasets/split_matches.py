#!/usr/bin/env python3
import argparse
import json
import random
import zipfile
from pathlib import Path


def split_counts(n: int) -> tuple[int, int, int]:
    """Allocate 70/20/10 using largest remainders, in train/test/validation order."""
    weights = (0.7, 0.2, 0.1)
    raw = [n * weight for weight in weights]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)[: n - sum(counts)]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Wyscout match JSON files reproducibly.")
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subsets = ("training", "test", "validation")
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals = {subset: 0 for subset in subsets}
    manifest = {"seed": args.seed, "ratios": {"training": 0.7, "test": 0.2, "validation": 0.1}, "files": {}}

    with zipfile.ZipFile(args.source_zip) as archive:
        filenames = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for filename in filenames:
            records = json.loads(archive.read(filename))
            indices = list(range(len(records)))
            rng.shuffle(indices)
            train_n, test_n, validation_n = split_counts(len(records))
            boundaries = (train_n, train_n + test_n, train_n + test_n + validation_n)
            index_groups = (indices[:boundaries[0]], indices[boundaries[0]:boundaries[1]], indices[boundaries[1]:boundaries[2]])
            counts = {}
            for subset, selected in zip(subsets, index_groups):
                subset_dir = args.output_dir / subset
                subset_dir.mkdir(exist_ok=True)
                output_records = [records[i] for i in selected]
                (subset_dir / filename).write_text(json.dumps(output_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                counts[subset] = len(output_records)
                totals[subset] += len(output_records)
            manifest["files"][filename] = {"source": len(records), **counts}

    manifest["totals"] = {"source": sum(totals.values()), **totals}
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
