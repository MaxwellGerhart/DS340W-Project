# dev

Development material. Nothing here is needed for the steps in the main [README](../README.md).

| Path | What it is |
|---|---|
| `REPRODUCTION.md` | Full write-up of the reproduction: results, and the 13 problems found in the published code. |
| `README_upstream.md` | The README this repository had before the cleanup (the OpenSTARLab Event package's own). |
| `results/` | Older and exploratory training runs: the learning-rate, sequence-length and positional-encoding experiments behind `REPRODUCTION.md`, and earlier demo runs. |
| `configs/` | Configs for those experiments. |
| `symmetry/` | Training logs of the symmetry experiment, and the script that launched all its runs. |
| `probe_equivariance.py`, `summarize_symmetry.py` | Earlier versions of what `reproduce_results.py` now does in one step. |
| `datasets/` | The Wyscout match files split 70/20/10, with the scripts that split and recombine them. |

The two scripts import `symmetry.py` from the repository root. To run one, give Python the root as its import path from the project folder:

```
set PYTHONPATH=.
.venv\Scripts\python dev\probe_equivariance.py --run-dir results/full_2epoch/out/train/20260904_102542
```

In PowerShell the first line is `$env:PYTHONPATH = "."`.

`datasets/combined/` is not stored in Git. Rebuild it with `python combine_matches.py . combined` from inside `datasets/`.
