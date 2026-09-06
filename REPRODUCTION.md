# Reproduction Notes — DS340W Phase 1

Reproduction of the event-modeling results from **OpenSTARLab: open approach for
spatio-temporal agent data analysis in soccer** (Complex & Intelligent Systems,
2025, DOI [10.1007/s40747-025-01965-y](https://doi.org/10.1007/s40747-025-01965-y)).

Model reproduced: **NMSTPP** on the Wyscout dataset.

The model code in `event/` is the authors' code, unmodified. The files added for
this reproduction are `prepare_data.py`, `make_demo_subset.py`, `run_training.py`,
and `configs/`.

---

## 1. Environment

The package requires **Python 3.12 or older** — see issue 1 below. Built with:

```powershell
py -3.12 -m venv C:\Users\maxwe\venvs\ds340w312
C:\Users\maxwe\venvs\ds340w312\Scripts\Activate.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install openstarlab-preprocessing pyyaml
pip install -e .
```

Verified stack:

| | |
|---|---|
| Python | 3.12.10 |
| torch | 2.14.0+cu126 |
| numpy | 1.26.4 (pinned <2.0 by openstarlab-preprocessing) |
| pandas | 3.0.5 |
| GPU | NVIDIA RTX 4060 Laptop, CUDA available |

## 2. Data

The Wyscout public dataset (Pappalardo et al. 2019, CC BY 4.0), which is
reference [39] of the parent paper's data availability statement.

```powershell
python prepare_data.py --data-dir C:/Users/maxwe/ds340w-data
```

Downloads events/matches, keeps the five leagues (England, France, Germany,
Italy, Spain), converts to UIED format and writes an 80/10/10 split by match
within each competition:

```
data.csv    2,684,396 rows
train.csv   2,148,737 rows (1461 matches)
valid.csv     267,927 rows ( 182 matches)
test.csv      267,732 rows ( 183 matches)
```

## 3. Training

```powershell
python run_training.py --config configs/train_NMSTPP_full.yaml
```

`configs/train_NMSTPP_demo.yaml` is the same configuration on a 150-match subset
with 2 epochs, used for the screen recording. Both use `test: False`, so neither
uses the package's truncated debug mode.

## 4. Results

Validation metrics after 2 epochs on the full five-league dataset, against the
values published in the parent paper:

| Metric | This reproduction | Paper (NMSTPP, Wyscout) |
|---|---|---|
| Action Accuracy ↑ | 0.666 | 0.67 |
| Action F1 ↑ | 0.166 | 0.17 |
| Time MAE ↓ | 3.28 | 3.34 |
| X MAE ↓ | 6.99 | 6.94 |
| Y MAE ↓ | 15.02 | 15.08 |

All five predictive metrics reproduce within noise.

Model complexity does not match: this run reports 168,707 parameters and 26.8M
FLOPs, against the paper's 121K and 296M. The predictive metrics agreeing while
the complexity figures differ points to a configuration difference — the shipped
yaml uses 15 input features, and FLOPs also scale with `seq_len`.

Artifacts are under `results/`: `_model_N.pth`, `_loss.csv` (per-epoch metrics),
`hyperparameters.json`, `_model_stats.txt`.

## 5. Issues found in the published code

Five problems had to be worked around. None are in the model code; all are in the
path from a fresh clone to training data.

1. **`openstarlab-preprocessing` pins `numpy<2.0`.** The newest conforming numpy
   is 1.26.4, which ships no Python 3.13 wheel, so pip falls back to a source
   build requiring MSVC. Effectively caps the project at Python 3.12.

2. **The dataset download URL is dead.** The authors' `example/NMSTPP_data.py`
   uses `https://figshare.com/ndownloader/files/<id>/<name>`, which returns
   **403 Forbidden**. `https://ndownloader.figshare.com/files/<id>` still works.

3. **A spoofed User-Agent silently produces empty downloads.** The same script
   sends a Chrome UA; figshare answers browser-like clients with an empty
   `202 Accepted` instead of the `302` redirect to S3. `raise_for_status()`
   passes, a 0-byte file is written, and the failure only surfaces later as
   `BadZipFile: File is not a zip file`.

4. **The example script generates the wrong schema.** It calls
   `preprocess_method="NMSTPP"`, which emits a zone-based format
   (`zone`, `zone_s`, `zone_dist2goal`, `time_diff`, …). Every training config in
   the Event package instead requires `action`, `delta_T`, `start_x`, `start_y`,
   `Period`, `success`, `dist2goal` and others, which only `preprocess_method="UIED"`
   produces. `UIED` also drops the `comp` column that the script's own splitting
   logic depends on, so the competition label must be re-attached.

5. **Preprocessing is single-core.** `preprocessing()` dispatches per-match work
   to a `ThreadPoolExecutor`; the pandas work is GIL-bound, so it runs
   effectively serial. Running one process per league cut the five-league
   preprocessing from **10h43m to 22 minutes** on a 12-core machine.

A sixth, performance only: `UEID_loader.__init__` validates sequence indices in a
pure-Python loop over every row with four scalar `.loc` lookups each. On the
2.1M-row training set this costs ~4m45s of single-threaded startup before any
GPU work begins, on every run. The operation is vectorizable.
