# Reproduction Notes — DS340W Phase 1

> **Paths in this document predate the repository cleanup.** The dataset now lives in `data/`
> (not `C:/Users/maxwe/ds340w-data`), and `train_NMSTPP_paper.yaml`, the `_seq*`, `_lr_*`, `_pefix`,
> `demo` and `paper_seqlen40` runs, and `probe_equivariance.py` are under `dev/`. To run things, follow
> the main [README](../README.md).

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

## 3. Running the pipeline

All model, inference and application code is the authors', unmodified. The files
added by this reproduction are `prepare_data.py`, `make_demo_subset.py`,
`run_training.py`, `run_downstream.py` and `configs/`; the two runners are thin
wrappers over `Event_Model(...).train()` and `Event_Model(...).inference(...)`.
The authors ship no runnable data-acquisition or driver script -- their
`event/sports/soccer/examples/example.py` is a stub full of `path/to/...`
placeholders -- so this glue is what a fresh clone requires, not a
reimplementation.

### Training

```powershell
# the configuration that reproduces the paper's predictive metrics
python run_training.py --config configs/train_NMSTPP_reproduction.yaml

# the configuration faithful to Appendix A (seq_len 40) -- does not train, see issue 9
python run_training.py --config configs/train_NMSTPP_paper.yaml
```

| config | seq_len | features | hidden_dim | purpose |
|---|---|---|---|---|
| `train_NMSTPP_reproduction.yaml` | 1 | 13 | 256 | reproduces the paper's metrics and parameter count |
| `train_NMSTPP_paper.yaml` | 40 | 13 | 256 | faithful to Appendix A; reproduces the paper's complexity but collapses to the majority class |
| `train_NMSTPP_full.yaml` | 1 | 15 | 1024 | phase 1; the authors' shipped `models/model_yaml` yaml with paths filled in |
| `train_NMSTPP_demo.yaml` | 1 | 15 | 1024 | phase 1; 150-match subset, 2 epochs, for the screen recording |

All four set `test: False`, so none uses the package's truncated debug mode.

`--fix-positional-encoding` repairs issues 7 and 8 at runtime, from
`run_training.py`, leaving `event/` untouched on disk.

### Inference, simulation and applications

```powershell
python run_downstream.py --run-dir results/reproduction/out/train/<timestamp> --realign
```

Runs, in order, test-set inference (Table 6), possession simulation with greedy
selection over 26 timesteps (Fig. 5), and the Poss-Util and HPUS applications
(Figs. 6-7). `--realign` corrects the row-misalignment defect described in
issue 10 before the applications consume `inference.csv`; without it they read
shifted rows, which is the published behaviour.

`--matches-json` points at a Wyscout `matches_<Country>.json` and is what makes
the Poss-Util figures readable: it recovers team names and final league positions
(see issue 13) and restricts the plot to five teams spread across the table,
mirroring the paper's Fig. 6. `--team-ranks` chooses which positions, defaulting
to `1,5,10,15,20`. Without the flag every team in the split is drawn on one axis,
which on the five-league test split is 98 overlapping curves under a legend that
covers the plot area.

The Fig. 8 heat map is not reachable from NMSTPP. `application/heatmap.py`
requires `start_x_<i>_prob` and `start_y_<i>_prob` columns over 101 spatial bins,
which only `LEM_inference` and `FMS_inference` emit; NMSTPP regresses location as
two scalars. The paper generates Fig. 8 from LEM_3, consistent with this.

## 4. Results

### Event prediction (Table 6)

Metrics on the **held-out test split** (183 matches, 267,732 events), from the
checkpoint in `results/full_2epoch`, against the values published for NMSTPP on
Wyscout:

| Metric | This reproduction (test) | Paper (NMSTPP, Wyscout) |
|---|---|---|
| Action Accuracy | 0.6641 | 0.67 |
| Action F1 | 0.1658 | 0.17 |
| Time MAE | 3.300 | 3.34 |
| X MAE | 6.959 | 6.94 |
| Y MAE | 15.036 | 15.08 |

All five predictive metrics reproduce within rounding. Phase 1 reported the same
figures on the validation split (0.666 / 0.166 / 3.28 / 6.99 / 15.02); the test
split agrees, so the result is not an artifact of the split used for early
stopping.

The checkpoint is 2 epochs of the shipped `models/model_yaml` configuration, so it
reports 168,707 parameters and 26.8M FLOPs rather than the paper's 121K and 296M.
Section 6 shows that the parameter count is a feature-list difference and the FLOP
count a `seq_len` difference, and that no single configuration reproduces the
metrics and the complexity together.

### Simulation (Fig. 5)

Greedy selection, 26 timesteps, 10 test matches (14,424 possessions at t=0).
`downstream/simulation/timestep_eval.csv`:

| timestep | 0 | 5 | 10 | 15 | 20 | 24 |
|---|---|---|---|---|---|---|
| possessions remaining | 14,424 | 3,398 | 1,499 | 748 | 397 | 231 |
| Action Acc | 0.487 | 0.697 | 0.728 | 0.754 | 0.751 | 0.749 |
| Time MAE | 3.63 | 2.25 | 2.22 | 2.24 | 1.93 | 1.76 |
| X MAE | 16.5 | 25.1 | 28.6 | 29.1 | 30.7 | 31.4 |
| Y MAE | 32.4 | 32.0 | 31.1 | 29.4 | 29.9 | 28.7 |

Possession count decays as the paper describes, and spatial error grows with
horizon -- X MAE roughly doubles from 16.5 to 31.4 -- which matches the paper's
observation that lateral position is the hardest quantity to carry forward.

Action accuracy behaves differently: it *rises* from 0.49 to about 0.75 rather
than falling from 0.66 to 0.5. The comparison is not like-for-like -- the paper's
Fig. 5 is LEM_3 on StatsBomb, whereas this is NMSTPP on Wyscout -- and the rise is
explicable, since possessions surviving many timesteps are increasingly dominated
by `short_pass`, the majority class. The value at the final timestep (0.118 at
t=25) is an edge artifact on 203 possessions.

### Applications (Figs. 6-7)

Poss-Util and HPUS over all 55,084 test-split possessions, written to
`downstream/applications/`: `poss_util.png`, `poss_util_plus.png`, `HPUS.png`,
`HPUS_plus.png`. Poss-Util means -0.0271 with 9.7% of possessions positive;
HPUS means 0.879 with a maximum of 13.774. These are computed from realigned rows
(issue 10); the as-published figures differ materially, quantified there.

**Fig. 6 reproduces qualitatively.** The paper plots the sides finishing 1st, 5th,
10th, 15th and 20th, and reports that the champion "demonstrated significantly
higher variation in Poss-Util compared to other teams", the rest showing
"relatively similar distributions". Running the same comparison on the 2017/18
Premier League -- Manchester City (1), Chelsea (5), Newcastle United (10),
Brighton & Hove Albion (15), West Bromwich Albion (20), with the table
reconstructed from the Wyscout match results -- gives the same picture: Manchester
City's curve is markedly wider and flatter (peak density 0.90) than the other four
(1.55-2.00), which sit close together. In Poss-Util+ the top two separate from the
rest, Manchester City and Chelsea carrying both the highest means and the widest
spreads. This is a different league, a different season and a different model from
the paper's version of the figure, so the agreement is about the pattern, not the
values.

**Fig. 7 reproduces in form.** `HPUS.png` and `HPUS_plus.png` plot one match
(Brighton & Hove Albion vs Tottenham Hotspur) as alternating five-minute HPUS
peaks for the two sides, the same presentation as the paper's Valencia-Sevilla
figure.

The Fig. 8 heat map is not reachable from NMSTPP -- see section 3.

### The paper's own figures, on La Liga

The paper draws Figs. 6-8 from the StatsBomb 2023/24 La Liga dataset, a
commercial purchase it states could not be published. Real Madrid, Athletic Club,
Deportivo Alaves, Mallorca and Granada, and the Valencia-Sevilla 0-0, are all in
data that is not obtainable, so those exact figures cannot be recreated.

The nearest available substitute is La Liga from Wyscout, which the same public
download already provides: a different season (2017/18) and a different provider,
but the same competition and the same figure specification. Written to
`downstream/applications_laliga/`:

```powershell
python run_downstream.py --run-dir <run> --stage applications --realign `
  --matches-json C:/Users/maxwe/ds340w-data/matches/matches_Spain.json `
  --restrict-to-league --match-id 2565890 --app-dir applications_laliga
```

**Fig. 6.** Teams finishing 1st, 5th, 10th, 15th and 20th, the paper's own
selection rule, with the table reconstructed from the Wyscout results: Barcelona,
Villarreal, Girona, Levante, Malaga. Restricted to the 38 Spanish matches in the
test split (the paper's equivalent slice was roughly 76 matches).

| rank | team | possessions | mean Poss-Util | sd |
|---|---|---|---|---|
| 1 | Barcelona | 440 | -0.0942 | **0.3560** |
| 5 | Villarreal | 422 | -0.0354 | 0.2254 |
| 10 | Girona | 616 | -0.0263 | 0.2121 |
| 15 | Levante | 619 | -0.0227 | 0.1807 |
| 20 | Malaga | 574 | -0.0244 | 0.1755 |

The paper reports that the champion "demonstrated significantly higher variation
in Poss-Util compared to other teams", with the rest showing "relatively similar
distributions". That is what appears here, and it is monotonic in league position:
Barcelona's spread is 1.6x the next-widest and roughly double the bottom three,
which sit within 0.05 of each other. The paper states the claim qualitatively;
on this data it holds as an ordering.

**Fig. 7.** The paper plots a 0-0 draw and shows that one side converted potential
attacks into actual attacks far better than the other. The Valencia-Sevilla
fixture is not in this dataset's test split -- the two 2017/18 meetings finished
0-2 and 4-0 -- but the split does contain a 0-0 draw involving Valencia, against
Eibar (match 2565890), which gives the same setup:

| | possessions | HPUS total | attacking possessions |
|---|---|---|---|
| Valencia | 200 | **195.08** | 19 |
| Eibar | 200 | 134.78 | 14 |

Level on the scoreboard and level on possession count, Valencia carries 45% more
HPUS and converts more possessions into attacks, concentrated after the 60th
minute in `HPUS_plus.png`. This mirrors the paper's reading of its own figure,
that a 0-0 scoreline can conceal a clear difference in attacking performance --
reached here from different data, and coincidentally about the same club.

### Not attempted

- **StatsBomb half of Table 6**, and the other five models (MAJ, Seq2Event, LEM_1,
  LEM_3, FMS). The StatsBomb 2023/24 La Liga data is a commercial purchase the
  authors could not redistribute.
- **Table 7 / RLearn.** A separate package, requiring J-League tracking data
  purchased from Data Stadium Inc.
- **Optuna hyperparameter search** (100 trials per model; the paper reports 3-4
  days per model on an RTX 3090).
- **A converged training run.** Every result above comes from a 2-epoch
  checkpoint. A 50-epoch run with the corrected feature list and `hidden_dim: 256`
  was started -- it would additionally match the paper's 121K parameter count --
  but at roughly 24 minutes per epoch it did not fit the available window, and
  `train()` writes no checkpoint until training completes, so a partial run yields
  nothing. `configs/train_NMSTPP_reproduction.yaml` is ready to run it.

## 5. Issues found in the published code

Thirteen problems were found. Items 1-6 lie in the path from a fresh clone to
training data and were present in the first phase of this reproduction. Items
7-13 are in the model, inference and application code itself, and were found by
running the pipeline end to end. `event/` is never edited: where a correction was
needed to test a hypothesis it is applied at runtime from `run_training.py`
(`--fix-positional-encoding`) or post hoc from `run_downstream.py` (`--realign`).

### Packaging and data path

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
   (`zone`, `zone_s`, `zone_dist2goal`, `time_diff`, ...). Every training config in
   the Event package instead requires `action`, `delta_T`, `start_x`, `start_y`,
   `Period`, `success`, `dist2goal` and others, which only `preprocess_method="UIED"`
   produces. `UIED` also drops the `comp` column that the script's own splitting
   logic depends on, so the competition label must be re-attached.

5. **Preprocessing is single-core.** `preprocessing()` dispatches per-match work
   to a `ThreadPoolExecutor`; the pandas work is GIL-bound, so it runs
   effectively serial. Running one process per league cut the five-league
   preprocessing from **10h43m to 22 minutes** on a 12-core machine.

6. **`UEID_loader.__init__` validates sequence indices in a pure-Python loop**
   over every row with four scalar `.loc` lookups each. On the 2.1M-row training
   set this costs 5-6 minutes of single-threaded startup before any GPU work
   begins, on every run. The operation is vectorizable.

### Model, inference and application code

7. **`PositionalEncoding` indexes the batch axis instead of the sequence axis.**
   `event/sports/soccer/models/NMSTPP.py` builds the `pe` buffer sequence-first,
   shape `(max_len, 1, d_model)`, then slices it in `forward` as
   `self.pe[:x.size(0), :]`. The encoder is constructed with `batch_first=True`,
   so `x` is `(batch, seq, d_model)` and `x.size(0)` is the **batch size**. The
   positional vector a sample receives is therefore determined by its position in
   the batch and is broadcast identically across every timestep. Verified
   directly: for a `(256, 40, 21)` input the slice taken has shape `(256, 1, 21)`,
   the encoding is constant across all 40 timesteps of a sample, and differs
   between samples of the same batch. Time carries no positional signal at all.

8. **The positional encoding is added twice.** `PositionalEncoding.forward`
   returns `dropout(x + pe)`, but the caller computes
   `X_concatenate + self.positional_encoding(X_concatenate)`. The encoder
   therefore receives `2*X + pe` rather than `X + pe`. Confirmed by direct tensor
   comparison.

9. **Multi-step configurations do not train.** Only `seq_len: 1` escapes the
   majority-class plateau; every `seq_len >= 2` sits on it. Two epochs on the demo
   subset, best validation epoch:

   | seq_len | 1 | 2 | 5 | 10 | 20 | 40 |
   |---|---|---|---|---|---|---|
   | Action Acc | **0.6254** | 0.5731 | 0.5729 | 0.5707 | 0.5724 | 0.5734 |
   | Action F1 | **0.1428** | 0.0807 | 0.0807 | 0.0805 | 0.0807 | 0.0807 |
   | FLOPs | 7.96M | 14.5M | 34.5M | 68.6M | 140M | 296M |
   | params | 121,121 | 121,121 | 121,121 | 121,121 | 121,121 | 121,121 |

   This is a cliff at `seq_len: 2`, not a gradual dilution, and `seq_len: 1` is
   the one case in which issue 7 is harmless -- a one-element sequence has no
   order to encode.

   Ruled out as causes: learning rate (identical collapse at 0.01, 0.001 and
   0.0001, with validation accuracy and F1 agreeing to four decimals across all
   three); NaN or infinite feature values (none present in any of the 13 feature
   columns); and early stopping alone (on the full dataset the training loss sat
   at 6.745x for five consecutive epochs, flat to four decimals, over roughly 40k
   gradient steps before patience expired).

   The root cause was not isolated further. Correcting issues 7 and 8 does not
   rescue training within three epochs on the demo subset, so while the
   positional-encoding defect is the most plausible explanation -- it is exactly
   the kind of fault that would leave `seq_len: 1` untouched while breaking every
   longer sequence -- that link is not established here. A longer run with early
   stopping disabled, on both the shipped and corrected encodings, would settle
   whether the plateau is permanent or merely slow to escape; it was not run to
   completion for this report.

10. **`UEID_inference` attaches predictions to the wrong events.** `end_idx` is a
    positional index into the frame returned by `UEID_preprocessing`, which has
    already dropped every `game_over`/`period_over` row and called
    `reset_index(drop=True)`. The prediction is then written back with
    `data_raw.loc[end_idx]`, where `data_raw` still contains those rows. The two
    index spaces diverge by one position per dropped marker, so a prediction
    lands on an event 1..N rows too early, N being the number of markers before
    it. The defect therefore compounds along the file, and its severity scales
    with the size of the split:

    | split | rows | misattached | max shift |
    |---|---|---|---|
    | 10 matches | 14,048 | 13,316 (94.8%) | 19 events |
    | full test split, 183 matches | 267,366 | **266,546 (99.7%)** | **365 events** |

    The reported metrics are unaffected -- losses are computed from `gt` inside
    the batch loop, before anything is written back -- but `inference.csv` is the
    input to every application, and there the distortion is large. Recomputing
    Poss-Util over all 55,084 test possessions from corrected rows against the
    as-published values:

    | | as published | realigned |
    |---|---|---|
    | mean Poss-Util | -0.0764 | -0.0271 |

    Pearson r between the two is **0.556**, and the mean differs by a factor of
    2.8. On the 10-match sample the same comparison gives r = 0.993, so a small
    test would suggest the defect is cosmetic when at full scale it is not.

11. **The application plotters build paths by string concatenation.**
    `plot_poss_util_dist` and its siblings evaluate `save_path + "poss_util.png"`,
    so a `save_path` without a trailing separator writes `applicationsposs_util.png`
    *beside* the output directory instead of inside it.

12. **Early stopping saves the wrong weights.** `UEID_train.train` keeps the best
    model as `best_model = model_state_dict`, where `model_state_dict` is what
    `epoch_function` returned from `model.state_dict()`. `state_dict()` returns a
    dict of *references* to the live parameter tensors, not copies, and
    `optimizer.step()` mutates those tensors in place. `best_model` is therefore
    an alias for the current weights, not a snapshot of the best epoch. Verified
    directly: after five further optimizer steps the stored dict holds the new
    weights, and `best_model['weight'].data_ptr() == model.weight.data_ptr()`.

    The consequence is that under early stopping the checkpoint written to disk
    contains the weights from the *final* epoch, while being named
    `_model_<best_epoch>.pth` and reported alongside the best epoch's validation
    loss. With the default `early_stop_patience: 5` the saved model is five epochs
    past the one the run claims to have selected. The metrics printed during
    training are computed live and are unaffected, but every checkpoint the
    package writes -- and so every inference, simulation and application result
    derived from one -- uses a model the run itself measured as worse. A one-line
    `copy.deepcopy(model_state_dict)` would fix it.

13. **The team-name mapping is hardcoded to one league.**
    `UEID_preprocessing` carries a `team_dict` listing exactly the 20 La Liga
    sides of the season the authors used, and applies it only if it resolves
    every row:

    ```python
    df["team_map"] = df["team"].copy().map(team_dict)
    if df["team_map"].isnull().sum() == 0:
        df["team"] = df["team_map"]
    ```

    For any other league or provider the lookup misses, the branch is skipped in
    silence, and the raw provider identifier flows through to every downstream
    consumer. Since `plot_poss_util_dist` and `plot_HPUS` use the `team` value
    directly as the legend entry, a Wyscout run produces figures labelled
    `1651.0` and `Team 1624.0 (Home)`. Nothing warns that the mapping did not
    apply. Recovering names means going back to the provider's match files, which
    `run_downstream.py --matches-json` does; league position is recoverable the
    same way, from the results.

## 6. Reproducing the paper's reported model complexity

The paper does not state `hidden_dim` for NMSTPP; Appendix A lists it only as an
Optuna search space, `[16, 256, 512, 1024, 2048]`. The value is recoverable by
inverting the complexity columns of Table 6 (296M FLOPs, 121K parameters), since
the encoder feed-forward block dominates both:

| configuration | FLOPs | params |
|---|---|---|
| shipped `models/model_yaml` yaml (15 features, `seq_len` 1, `hidden_dim` 1024) | 26.8M | 168,707 |
| paper feature list, `seq_len` 40, `hidden_dim` 1024 | 956,579,336 | 154,145 |
| paper feature list, `seq_len` 40, **`hidden_dim` 256** | **295,976,456** | **121,121** |
| Table 6, NMSTPP / Wyscout | 296M | 121K |

Two independent configuration errors caused the mismatch reported in phase 1. The
parameter count came from `models/model_yaml/train_NMSTPP.yaml` adding
`home_score` and `away_score` to the paper's nine `other_features` -- the copy in
`examples/` carries the correct list. The FLOP count came from `seq_len: 1`, where
Appendix A and page 16 both specify 40.

Resolving them exposes a contradiction that no single configuration satisfies.
The complexity the paper reports requires `seq_len: 40`; at `seq_len: 40` the
released code does not train (issue 9). The configuration that reproduces the
paper's *predictive* metrics is `seq_len: 1`, which matches the reported
parameter count but falls 37x short on FLOPs.
