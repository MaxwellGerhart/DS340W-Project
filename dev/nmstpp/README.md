# NMSTPP reproduction (early development)

This was early development and testing for the DS 340W project. It is not part of [Beyond Possession](../../README.md). It has two parts:

1. **Reproduction.** We train the NMSTPP soccer event model from the OpenSTARLab paper on the public Wyscout dataset and reproduce its published results.
2. **Symmetry experiment.** We test whether building the pitch's left-right symmetry into that model helps.

## What you need

- Windows, macOS or Linux. No GPU is needed.
- Python 3.12 or 3.13. Check with `py --version`.
- Internet for the install, and about 1.5 GB of free disk space.

The data for the steps below is already in the repository. Nothing else has to be downloaded.

## Run it

This takes about 10 minutes. The commands are for Windows (PowerShell or Command Prompt).

### 1. Download the code

On GitHub, click **Code > Download ZIP** and unzip it. Open a terminal in the `dev/nmstpp` folder, which contains this `README.md`. Run every command below from that folder.

### 2. Install

```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

If `py` is not found, use `python` instead. If you have several Python versions, pick one with `py -3.12 -m venv .venv`.

### 3. Train the model

```
.venv\Scripts\python run_training.py --config configs/train_NMSTPP_demo.yaml
```

This trains NMSTPP for 2 epochs on 60 matches, which takes about 2 minutes. It proves the training code runs. It is too small a run to give the reported results, which come from step 4.

On a computer without a GPU it should end like this. With a GPU the losses come out slightly different.

```
Epoch [2/2]
Training Loss: 6.7541
Validation Loss: 6.3150

Best epoch: 2 | Best validation loss: 6.3150 | Corresponding training loss: 6.7541
FLOPs: 26802696 | Number of parameters: 168707
```

### 4. Reproduce the reported results

```
.venv\Scripts\python reproduce_results.py
```

This loads the models we trained on the full dataset (saved in `results/`) and scores them again on the held-out test set of 183 matches. It takes about 2 minutes and prints three tables.

**Part 1** is the reproduction, next to the numbers the OpenSTARLab paper publishes:

```
  metric                  this run     paper
  Action accuracy           0.6641      0.67
  Action F1                 0.1658      0.17
  Time MAE (s)              3.2999      3.34
  X MAE                     6.9586      6.94
  Y MAE                    15.0359     15.08
```

**Part 2** checks whether that model treats the left and right side of the pitch the same:

```
                             ACC_action    F1_action    F1_global   MAE_deltaT        MAE_x        MAE_y
  as published                   0.6641       0.1658       0.1672       3.2999       6.9586      15.0359
  mirrored test set              0.6631       0.1647       0.1656       3.3066       7.0400      15.1597
  test-time frame avg            0.6635       0.1642       0.1649       3.3054       6.9562      14.8716

                                y_gap_m      x_gap_m     dt_gap_s  action_flip    action_tv
  equivariance gap, model        3.9858       1.4786       0.2584       0.0131       0.0388
           frame-averaged       1.3e-06      2.7e-07      6.3e-08      0.0e+00      9.1e-09
```

**Part 3** is the symmetry experiment, 3 methods with 2 seeds each:

```
                            baseline       baseline+FA           augment        augment+FA          frameavg
  ACC_action        0.6698 +- 0.0013  0.6694 +- 0.0016  0.6696 +- 0.0012  0.6701 +- 0.0010  0.6687 +- 0.0033
  F1_action         0.1907 +- 0.0018  0.1881 +- 0.0014  0.1888 +- 0.0023  0.1889 +- 0.0037  0.1795 +- 0.0055
  F1_global         0.1982 +- 0.0017  0.1946 +- 0.0031  0.1934 +- 0.0054  0.1934 +- 0.0070  0.1833 +- 0.0022
  MAE_deltaT        3.2364 +- 0.0251  3.2388 +- 0.0272  3.3178 +- 0.1543  3.3192 +- 0.1548  3.3477 +- 0.0765
  MAE_x             6.7242 +- 0.0592  6.7006 +- 0.0014  6.5979 +- 0.0162  6.5497 +- 0.0172  6.8029 +- 0.2601
  MAE_y            14.9190 +- 0.2140 14.7091 +- 0.0500 14.8356 +- 0.1163 14.6818 +- 0.0580 14.7621 +- 0.0605
  mirrored_MAE_y   14.9919 +- 0.1903                 - 14.8488 +- 0.1477                 - 14.7621 +- 0.0605
  y_gap_m           3.8852 +- 1.0726                 -  3.0444 +- 1.3795                 -  0.0000 +- 0.0000
  x_gap_m           1.5489 +- 0.9267                 -  1.3316 +- 0.3683                 -  0.0000 +- 0.0000
  action_flip       0.0275 +- 0.0196                 -  0.0265 +- 0.0069                 -  0.0000 +- 0.0000
```

After each part the script prints how far its numbers are from the ones saved when the models were trained. That difference should be about `1e-07` or smaller, so a last printed digit can occasionally differ by one.

### macOS and Linux

Use `python3 -m venv .venv`, and write `.venv/bin/python` wherever the commands say `.venv\Scripts\python`.

## What is in this repository

| Path | What it is |
|---|---|
| `run_training.py` | Trains a model. Used in step 3. |
| `reproduce_results.py` | Scores the saved models and prints the results. Used in step 4. |
| `symmetry.py` | The left-right mirror and frame averaging used in the symmetry experiment. |
| `train_symmetry.py` | Trains one run of the symmetry experiment. |
| `prepare_data.py` | Downloads the full Wyscout dataset and converts it for training. |
| `make_demo_subset.py` | Cuts the small training set used in step 3 out of the full one. |
| `run_downstream.py` | Runs the paper's inference, simulation and figure code on a trained model. |
| `configs/` | Training settings. |
| `sample_data/` | The test set and the small training set, so steps 3 and 4 need no download. |
| `results/` | The models we trained on the full dataset, and their results. |
| `event/` | The OpenSTARLab Event package, unmodified. |
| `dev/` | Development notes, experiments and older runs. Not needed to run anything above. |

## Training on the full dataset (optional)

The models in `results/` were trained this way on an RTX 4060 laptop GPU. It takes hours, and retrained models score close to the saved ones but not identically.

```
.venv\Scripts\python -m pip install openstarlab-preprocessing requests
.venv\Scripts\python prepare_data.py
.venv\Scripts\python run_training.py --config configs/train_NMSTPP_full.yaml --epochs 2
.venv\Scripts\python train_symmetry.py --config configs/train_NMSTPP_reproduction.yaml --arm baseline --seed 0 --save-path results/symmetry_retrained
```

- The first line needs Python 3.12. It does not install on 3.13.
- `prepare_data.py` downloads the dataset into `data/` (1.5 GB) and takes 20 to 25 minutes.
- The third line trains the model behind part 1.
- The fourth line trains one run of part 3. The arms are `baseline`, `augment` and `frameavg`, and we used seeds 0 and 1. Score the retrained runs with `reproduce_results.py --symmetry-root results/symmetry_retrained`.

The full write-up of the reproduction, including the problems found in the published code, is in [dev/REPRODUCTION.md](dev/REPRODUCTION.md).

## Credits

- `event/` is the [OpenSTARLab Event package](https://github.com/open-starlab/Event) by Calvin Yeung and Keisuke Fujii (Apache-2.0). Paper: [OpenSTARLab: open approach for spatio-temporal agent data analysis in soccer](https://doi.org/10.1007/s40747-025-01965-y).
- The data is the public Wyscout dataset: Pappalardo et al., *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data, 2019 (CC BY 4.0).
- The symmetry experiment follows TacticAI: Wang et al., Nature Communications, 2024.
