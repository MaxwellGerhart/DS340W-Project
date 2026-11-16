# Beyond Possession

DS 340W project: a soccer tactical-analysis system. It calculates tactical metrics from match data, has analyst agents turn them into observations backed by evidence, checks every number with a critic, and shows the result in a dashboard.

The project is the [`beyond-possession/`](beyond-possession/) folder. The [`dev/`](dev/) folder holds early development and testing and is not needed to run anything.

## What you need

- Python 3.11 or newer. Check with `py --version`.
- Nothing else. There is nothing to install and no data to download.

## Run it

This takes about 5 minutes. The commands are for Windows (PowerShell or Command Prompt).

### 1. Download the code

On GitHub, click **Code > Download ZIP** and unzip it.

### 2. Open a terminal in the `beyond-possession` folder

In File Explorer, open the unzipped folder, then open the `beyond-possession` folder inside it. Right-click an empty area and choose **Open in Terminal**.

Run every command below from this folder. If `py` is not found, use `python` instead.

### 3. Start the dashboard

```
py run.py --no-ai
```

The dashboard opens in your browser at http://127.0.0.1:8765. Click **Analyze match** to analyze the built-in demo match.

Press `Ctrl+C` in the terminal to stop it before the next step.

### 4. Run the tests

```
py -m unittest discover -s tests
```

It should end like this (the time will differ):

```
Ran 62 tests in 10.310s

OK
```

### 5. Produce the analysis report

```
py -m beyond_possession analyze --output artifacts/demo
```

It prints:

```
Report saved to artifacts/demo; 156 observations.
```

Open `artifacts/demo/report.md`. It is the same on every computer and starts like this:

```
# Beyond Possession: tactical report

Match: demo-001
System: multi | Backend: offline | Model: none
Run: 97026e106ead4559 | Data SHA-256: d61698582c674dc21398242736a2cb8ae41e2141dbabad88867bb639a6d17d28

## Evidence-backed observations

- **[verified]** Northbridge FC | 0-15 min: territory share = 60.32%.
```

### 6. Run the experiment

```
py -m beyond_possession evaluate --repeats 2 --output artifacts/experiment
```

This runs all six systems (baseline, single analyst, multi-agent and three ablations) twice each. It prints:

```
Experiment saved to artifacts/experiment; 0 failed runs. Human scores are intentionally blank.
```

Open `artifacts/experiment/results.json`. Every run has `numerical_faithfulness` 1.0 and `reference_coverage` 1.0, and every system has `mean_evidence_jaccard` 1.0. The multi-agent system makes 156 observations per run.

To run this step again, delete the `artifacts/experiment` folder first or use a new folder name.

### macOS and Linux

Use `python3` instead of `py`.

## The AI analyst (optional)

`--no-ai` runs the built-in analysis only. Without it, `py run.py` also sets up a local AI model, which downloads several GB the first time. See the [project README](beyond-possession/README.md).

## What is in this repository

| Path | What it is |
|---|---|
| `beyond-possession/run.py` | Starts the dashboard. |
| `beyond-possession/beyond_possession/` | The code: data, analytics, agents, critic, experiments and the dashboard server. |
| `beyond-possession/data/` | The built-in demo match and the catalogue of 1,941 real matches. |
| `beyond-possession/tests/` | The tests run in step 4. |
| `beyond-possession/docs/` | Architecture, data format and research plan. |
| `dev/` | Early development and testing. Not part of the project. |

## Credits

The match catalogue comes from the public Wyscout dataset: Pappalardo et al., *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data, 2019 (CC BY 4.0).
