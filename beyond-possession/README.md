# Beyond Possession

**A reproducible soccer tactical-analysis research project for DS 340 W.**

Specialized analysts, deterministic soccer analytics, and a numerical critic, with an interactive dashboard and an experiment runner. Built from the capstone concept *Beyond Possession: A Multi-Agent AI System for Explainable Tactical Analysis in Soccer*.

**Status:** working research prototype. Synthetic data is included. Offline rules require no installation beyond Python and no API key. Optional local Ollama models turn evidence selection into LLM specialist calls. Neither synthetic results nor offline comparisons establish that multi-agent AI is superior.

## No API quotas: local AI is the default

The project now defaults to **Ollama with qwen2.5:3b**, running on your own machine. No API key, paid plan, daily provider quota, or per-minute provider quota is needed. The first launch downloads the runtime and model; later launches use the cache and can run offline.

Local inference is not literally unlimited: RAM, processing speed, context length and available disk space still apply. This default is a lightweight starting point, not a claim that it matches the quality of larger hosted models. Keep `AI_PROVIDER=ollama` in `.env`. Groq remains optional.

The Python TLS certificate fix also covers the local runtime installer. Certificate and hostname verification remain enabled. It uses the public CA bundle shipped with certifi/pip when available.

## Optional hosted API: Groq

If you set `AI_PROVIDER=groq`, the F5 launch uses **Groq with openai/gpt-oss-20b**. It does not install Ollama or download model files. Built-in analysis and match browsing still work without a key.

1. Sign in at https://console.groq.com/keys and create a key using your own free-tier account.
2. In VS Code, open the local **.env** file next to run.py. A blank template has been created; a fresh checkout can copy **.env.example** to **.env**.
3. Set `GROQ_API_KEY=your_actual_key`. Leave `AI_PROVIDER=groq`.
4. Restart run.py, select **Free API (Groq)**, and click **Check AI connection**.
5. Use event data or the synthetic demo for AI tactical analysis. The saved match catalogue is for browsing; it does not contain tactical events.

Keep the account on its Free tier to avoid paid API usage. Groq sets request and token limits; these can change and depend on your account. The app cannot verify your account's billing tier or enforce free pricing if you choose a paid account. It does not upgrade plans, add billing details, switch providers or retry rate-limited calls automatically.

Your key stays in the backend environment, excluded from Git and exported reports. Only calculated evidence is sent to the fixed Groq API endpoint. The app sends role-scoped metric values, intervals, team labels and game-state context; it does not send the original entire dataset. Internet is required for API analysis. Teammates should create their own keys and .env files.

**Testing:** API behavior is tested with mocked responses. No live Groq inference is claimed until an actual key has been added and tested.

## Local AI setup (default)

Press F5 normally, or run `python run.py --local-ai`. Before opening the dashboard, the launcher:

1. Reuses an available Ollama service, or downloads the official portable runtime for your platform.
2. Checks the runtime archive against the SHA-256 digest published on the official GitHub release.
3. Starts a project-local service on localhost port 11435 when needed.
4. Downloads **qwen2.5:3b** if it is missing, with progress shown in VS Code's terminal.
5. Selects local AI in the dashboard once setup succeeds.

The first run needs internet and several GB of free storage. It may take several minutes. Later runs reuse the cache and can work offline. This prepares a model for inference; it does not train a model. Runtime compatibility, available memory and model quality still vary by machine.

The runtime and model are cached under **.local-ai/**, which is excluded from Git and project ZIPs. If an existing Ollama service is reused, its own model cache is used instead. No administrator install is attempted. No system-wide services are added. A project-owned AI server is stopped when the dashboard exits; an existing Ollama service is left running.

Automatic portable setup supports macOS, Windows x64, and Linux x64/ARM64. Linux needs tar with zstd support. Other platforms can use a manually installed compatible Ollama service. The runtime release is pinned in `beyond_possession/local_ai.py` for repeatable setup.

To skip all AI downloads, select **Run dashboard without AI downloads** in VS Code, or run `python run.py --no-ai`. To populate the cache without launching the dashboard, run **setup_ai.py**. If downloads fail, the dashboard still starts in built-in mode and shows that AI setup did not finish. It does not pretend that a model ran.

You cannot guarantee an already-downloaded multi-GB model on a brand-new machine from a small GitHub checkout. This first-run setup makes the download automatic, then reuses it. A completely offline first install would require distributing a separate runtime/model bundle for each platform.

## Open and run in VS Code

1. Extract this project and open **Beyond-Possession.code-workspace** with VS Code. Alternatively, use **File > Open Folder** and select this project's `beyond-possession` folder.
2. Install the recommended **Python** and **Python Debugger** extensions from Microsoft when VS Code offers them.
3. Choose **Python: Select Interpreter** in the Command Palette and select **Python 3.11 or newer**. Python must already be installed on your laptop. No pip dependencies are required. The default launch prepares local Ollama once, then reuses the downloaded model.
4. Press **F5**, or open **Run and Debug**, select **Run dashboard**, and click the green play button. The dashboard opens in your browser automatically.

You can also open **run.py** and click **Run Python File** at the top right. Use the Python extension's Run button, rather than a separate Code Runner extension.

Stop with **Shift+F5** when debugging, or **Control+C** in the integrated terminal when using Run Python File. The dashboard loads the saved real match catalogue when available; otherwise it starts with synthetic demo data. The selected interpreter and paths are portable across macOS, Windows and Linux.

The Run and Debug menu also contains **Export demo report** and **Combine the seven match datasets**. For combining, first save all seven JSON files under `data/raw/`. The match catalog remains separate from tactical event analytics.

**Terminal > Run Task** offers dashboard startup, tests, dataset merging, and report export. Tests also appear in VS Code's Testing panel.

If port 8765 is busy, stop the earlier server or run `python run.py --port 8766` in VS Code's terminal. If your browser does not open automatically, use the URL printed in the terminal.

## Start in one command

Install **Python 3.11 or newer**, open a terminal inside this folder, and run:

```bash
python3 -m beyond_possession serve
```

Open **http://127.0.0.1:8765**. On Windows, use `py` instead of `python3`.

The dashboard automatically analyzes a synthetic match. Choose a team and interval to explore TCI, action share, passing networks, tracking-based spatial estimates, evidence, and critic decisions. Change systems to explore the ablations. Export JSON or Markdown reports.

**There are no pip dependencies, paid services or cloud accounts. The optional local Ollama launch downloads the local model once; use `python run.py --no-ai` to skip AI setup.** The dashboard works offline. Files selected in the dashboard are analyzed locally; they are not uploaded to a model service in offline mode. Ollama mode sends calculated evidence to the configured Ollama endpoint.

## What works now

- Seeded synthetic event data and full-team tracking snapshots.
- Strict CSV/JSON validation and a documented provider-neutral data contract.
- Interval analytics: event territory share, progressive passing, pass completion, shot share, recovery location, and pressure-event rate.
- Provisional, configurable event Tactical Control Index with explicit weights and missing-data behavior.
- Directed passing graphs with edge counts, density, and receiving hub share.
- Nearest-player spatial control, central attacking access, and outfield width/depth when full snapshots exist.
- Manager orchestration, spatial/passing/defensive/game-state specialists, and a numerical/reference critic.
- Optional local LLM specialist calls with structured JSON output; deterministic offline scaffolding.
- Three main comparison systems and three ablations, repeatability measurements, usage/latency logs, anonymous reviewer packets, and blank human-rating forms.
- JSON/Markdown report exports and optional SQLite run history.
- Public analytical tool functions, automated tests, and GitHub Actions CI.

## Run analyses and experiments

```bash
# Synthetic demo; produces report.json and report.md
python3 -m beyond_possession analyze --output artifacts/demo

# Persist the full run in SQLite as well
python3 -m beyond_possession analyze --database artifacts/runs.sqlite

# Analyze the included CSV and tracking snapshots
python3 -m beyond_possession analyze --events data/demo/events.csv --tracking data/demo/tracking.json --output artifacts/from-files

# Event-only data: spatial metrics remain unavailable
python3 -m beyond_possession analyze --events data/demo/events.csv --output artifacts/event-only

# All six systems, twice each; use a new output folder per experiment
python3 -m beyond_possession evaluate --repeats 2 --output artifacts/experiment-01

# Regenerate synthetic data with another seed
python3 -m beyond_possession demo --seed 73 --output data/generated

# Verify the project
python3 -m unittest discover -s tests -v
```

For multi-match CSVs, select one match with `--match-id YOUR_ID`. The dashboard has a match-ID field for this too. Run separate experiments on each held-out match; do not mix matches into one passing graph. Experiment outputs include:

```text
artifacts/experiment-01/
  results.json           Automatic checks, failures and repeatability
  blinded/
    R001.md              Reports without system labels
    ratings.csv          Empty scores for three independent reviewers
  private/
    review_key.json      System/repeat mapping; KEEP AWAY FROM REVIEWERS
    R001.json            Complete run, evidence and traces
```

Offline experiments usually have perfect numerical faithfulness by construction. This is a software property, not a research outcome. Tactical validity, usefulness, contradiction rates and cost remain unscored until measured properly. Report style and length may partially reveal systems even after labels are removed.

## Optional real LLM analysts with Ollama

Install [Ollama](https://ollama.com/), run its local service, and pull a model your machine can support:

```bash
ollama pull qwen2.5:3b
```

Then select **Local LLM** in the dashboard or run:

```bash
python3 -m beyond_possession analyze --backend ollama --output artifacts/local-llm
python3 -m beyond_possession evaluate --backend ollama --repeats 2 --output artifacts/llm-experiment-01
```

Optional configuration:

```bash
export OLLAMA_MODEL=qwen2.5:3b
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Windows PowerShell: `$env:OLLAMA_MODEL = "qwen2.5:3b"`.

Requests use a 16,384-token context setting and compact evidence records. Check actual context and memory support on the chosen model. Model files can be several gigabytes. Local inference needs suitable memory and can be slow. No paid API key is required. The offline backend never calls an LLM; run.py selects the local model after automatic setup succeeds. A failed model call produces an error, not a fabricated successful analysis. Model selection and real inference must be validated on your hardware before collecting experimental results.

This first implementation intentionally constrains LLMs to choosing precomputed observations and approved alternative explanations. The final numerical text is rendered from a schema, not unrestricted prose. Specialists do not yet perform autonomous multi-step tool planning or open-ended debate. The manager and critic are deterministic components. See [architecture](docs/ARCHITECTURE.md) before describing the system in your paper.

## Add the real dataset later

Start with [DATA_CONTRACT.md](docs/DATA_CONTRACT.md). Place private raw files under `data/raw/` (ignored by Git), write a provider adapter, and produce the normalized event CSV and optional tracking JSON.

The current loader **does not directly read raw Wyscout/Figshare, StatsBomb, Metrica, or SkillCorner files**. Their fields, coordinates, clocks, event taxonomies and licenses differ. The provider adapter is the intended integration point once the actual dataset is supplied. Event datasets alone cannot enable tracking-based metrics. StatsBomb 360 partial freeze frames also do not meet this prototype's full-team tracking assumptions.

## Share on GitHub

Create an empty repository on GitHub. From inside this project folder:

```bash
git init
git add .
git commit -m "Initial Beyond Possession research prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/beyond-possession.git
git push -u origin main
```

Invite your teammate through the repository's **Settings > Collaborators**. They can then run:

```bash
git clone https://github.com/YOUR_USERNAME/beyond-possession.git
cd beyond-possession
python3 -m beyond_possession serve
```

No real repository is created or published by this download. The included data is generated and safe to share as synthetic fixtures. Check licensing before committing external datasets. Real-data folders, environments, SQLite history, experiments and credentials are ignored by default.

## Structure

```text
beyond_possession/
  data.py          Normalization, validation and synthetic generation
  analytics.py     Deterministic metrics and spatial/network calculations
  tools.py         Public analytical tool interface
  agents.py        Specialist roles, local model backend and critic
  pipeline.py      Analysis, provenance and report/SQLite exports
  evaluation.py    Experiments and blinded human review
  server.py        Local HTTP API and dashboard delivery
  static/          Dashboard HTML, CSS and JavaScript
  __main__.py      CLI
 data/demo/        Synthetic CSV, tracking snapshots and provenance
 docs/             Data contract, architecture, research plan and teammate tasks
 tests/            Data, numerical, adversarial-agent and integration checks
 .github/workflows/ci.yml
```

## Research boundaries

TCI is an unvalidated event proxy, not an established scientific measure. Action share is not possession time. Spatial estimates use static nearest-player geometry, not velocities, reaction time, arrival probability, or a calibrated pitch-control model. Width/depth are not conditioned on defensive phases, so the dashboard calls them geometry rather than proven defensive quality. Adjacent-window threshold changes are exploratory, not statistical change-point inference. No calibrated confidence scores, bootstrap intervals, fatigue estimates, causal effects, passing-lane visibility, GNNs, or player simulations are claimed.

See [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) and [TEAM_HANDOFF.md](docs/TEAM_HANDOFF.md) for the next work.

## Combine the seven supplied match files

Save the seven `matches_*.json` attachments together in `data/raw/`, then run:

```bash
python3 -m beyond_possession combine
```

Or select their folder explicitly:

```bash
python3 -m beyond_possession combine --input-dir "/path/to/downloaded/files"
```

This creates **one** file, `data/real/combined_matches.json`, containing original match records, a source manifest, SHA-256 hashes, ID-level provenance and counts. Identical duplicate IDs are merged; conflicting records stop the operation without replacing existing output. Input records retain their original fields. JSON arrays and objects with a `matches` array are supported.

The catalog is readable through `beyond_possession.matches.load_catalog(path)`. It is separate from the event/tracking analytics input. Files with match lineups, scores and fixtures do not by themselves supply passes or player coordinates.

**Current verification:** the local copies of all seven match files were read and combined into 1,941 unique records with no duplicates. The catalogue is integrated into the dashboard. Tactical analysis on real event data remains pending because these files contain match metadata.

## Troubleshoot Ollama

The dashboard defaults to **Offline rules**, which does not need Ollama. Selecting **Local LLM** requires a separately installed Ollama service and a downloaded model.

1. Install and open the Ollama app from https://ollama.com/download.
2. In VS Code's terminal, run `ollama pull qwen2.5:3b`.
3. Keep the Ollama app running. Alternatively, start `ollama serve` in a separate terminal if it is not already running.
4. Click **Check Ollama** in the dashboard or run `python -m beyond_possession doctor`.
5. When the check succeeds, select Local LLM and analyze again.

Specialist Ollama requests now run sequentially to reduce simultaneous model-memory pressure. Each generation request has a 300-second timeout; connection checks use a 5-second timeout. HTTP errors include Ollama's returned error, rather than only the agent name.

If the model cannot fit in memory, pull a smaller model, then set `OLLAMA_MODEL` to its exact installed name before restarting the dashboard. The default remains qwen2.5:3b. Actual inference speed depends on your laptop.

## Upload files in the dashboard

- **Event data (CSV or JSON):** choose one event file in either format. JSON can be an array or an object with an `events` array. The same documented event fields are required in both formats. Tracking JSON is optional, and requires the documented full-team snapshot format.
- **Match JSON files:** select the league/tournament files together, click **Load match files**, then **Download combined JSON**. This displays match records and creates one combined catalogue. No CSV is needed for this step.

The `matches_*.json` filenames suggest match catalogues, but inspect their records before deciding what is available. Match details, scores and lineups cannot substitute for event-by-event passes, shots and coordinates. Raw provider event JSON also requires mapping to the project's event schema.

## Your real match catalogue

The seven local league/tournament files have now been combined into `data/real/combined_matches.json`: **1,941 unique matches, zero duplicate records**. It preserves match details, scores, lineups, source filenames and source-file hashes. The dashboard loads this saved catalogue automatically at startup. Use **Download combined JSON** to export it.

These files do not contain individual passes, shots or player-location snapshots. To calculate tactical metrics, supply a separate event dataset. For normalized event records, JSON alone or CSV alone is enough; both are never required. A provider-specific raw event format must first be mapped to the documented schema.

Open this project folder directly in VS Code.

## Free API quota protection

Opening the dashboard never runs a model automatically. Loading match records and changing team/interval filters also do not call Groq. Click **Analyze match** explicitly to run an analysis.

The dashboard defaults to **Single analyst (1 API call)**. Specialist mode remains available and can make up to four calls; it stops immediately when one fails. A bounded API prompt sends a balanced selection of at most 24 metric records per role, rather than the entire evidence catalogue. The full deterministic metrics remain in the dashboard. This changes what evidence the model can inspect, so record this prompt policy in research comparisons.

Successful dashboard responses are cached in server memory for one hour (up to 64 entries), scoped by key, model, role, seed and exact prompt. Repeating an identical dashboard request can reuse a cached response. CLI/research runs do not use this cache. The UI shows actual model requests and cache hits. Errors are not cached or retried automatically.

An HTTP 429 does not necessarily mean your daily free allowance is gone. It can be a per-minute token/request limit or an oversized request. The error now includes Groq's actual explanation and Retry-After value when provided. Account-specific limits apply at organization level; changing keys does not create a new allowance. Check https://console.groq.com/settings/limits for your current limits.

## Location in the team repository

This app lives under `beyond-possession/` alongside the existing OpenSTARLab reproduction. Open `beyond-possession/Beyond-Possession.code-workspace` in VS Code, or run `cd beyond-possession` then `python run.py`. The original reproduction remains at the repository root. The shipped match catalogue is gzip-compressed and read automatically.
