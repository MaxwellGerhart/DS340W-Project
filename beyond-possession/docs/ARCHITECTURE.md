# Implementation architecture

## Data flow

1. Validate normalized event records and optional full tracking.
2. Compute every fixed interval for both teams; calculate descriptive adjacent-window TCI shifts.
3. Build an evidence catalog keyed by interval and metric.
4. Scope evidence to spatial, passing, defensive and game-state roles.
5. Each role selects observations using deterministic rules or an Ollama model.
6. The critic checks the reference, copied value, allowed alternative explanation and sample threshold.
7. The manager deduplicates references and assembles the final structured report.
8. Save data fingerprints, configuration, model/seed, evidence, decisions, tokens, and latency.

Local Ollama specialist calls run sequentially to reduce memory pressure and queue timeouts on laptops. Offline tasks and injected test backends can run concurrently; final assembly follows a stable role order. The deterministic critic does not ask a second model to judge facts. This controls one hallucination surface but does not establish tactical validity.

The current model has a constrained task: select evidence IDs and copy numbers. No unrestricted narrative, arbitrary code execution, autonomous tool loop, or recursive agent debate is implemented. `tools.MatchTools` supplies the public interfaces for a later tool-using version. All metrics are precomputed for this first working baseline. The manager records exploratory changes but does not adaptively repartition the timeline yet.

## Systems

| Configuration | Analyst behavior | Context | Critic |
| --- | --- | --- | --- |
| baseline | Fixed subset of event metric templates | Included in records | Enabled |
| single | One analyst selects from all available metrics | Included | Enabled |
| multi | Four specialist scopes | Included | Enabled |
| no_critic | Same specialists | Included | Quality checks bypassed; basic renderability still required |
| no_game_state | Three specialist scopes; context removed from analyst inputs and claims | Excluded from analyst layer | Enabled |
| no_specialists | One manager analyst selects from all metrics | Included | Enabled |

In this initial constrained architecture, `single` and `no_specialists` have effectively equivalent reasoning scope. They are retained as named configurations to support future manager-routing work, not treated as independent experimental contributions. The `summary` still contains calculated score context in no_game_state runs, but analyst prompts and exported claims do not. That ablation tests context access by the analyst, not removal of underlying raw data from the full run archive.

All systems share deterministic metric computation, templates and schema parsing. The single system has the same numerical guard for fairness; it does not call an LLM critic. Offline variants exercise routing and verification but do not simulate empirical LLM quality differences.

## Tactical Control Index

`TCI = 100 * (0.30 * territory_share + 0.25 * progressive_pass_rate + 0.20 * completion_rate + 0.25 * shot_share)`

All components are fractions. Missing any component yields null; no silent imputation or reweighting occurs. A team may have a low TCI despite high action volume, which demonstrates the proposed distinction between possession-like volume and progressive control. The two teams' TCIs need not sum to 100 because completion and progressive pass rates are team-specific.

Weights are a research hypothesis. They are accepted by the Python API through `analyze(..., weights={...})`; they must include all four keys, be finite and nonnegative, and sum to one. They are recorded in each report. The interface uses the documented default weights.

## Spatial methods

An 18 by 12 grid assigns each cell to the nearest player using physical pitch-scaled Euclidean distance. Ties contribute 0.5. Overall control is the mean assignment; central control is averaged in the selected team's central attacking third. Interval metrics average all snapshots; the dashboard surface shows the middle snapshot, explicitly labeled with its time.

This is a Voronoi proxy, not a reproduction of a published probabilistic pitch-control model. It does not use ball arrival, acceleration, reaction time, velocity, or goalkeeper influence parameters. Width/depth summarize outfield geometry across all phases, not phase-conditioned defensive organization.

## Evidence and verification

A claim's evidence record contains team, [start,end), metric, definition, full-precision value, sample size, and example source IDs. The original normalized input plus the interval/filter definition reproduces the aggregate; example IDs are not the complete denominator. Reports include a SHA-256 fingerprint of the entire normalized event/tracking input. Preserve that input alongside reports for a complete reproducibility archive.

Default minimum checks are 10 team events for event observations and 3 snapshots for spatial observations. These are software guardrails, not validated statistical confidence thresholds. In particular, a rare-event metric can have a small effective denominator despite many team events. Provider-specific denominator and missing-capability checks are required before formal data collection.

The model cannot provide arbitrary free prose in this version; final sentences use a fixed metric template. A malicious wrong value is rejected by the critic; a deliberately unchecked report can retain it, enabling meaningful adversarial software tests. No numeric confidence is invented. Tactical importance, representativeness, causality, and semantic contradictions require expert review.

## Operations

- Python standard library only, Python 3.11+.
- Dashboard binds to 127.0.0.1; not an authenticated public hosting service.
- Non-CORS HTTP API with per-launch token for mutations and bounded request bodies.
- Browser strings are escaped before HTML insertion.
- Uploaded datasets stay in memory in dashboard mode; export reports to retain results.
- CLI SQLite history is append-only, including repeated runs with the same configuration ID.
- Failed model calls do not silently fall back to offline rules.
- Tokens are logged when returned by Ollama. Monetary cost is null because no cost model is implemented.
- The run ID fingerprints data and configuration, not the model output. Store timestamps and full reports to distinguish repeated nondeterministic runs.

Groq API budget policy: at most 24 metric records per model call, balanced across team/interval groups; only compact score/card context is included. Default dashboard system is single analyst. The full multi configuration is still four separate role calls, with immediate stop on any failure. One-hour in-memory response caching is enabled only in dashboard requests, never CLI experiments. This policy is included in analysis metadata.
