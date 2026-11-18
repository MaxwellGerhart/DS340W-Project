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

AI specialist calls run sequentially and stop on the first provider error. Offline tasks may run concurrently. The original numeric critic remains available, and optional AI-only/hybrid modes use a separate model inference to judge usefulness, reasoning and contradictions. See [the three approach changes](APPROACH_CHANGES.md) for the current controls, schemas and validation limits.

Template mode selects evidence IDs and copies numbers. Free-writing mode allows original sentences with up to three citations and explicit numeric bindings, checked independently before rendering. One bounded content correction is supported. No arbitrary code execution, autonomous tool loop or recursive debate is implemented. Metrics remain precomputed and intervals are fixed.

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

All systems share deterministic metrics and schema parsing. Writing/critic controls are independent treatments, except the statistical baseline always uses templates and numeric checks and no_critic deliberately disables checks. Offline variants exercise routing and verification but do not simulate AI writing or judgment. Approach-comparison packets hold the analyst system constant.

## Tactical Control Index

`TCI = 100 * (0.30 * territory_share + 0.25 * progressive_pass_rate + 0.20 * completion_rate + 0.25 * shot_share)`

All components are fractions. Missing any component yields null; no silent imputation or reweighting occurs. A team may have a low TCI despite high action volume, which demonstrates the proposed distinction between possession-like volume and progressive control. The two teams' TCIs need not sum to 100 because completion and progressive pass rates are team-specific.

Weights remain a research hypothesis. Explicit `weights={...}` and validated learned TCI artifacts are supported, with finite nonnegative weights summing to one. The interface uses fixed weights until a learned model is trained or uploaded. Training fits future xG share on Training matches; separate Test scoring compares learned and original weights. Known Validation IDs stay locked. Bundled event files lack xG labels, so no learned model is bundled.

## Spatial methods

An 18 by 12 grid computes the nearest player on each team using physical pitch-scaled Euclidean distance. Each cell now uses a smooth weight: sigmoid((opponent distance - own distance) / 3 meters). Ties contribute 0.5 and the two teams' weights sum to one. This reduces abrupt cell flips caused by tiny position changes near a boundary. The 3 meter scale is an uncalibrated starting setting. Overall control is the mean weight; central control is averaged in the selected team's central attacking third. Interval metrics average all snapshots; the dashboard surface shows the middle snapshot, explicitly labeled with its time. `control_surface(..., method='nearest')` retains the original binary proxy for comparison. Reports record the spatial method and scale.

This is a soft distance-based proxy, not a reproduction of a published probabilistic pitch-control model. It does not use ball arrival, acceleration, reaction time, velocity, or goalkeeper influence parameters. Reduced boundary jitter does not establish better tactical accuracy; that requires validation with real tracking data. Width/depth summarize outfield geometry across all phases, not phase-conditioned defensive organization.

## Evidence and verification

A claim's evidence record contains team, [start,end), metric, definition, full-precision value, sample size, and example source IDs. The original normalized input plus the interval/filter definition reproduces the aggregate; example IDs are not the complete denominator. Reports include a SHA-256 fingerprint of the entire normalized event/tracking input. Preserve that input alongside reports for a complete reproducibility archive.

Default minimum checks are 10 team events for event observations and 3 snapshots for spatial observations. These are software guardrails, not validated statistical confidence thresholds. In particular, a rare-event metric can have a small effective denominator despite many team events. Provider-specific denominator and missing-capability checks are required before formal data collection.

Free-written prose is checked using extracted numeric tokens and declared field bindings, with displayed-precision rounding. Numeric checks do not prove the nonnumeric reasoning. AI review is a separate, fallible inference and defaults to the same base model; optional GEMINI_CRITIC_MODEL allows another Gemini model. Verdicts and model identity are logged. AI-only template and no_critic ablations can retain numeric errors and label them accordingly. Human tactical validity scores remain unset.

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
