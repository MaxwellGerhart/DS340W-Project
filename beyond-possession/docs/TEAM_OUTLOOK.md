# Team outlook and next actions

The dashboard keeps two System choices: Single analyst and Specialist analysts. The other research systems remain in CONFIGS, the Python API and CLI. They have been removed from the visible menu, not deleted from the project.

## Using the outlook

1. Analyze a synthetic demo or an event dataset. A match catalogue alone cannot supply tactical evidence.
2. Use Focus team and Interval to choose the team and cutoff. The default selects a recent complete interval with enough outlook history.
3. Read the approximate next-phase estimates, cited priorities, match actions, practice exercises and monitoring suggestions.
4. By default, the initial selection receives an AI explanation after the main analysis. Disable this extra call under Report settings if desired. Filter changes alone are free of inference calls; use Expand with AI for the newly selected view.
5. Download JSON for every snapshot and full provenance, or Report for the latest team outlooks and any historical snapshots expanded with AI.

Inline help explains data sources, AI engines, analyst systems, writing, critic choices, weights, interval length, upload fields, training, filters, evidence, verification labels, charts, downloads and the outlook. A metric dictionary remains available before a match is analyzed.

## What is projected

The projection is a transparent, untrained trend baseline for a **comparable next playing phase**, conditional on similar circumstances continuing. It is not a fitted winner, scoreline or xG forecast. It does not predict events that already occurred in a later interval. Applying it at the end of a recorded match is a planning scenario, not a claim that extra playing time exists.

Use up to the three most recent adjacent complete intervals ending at or before the selected cutoff. Each needs at least ten team events. Completion/progression estimates also need ten attempted passes in every contributing interval. At least two intervals are required. Partial or sparse final intervals are excluded, with the actual evidence cutoff shown explicitly. Missing metrics remain unavailable.

For a metric's historical fractions v1 ... vn, the estimate is:

`clip(weighted_mean(v, weights=1..n) + 0.5*(vn-v1)/(n-1), 0, 1)`

Projected TCI is recalculated from the four projected components and the same fixed or learned weights as the analysis. If a component is unavailable, projected TCI is unavailable too. The displayed recent range is the observed historical minimum and maximum, not a statistical prediction interval. No confidence probability is fabricated.

The coefficient, sample gates and recommendation thresholds are uncalibrated prototype choices. Predictive accuracy and coaching utility require separate held-out evaluation. The module never opens the Validation dataset or learns from future windows.

## What is recommended

The rule-based plan selects from evidence-linked priorities: forward progression, ball security, chance creation, receiving variety, pressing outcomes, central access, attacking territory and balanced support. Only available metrics can trigger a priority; central-space advice requires sufficient tracking. Numerical priority scores are internal heuristic ranks, not measured expected benefits.

Each priority has a match action, practice idea and metric to monitor. These are hypotheses to test with coaching judgment and footage. Shot counts do not establish chance quality, pressure frequency does not establish pressing success, and passing concentration does not by itself establish predictability.

The optional AI call receives only the selected team's historical metric records through the displayed cutoff, qualitative trend directions and the bounded priorities. It explains and reorders supplied priorities, with citations. Calculated projection values stay in code; AI prose cannot introduce numeric statistics, invented priorities, unknown references or explicit guarantees. Player health, fatigue, injuries, unobserved passing lanes and invented formations are unsupported. A valid citation does not prove a recommendation's reasoning or effect.

## Failure and provenance behavior

If AI explanation fails, its error is shown within the outlook section. The already calculated trend and rule-based plan stay visible with their original labels; they are never presented as AI-generated output. The main match analysis remains intact. Successful explanations are cached for the stored report/team/interval selection and include model identity and call usage.

The endpoint uses an opaque handle for a server-held analysis, so clients cannot submit an arbitrary summary as if it had been analyzed. Only three recent reports are retained in memory. A server restart or eviction requires another analysis. Late responses from a previously selected team cannot replace the current team's view.

Research CLI evaluations continue to measure the analyst systems as before. Their reports include deterministic outlook snapshots, while optional AI outlook generation is a separate dashboard task and does not alter the original analyst/critic call budget.
