# Teammate handoff

## First shared session

1. Put this folder into one GitHub repository and invite the team.
2. Each teammate runs `python3 -m beyond_possession serve` and tests the demo.
3. Run the test suite, then generate one CLI report and one offline experiment.
4. Agree on dataset/provider, one match to inspect manually, and the normalization rules.
5. Create issues from the workstreams below. Use branches and pull requests.

## Workstreams

| Owner | Scope | Main files | First reviewable deliverable |
| --- | --- | --- | --- |
| Lead / integration | Scope, interfaces, merge review, reproducibility | pipeline.py, docs/ | Agreed data contract and runnable shared repository |
| Data teammate | Provider mapping, licensing, normalized outputs | data.py, future adapters/ | One real match, provenance and fixture-based adapter tests |
| Analytics teammate | Validate metrics, network analysis, TCI sensitivity | analytics.py, tools.py | Hand-calculated checks and documented metric limitations |
| Agents / evaluation teammate | Local model trials, prompts, fair budgets, blinded review | agents.py, evaluation.py | One actual model report plus a reviewed experimental protocol |

If fewer people are available, combine data and analytics; keep one person accountable for integration.

## Definition of done for the first deliverable

- Everyone can start the dashboard from the README.
- A licensed real-match sample can be normalized and validated.
- Event-only outputs explicitly omit unsupported spatial analysis.
- At least one calculation matches a manual audit.
- Every final observation references calculated evidence.
- The team can explain the difference between offline rules and actual model calls.
- A sample baseline, single and specialist report can be generated.
- Reviewer rubric is agreed, but research findings are not invented.

## Useful GitHub issues

- Add provider adapter after dataset selection.
- Add per-provider capability flags so absent event types are unavailable, not zero.
- Replace goalkeeper heuristics with lineup roles.
- Audit clocks and attacking directions across halftime.
- Add denominator-specific reliability thresholds.
- Compare TCI weight and window sensitivity.
- Extend specialist output to interpretable claims with independently checkable facts.
- Add true tool-calling and bounded revision loops if required by the research scope.
- Match model token budgets across systems.
- Collect blinded expert reviews on held-out matches.
