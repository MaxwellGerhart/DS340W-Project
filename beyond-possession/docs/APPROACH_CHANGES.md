# The three alternative approaches

Implemented from the three Group 25 presentations by Arnav Bedi and Maxwell Gerhart:

1. [Let the Data Pick the TCI Weights](https://docs.google.com/presentation/d/1j-GFFARIW56cIedNDYjLp8Dwq8UUy0TR/edit)
2. [An AI Critic](https://docs.google.com/presentation/d/1sYBUd9gcK8PPrmE2t6bUDqWR2HpCNaCi/edit)
3. [Free Writing, Then a Fact-Check](https://docs.google.com/presentation/d/1co283DugV2a7-ePNTuocDk0obxLhqTAd/edit)

The dashboard defaults to free writing and hybrid numeric/AI review when using shared Gemini. Original templates and the original numeric critic remain selectable. The Python API and CLI keep their original defaults for reproducibility. The statistical baseline always uses templates and numeric checks. No research improvement is claimed from implementation tests.

## 1. Learn TCI weights

Before: territory 0.30, progressive pass rate 0.25, completion 0.20, shot share 0.25.

Now: `tci_learning.fit_tci` can fit four nonnegative weights that sum to one. The historical interval's feature vector contains the same four fractions. Its target is the team's xG divided by both teams' xG in the **following 15 minutes**. Thus the learned TCI divided by 100 predicts future xG share, rather than an unscaled xG count. This choice preserves the original index scale but makes chance creation the training objective.

Projected gradient descent minimizes match-balanced mean squared error. A four-by-four Gram matrix keeps fitting fast. No xG value enters the historical features. Fully observed horizons within a single period are required; missing feature windows and zero-total-future-xG windows are excluded and counted. Coverage stops conservatively at the last observed event. Each independent match has equal total loss weight.

Training needs at least three eligible matches and six eligible team windows. This is a software minimum, not an adequate research sample size. Every shot/goal in each supplied match must include provider `xg` in [0,1]. Values are never guessed from shot positions. Shots/goals must not be duplicated. Existing normalized CSV files may add an optional `xg` column; JSON records may add `"xg": 0.12` on a shot or goal. All existing required event fields still apply. CSV and JSON are alternatives.

**The included match catalogue and Real Madrid vs Barcelona event file have no xG labels. No trained weights or invented accuracy results are bundled.** Obtain provider xG labels for multiple Training and Test matches before fitting. Test matches must be disjoint from the recorded Training IDs. Known frozen Test/Validation IDs cannot be fitted, and known Validation IDs cannot be tested through this development workflow. No Validation data file is opened. New unassigned match IDs rely on the user's explicit Training/Test file selection.

Use the dashboard's **Train TCI weights from future xG** panel, select Training events and optional Test events, and click **Train TCI weights**. Successful fitting selects Learned model automatically. Download the artifact to keep it, or upload a previous artifact. Its weights, target, window, training fingerprint and match IDs are recorded in analysis reports. Use the training window when applying the model. Test results show both fixed and learned MSE/MAE, averaged across whole matches.

From the `beyond-possession` folder:

```sh
python -m beyond_possession train-tci --events training_events.json --output artifacts/tci-model.json
python -m beyond_possession test-tci --events test_events.json --model artifacts/tci-model.json --output artifacts/tci-test.json
python -m beyond_possession analyze --events test_events.json --match-id YOUR_TEST_MATCH --tci-model artifacts/tci-model.json --output artifacts/learned-report
```

Lower Test error is possible, not guaranteed. The target's relationship to shooting, league transfer and human tactical-control ratings require separate experiments. Do not tune on the final Validation set.

## 2. A separate AI tactical critic

The critic receives the proposed report, cited deterministic stats, sample sizes, intervals and available score/card context. It returns exactly one pass/fail verdict and a short reason per claim. It checks usefulness, unsupported interpretations and contradictions across the report. Missing, duplicate, malformed and unknown verdicts stop the run; they never count as a pass.

| Critic control | Behavior |
| --- | --- |
| Original numeric critic | Existing copied-value and sample checks, plus prose number checks when writing is free |
| AI critic only | Experimental replacement of the template numeric gate; claims are labeled `ai_reviewed`, never numerically verified |
| Numbers + AI tactical review | Numeric checks first; a separate AI call reviews the surviving report |

Free writing retains its own numeric fact-check in both AI and hybrid critic modes. The deliberate `no_critic` ablation bypasses numeric and AI checks and labels its findings unchecked. `no_game_state` removes score context from both writer and reviewer inputs. Baseline/offline controls stay explicit and never simulate AI judgments.

To keep the entire review in one bounded call, AI critics review at most eight role-balanced findings with at most 24 cited metric records. Omitted candidates are counted as budget omissions rather than hidden factual rejections. The reviewer sees all eight together. Verdicts, reasons, model, tokens, latency and correction calls are included in exports.

By default this is a separate reviewer inference using the same base model as the writer. It is **not an independent-model ensemble**. Personal Gemini users can set `GEMINI_CRITIC_MODEL` in their local environment to use another accessible Gemini model. The shared service supports the same optional server-side setting. A different model is recorded when used. Either reviewer can still share the writer's mistakes. Human ratings remain blank until actual reviewers supply them.

## 3. Free writing with number checks

The writer can compose original sentences and connect up to three cited metric records. The code renders its sentence directly, rather than substituting a template. Input stays within the specialist's evidence scope; hosted writers receive at most 24 records per call, balanced across intervals and rotated across metric types.

Each sentence includes a primary copied value, evidence IDs and a binding for each occurrence of a printed number. The checker independently extracts digits, percentages, signed decimals, minute/score mentions and common number words. Each token must map to its declared cited metric or interval/context field. Percentages are converted from fractions, meter/TCI values retain their units, and rounding tolerance is half the final printed decimal. Unknown references, unbound numbers, wrong values, wrong units and small supporting samples are rejected. Exact digits inside cited team names are treated as identifiers.

Failed drafts go back with specific reasons **once per analyst call**. A second failed draft stays rejected. This is a content repair, not a retry after a provider/quota error. Earlier failures remain in the audit. Arbitrary derived differences, ratios and unbound numbers are intentionally unsupported. Unsupported facts without numbers can still pass numeric checking, which is why tactical AI review and expert evaluation matter.

## Compare the approaches

Select writing/critic controls in the app, or use CLI flags:

```sh
python -m beyond_possession analyze --backend shared --system single --writing free --critic hybrid --output artifacts/free-hybrid
python -m beyond_possession analyze --backend shared --system single --writing template --critic numeric --output artifacts/original
python -m beyond_possession evaluate --backend gemini --events test_events.json --match-id YOUR_TEST_MATCH --compare-approaches --repeats 2 --output artifacts/approach-comparison
```

The comparison produces blinded review packets for original templates/numeric critic, templates/AI-only critic, templates/hybrid critic, free writing/numeric critic and free writing/hybrid critic. It uses the same single-analyst system and seed within a repetition. Private mappings preserve the treatment, failures and resource use. If desired, apply the same learned TCI artifact with `--tci-model` across all five treatments so weight changes do not confound the writing/critic comparison. Compare fixed versus learned TCI separately using `test-tci`.

Shared Gemini uses a hosted cache and is for demonstrations, so the formal `evaluate` command requires a personal Gemini/Groq key or local Ollama. Free writing and AI review add inference calls and can reach provider limits sooner. No automatic quota retry, provider switch, or offline/template fallback is used.

## Code map

| Module | Responsibility |
| --- | --- |
| `tci_learning.py` | Future-xG examples, constrained training, model fingerprints and disjoint Test comparison |
| `narrative.py` | Fixed task schemas/prompts, number extraction and binding checks, fail-closed critic verdicts |
| `model_tasks.py` | Bounded direct Gemini/Groq/Ollama writer and reviewer calls |
| `agents.py` | Specialist scoping, content repair, numeric/AI/hybrid modes and audit trail |
| `shared_ai.py`, `hosted_ai_service/worker/index.js` | Scoped public writing and critic endpoints with server-held credentials |
| `server.py`, `static/` | Training uploads, model reuse and approach controls |
| `evaluation.py` | Blinded approach comparisons and separate numerical/prose audit statistics |

Implementation tests use synthetic fixtures only. They verify software contracts and guardrails, not coaching quality, model accuracy or improvement over the parent paper.
