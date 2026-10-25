# Research plan and paper structure

## Question

Can specialized AI analysts using deterministic evidence improve numerical faithfulness and tactical usefulness compared with a single model and a statistical template baseline?

The current project operationalizes the workflow. It does not answer the question yet.

## Before experimentation

1. Integrate the real dataset through a tested provider adapter.
2. Declare which metrics are supported by its event/tracking coverage.
3. Use separate development and held-out matches. Do not tune prompts or index weights on held-out matches.
4. Select a fixed local model, quantization, hardware configuration and generation parameters.
5. Establish comparable model-call/token budgets and report the extra cost of specialists.
6. Decide whether constrained selection is sufficient for the paper's question, or extend all systems to comparable tool use and open narrative with independent fact checking.
7. Pilot the human rubric with soccer-knowledgeable reviewers and revise before collecting final ratings.

## Proposed outcomes

| Outcome | Current support | Formal experiment requirement |
| --- | --- | --- |
| Numerical faithfulness | Compare selected values to deterministic evidence | Also audit extraction/mapping and meaningful natural-language claims if introduced |
| Reference coverage | Fraction with source-example IDs | Review whether evidence actually supports each interpretation |
| Tactical validity/usefulness | Blank 1 to 5 reviewer fields | Two or three blinded reviewers and inter-rater agreement |
| Insight diversity | Distinct metric/evidence counts available | Human rubric for useful distinct insights, normalized for report length |
| Contradictions | Human field only | Define conflicting claims and score independently |
| Reproducibility | Evidence-set Jaccard across repeats | Match-level repeated trials; document model nondeterminism |
| Latency and tokens | Wall time and returned model usage | Same hardware; end-to-end timing and model-call budgets |
| Cost | Null | Specify hardware/runtime valuation if relevant |

The critic's copied-value check is expected to produce high numerical faithfulness under constrained templates. Do not present this alone as evidence of superior tactical reasoning. In offline mode these values are selected and copied deterministically.

## Ablations

Compare specialists with/without critic and context. `no_specialists` is currently equivalent to single-analyst scope; strengthen the manager protocol before treating it as a separate finding. The critic ablation can demonstrate resistance to injected numeric errors in software tests, but real-world effect sizes must come from actual model outputs on held-out matches.

## Analysis plan

Use matches, not individual events or overlapping claims, as independent evaluation units. Record all model failures. Compare systems within each held-out match. Aggregate blinded reviewer scores and report uncertainty using match-level resampling only when enough independent matches exist. Predefine minimum sample sizes and missing-data handling. Never convert unavailable measures to zero.

Test TCI sensitivity across sensible alternative weights, window lengths and provider definitions. Validate it against human tactical-control ratings before claiming it measures control. Keep possession time separate from action share if actual possession durations become available.

## Paper outline

1. Introduction: tactical explanation beyond possession volume, research question and hypotheses.
2. Related work: soccer event/tracking analytics, explainable tactical assistance, multi-agent analytical workflows.
3. Data: provider, license, coverage, preprocessing, clocks, coordinates, held-out split and limitations.
4. Methods: TCI definition, deterministic tools, agent scopes, critic rules, model configuration and evidence schema.
5. Experimental design: baselines, budgets, ablations, reviewer blinding, rubric and agreement.
6. Results: measured values only, including failures, latency and quality tradeoffs.
7. Discussion: what constrained generation proves, what it does not prove, and event-only limitations.
8. Reproducibility: repository, environment, sample fixtures, prompts, seeds and dataset access instructions.

No result percentages in the original concept are implemented as observed findings. They were examples of possible future results.
