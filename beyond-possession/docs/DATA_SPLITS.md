# Training, Test, and unseen Validation datasets

The 1,941-match catalogue is divided into three separate JSON datasets:

| Dataset | Target ratio | Matches | File |
| --- | --- | --- | --- |
| Training | 70% | 1,359 | `data/real/splits/training.json` |
| Test | 20% | 388 | `data/real/splits/test.json` |
| Validation | 10% | 194 | `data/real/splits/validation.json` |

This follows the assignment's terminology: Training and Test support development,
debugging and model selection; Validation is the final unseen holdout. In other
machine-learning conventions, the development and final evaluation names are
often reversed. Use the names above consistently in this project.

Whole matches are kept together. The split is stratified by `competitionId`,
uses seed 42, and uses deterministic hash ordering so input-file order cannot
change membership. Largest-remainder allocation gives exact total sizes while
preserving competition proportions as closely as possible. This is a random
match-level split, not a chronological forecast evaluation.

The original match fields, lineups and provenance are retained. No match ID
appears in more than one set, and every original match appears exactly once.
`manifest.json` records assignments, ratios, competition counts, seed and hashes.
Keep this manifest fixed for the rest of the experiment. Running the split
command again refuses to overwrite an existing split directory.

## Keeping Validation unseen during development

Creating the partitions necessarily reads the original input once. After that:

- The dashboard opens only `training.json` and `test.json`. It does not open
  `validation.json`; even a missing Validation file does not affect development.
- Uploading the original full match catalogue filters known Validation IDs out
  of its development view and download.
- Tactical analysis and experiment entry points reject matches assigned to
  Validation before an analyst/model is called.
- The final evaluation is available only through the explicit CLI flag
  `--final-validation`. Do not use it until code, prompts, settings and model
  choices are frozen and you are satisfied with Training and Test.
- Split tests use separate synthetic fixtures. They do not evaluate the real
  Validation matches.

The manifest can be read to enforce membership without inspecting held-out match
features or outcomes. These are workflow safeguards, not encryption or filesystem
access controls. Do not inspect the Validation file manually or use the original
full catalogue for model development. This protects future work; it cannot undo
any browsing of the full catalogue that happened before the split was frozen.

## Dataset limitation

These JSON files contain **match metadata**. They are not next-event training
records and contain no player-tracking snapshots. When the real event or tracking
data is added, join it to `manifest.json` by match ID and send all rows/snapshots
for each match to its assigned partition. Do not split event rows independently
or generate a new random split, which would risk match-level leakage.

The separate OpenSTARLab reproduction's original event-split scripts must be
updated to use these assignments before any NMSTPP training with these datasets.
The current dashboard performs analysis/inference; it does not train a neural
model on these metadata files. No training, testing scores or final Validation
results have been produced by creating the partitions.

For a different input catalogue, create its own new split directory:

```bash
python3 -m beyond_possession split --catalog path/to/catalogue.json --output data/real/new_splits
```

The normal dashboard uses `data/real/splits/`. Changing the input dataset or its
split directory requires deliberately adopting that dataset's manifest, rather
than mixing assignments across experiments.
