# Match dataset split

The original 1,941 match records were randomly divided into training, test, and validation subsets using a 70% / 20% / 10% ratio and fixed seed `42`.

- `training/`: 1,359 records (70%)
- `test/`: 388 records (20%)
- `validation/`: 194 records (10%)

Splitting was stratified by source JSON file, so all seven competitions are represented in every subset. Each output directory contains the same seven filenames as the source archive. The match objects and their fields are unchanged; only their assignment and order differ.

`split_manifest.json` gives the count for every source file and subset. To reproduce the split, run:

```bash
python3 split_matches.py matches.zip matches_split --seed 42
```
