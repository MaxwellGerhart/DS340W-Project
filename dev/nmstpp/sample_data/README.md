# sample_data

Part of the public Wyscout dataset, already converted for training, so the steps in the main [README](../README.md) run without downloading anything.

| File | Contents |
|---|---|
| `test.csv.gz` | The full held-out test set: 183 matches, 267,732 events, from five leagues. |
| `train_demo.csv.gz` | 60 matches for the short training run. |
| `valid_demo.csv.gz` | 10 matches for the short training run. |

These are the exact files `prepare_data.py` and `make_demo_subset.py` write, compressed with gzip. Running those two scripts rebuilds them from the original download.

Source: Pappalardo et al., *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data 6, 236 (2019). Licensed CC BY 4.0. Converted with the OpenSTARLab PreProcessing package.
