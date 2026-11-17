# Match catalogue

## Frozen dataset partitions

The catalogue is now split by match into `splits/training.json` (1,359 matches),
`splits/test.json` (388), and `splits/validation.json` (194). The ratios are
70% / 20% / 10%, with seed 42 and competition stratification. The dashboard reads
Training and Test only; Validation is reserved for the final unseen evaluation.
See [`../../docs/DATA_SPLITS.md`](../../docs/DATA_SPLITS.md) for the policy and
how future event/tracking data must inherit these match assignments.

## Original combined source

`combined_matches.json.gz` contains 1,941 unique match records from the supplied England, France, Germany, Italy, Spain, European Championship and World Cup files. It is gzip-compressed for sharing and is read automatically by the dashboard without an extra installation. The downloaded combined JSON remains ordinary JSON.

Source: Pappalardo et al. (2019), Wyscout public soccer match event dataset. Dataset collection: https://figshare.com/collections/Soccer_match_event_dataset/4415000/2 . The existing repository's reproduction documentation identifies this dataset as CC BY 4.0. The combined catalogue preserves source filenames, SHA-256 hashes and ID-level provenance; records are deduplicated by match ID without changing their fields. This file contains match metadata, not tactical event records or tracking data.

The expanded local combined_matches.json is ignored by Git. Once the frozen
splits exist, the dashboard uses Training and Test only; the original combined
source remains available for provenance and must not be used for development.
