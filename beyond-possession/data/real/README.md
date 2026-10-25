# Match catalogue

`combined_matches.json.gz` contains 1,941 unique match records from the supplied England, France, Germany, Italy, Spain, European Championship and World Cup files. It is gzip-compressed for sharing and is read automatically by the dashboard without an extra installation. The downloaded combined JSON remains ordinary JSON.

Source: Pappalardo et al. (2019), Wyscout public soccer match event dataset. Dataset collection: https://figshare.com/collections/Soccer_match_event_dataset/4415000/2 . The existing repository's reproduction documentation identifies this dataset as CC BY 4.0. The combined catalogue preserves source filenames, SHA-256 hashes and ID-level provenance; records are deduplicated by match ID without changing their fields. This file contains match metadata, not tactical event records or tracking data.

The expanded local combined_matches.json is ignored by Git. If both forms exist, the dashboard uses the expanded local copy.
