import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from beyond_possession import dataset_splits as splits
from beyond_possession.data import demo
from beyond_possession.pipeline import analyze
from beyond_possession.evaluation import run_experiment


class DatasetSplitTests(unittest.TestCase):
    def records(self):
        return [{'wyId': i, 'competitionId': 'a' if i < 38 else 'b',
                 'teamsData': {'home': {'lineup': [1, 2, 3]}}, 'label': 'Match '+str(i)}
                for i in range(50)]

    def test_ratio_coverage_disjointness_and_original_fields(self):
        records = self.records()
        before = copy.deepcopy(records)
        parts, strata = splits.partition_matches(records)
        self.assertEqual({name: len(rows) for name, rows in parts.items()},
                         {'training': 35, 'test': 10, 'validation': 5})
        sets = [set(r['wyId'] for r in rows) for rows in parts.values()]
        self.assertEqual(set.union(*sets), set(range(50)))
        for i, a in enumerate(sets):
            for b in sets[i+1:]:
                self.assertFalse(a & b)
        self.assertEqual(records, before)
        self.assertEqual(sorted((r for rows in parts.values() for r in rows), key=lambda r: r['wyId']), records)
        self.assertEqual(sum(counts['training'] for counts in strata.values()), 35)

    def test_repeatable_independent_of_order_and_sensitive_to_seed(self):
        records = self.records()
        a, _ = splits.partition_matches(records, 42)
        self.assertEqual(a, splits.partition_matches(records[::-1], 42)[0])
        self.assertNotEqual(a, splits.partition_matches(records, 43)[0])

    def test_reject_duplicates_invalid_ratios_and_too_few_matches(self):
        with self.assertRaises(ValueError):
            splits.partition_matches([{'wyId': 1}, {'wyId': '1'}])
        for ratios in [(.7, .2, .2), (.7, 0, .3), (float('nan'), .2, .1)]:
            with self.assertRaises(ValueError):
                splits.partition_matches(self.records(), ratios=ratios)
        with self.assertRaises(ValueError):
            splits.partition_matches([{'wyId': 1}])

    def test_development_never_opens_validation_and_split_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'catalog.json.gz'
            with gzip.open(source, 'wt') as f:
                json.dump({'matches': self.records()}, f)
            target = root/'splits'
            manifest = splits.write_match_splits(source, target)
            before = (target/'manifest.json').read_bytes()
            # A broken/missing held-out file must not matter during development.
            (target/'validation.json').unlink()
            catalogue = splits.load_development_catalogue(target)
            self.assertEqual(len(catalogue['matches']), 45)
            self.assertTrue(all(manifest['match_splits'][str(r['wyId'])] != 'validation'
                                for r in catalogue['matches']))
            self.assertEqual(catalogue['split_policy']['validation_excluded'], 5)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                splits.write_match_splits(source, target, seed=123)
            self.assertEqual((target/'manifest.json').read_bytes(), before)

    def test_changed_development_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.json'
            source.write_text(json.dumps(self.records()))
            target = root/'splits'
            splits.write_match_splits(source, target)
            (target/'test.json').write_text('{"matches": []}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                splits.load_development_catalogue(target)

    def test_validation_blocked_before_analyst_or_experiment_is_called(self):
        rows, _ = demo()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{'match_id': rows[0]['match_id']}]+[{'match_id': 'fixture-'+str(i)} for i in range(49)]
            source = root/'source.json'
            source.write_text(json.dumps(records))
            target = root/'splits'
            manifest = splits.write_match_splits(source, target)
            heldout = next(k for k, v in manifest['match_splits'].items() if v == 'validation')
            for row in rows:
                row['match_id'] = heldout
            with patch.object(splits, 'DEFAULT_SPLIT_DIR', target), patch('beyond_possession.pipeline.run_agents') as model:
                with self.assertRaisesRegex(ValueError, 'unseen Validation'):
                    analyze(rows)
                with self.assertRaisesRegex(ValueError, 'unseen Validation'):
                    run_experiment(rows, [], root/'experiment')
                model.assert_not_called()
                self.assertFalse((root/'experiment').exists())
            with patch.object(splits, 'DEFAULT_SPLIT_DIR', target):
                report = analyze(rows, final_validation=True)
                self.assertEqual(report['config']['dataset_access']['mode'], 'final_validation')

    def test_full_catalogue_upload_filters_heldout_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.json'
            original = {'matches': self.records(), 'summary': {'unique_matches': 50}}
            source.write_text(json.dumps(original))
            target = root/'splits'
            splits.write_match_splits(source, target)
            visible = splits.development_view(original, target)
            self.assertEqual(len(visible['matches']), 45)
            self.assertEqual(len(original['matches']), 50)


if __name__ == '__main__':
    unittest.main()
