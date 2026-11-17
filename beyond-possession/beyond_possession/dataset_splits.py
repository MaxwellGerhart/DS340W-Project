"""Fixed match-level splits; Validation is the instructor's final unseen set.

Only the initial partitioning reads all records. Development readers load the
manifest and Training/Test files, never the Validation file.
"""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .matches import match_key, records_from_json

DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[1] / 'data/real/splits'
SPLITS = ('training', 'test', 'validation')
PURPOSES = {'training': 'Fit models and develop the implementation.',
            'test': 'Development testing, debugging, and model selection.',
            'validation': 'Final unseen evaluation only after development is frozen.'}


def _apportion(total, group_sizes):
    """Integer largest-remainder allocation with stable ties and exact totals."""
    size = sum(group_sizes.values())
    if not size:
        return {key: 0 for key in group_sizes}
    allocation = {key: total*n//size for key, n in group_sizes.items()}
    order = sorted(group_sizes, key=lambda key: (-(total*group_sizes[key] % size), key))
    for key in order[:total-sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def partition_matches(records, seed=42, ratios=(.7, .2, .1)):
    if (len(ratios) != 3 or any(not isinstance(r, (int, float)) or not math.isfinite(r)
                              or r <= 0 for r in ratios)
            or not math.isclose(sum(ratios), 1, abs_tol=1e-12)):
        raise ValueError('Use three positive Training/Test/Validation ratios that sum to one.')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('The split seed must be an integer.')
    records = records_from_json(records)
    keys = [match_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate match IDs must be resolved before splitting.')
    # Convert decimal ratios into integer weights for exact rounding.
    from fractions import Fraction
    fractions = [Fraction(str(r)) for r in ratios]
    denominator = math.lcm(*(r.denominator for r in fractions))
    weights = {name: int(r*denominator) for name, r in zip(SPLITS, fractions)}
    totals = _apportion(len(records), weights)
    if any(n == 0 for n in totals.values()):
        raise ValueError('Not enough matches for three nonempty partitions at these ratios.')
    groups = {}
    for record in records:
        group = str(record.get('competitionId', 'unknown'))
        groups.setdefault(group, []).append(record)
    sizes = {group: len(rows) for group, rows in groups.items()}
    training = _apportion(totals['training'], sizes)
    remaining = {group: sizes[group]-training[group] for group in sizes}
    test = _apportion(totals['test'], remaining)
    partitions = {name: [] for name in SPLITS}
    strata = {}
    for group, rows in sorted(groups.items()):
        # Hash ranking is repeatable across Python versions and input row ordering.
        rows = sorted(rows, key=lambda record: hashlib.sha256(
            json.dumps([seed, match_key(record)], separators=(',', ':')).encode()).digest())
        a, b = training[group], training[group]+test[group]
        chunks = {'training': rows[:a], 'test': rows[a:b], 'validation': rows[b:]}
        strata[group] = {name: len(chunk) for name, chunk in chunks.items()}
        for name, chunk in chunks.items():
            partitions[name].extend(chunk)
    for name in SPLITS:
        partitions[name].sort(key=match_key)
    sets = {name: {match_key(r) for r in rows} for name, rows in partitions.items()}
    if (set.union(*sets.values()) != set(keys)
            or sum(len(s) for s in sets.values()) != len(keys)
            or any(len(partitions[name]) != totals[name] for name in SPLITS)):
        raise RuntimeError('Split integrity check failed.')
    return partitions, strata


def write_match_splits(source, directory=DEFAULT_SPLIT_DIR, seed=42, ratios=(.7, .2, .1)):
    source, directory = Path(source), Path(directory)
    if directory.exists():
        raise ValueError('Split directory already exists. Keep the frozen assignments; use a new directory for a different dataset.')
    if source.suffix == '.gz':
        with gzip.open(source, 'rt', encoding='utf-8-sig') as file:
            catalogue = json.load(file)
    else:
        catalogue = json.loads(source.read_text(encoding='utf-8-sig'))
    records = records_from_json(catalogue)
    partitions, strata = partition_matches(records, seed, ratios)
    original = catalogue if isinstance(catalogue, dict) else {}
    canonical = json.dumps(sorted(records, key=match_key), sort_keys=True,
                           ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    manifest = {'schema_version': 'match-splits/1', 'seed': seed,
        'ratios': dict(zip(SPLITS, ratios)), 'split_unit': 'match_id',
        'stratified_by': 'competitionId', 'total_matches': len(records),
        'catalogue_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
        'splits': {}, 'competition_counts': strata,
        'match_splits': {match_key(record): name for name, rows in partitions.items() for record in rows},
        'policy': 'Training and Test are development sets. Validation is held out until final evaluation.',
        'data_type': 'Match metadata only; event/tracking records must inherit these same match assignments.'}
    output = {}
    for name, rows in partitions.items():
        ids = {match_key(r) for r in rows}
        content = {'schema_version': 'match-catalog/1', 'split': name, 'purpose': PURPOSES[name],
            'matches': rows, 'sources': original.get('sources', []),
            'provenance': {key: value for key, value in original.get('provenance', {}).items() if key in ids},
            'summary': {'unique_matches': len(rows), 'input_records': len(rows),
                        'source_files': len(original.get('sources', [])), 'identical_duplicates_removed': 0},
            'usage_note': manifest['data_type']}
        raw = (json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
        output[name+'.json'] = raw
        manifest['splits'][name] = {'file': name+'.json', 'matches': len(rows),
            'sha256': hashlib.sha256(raw).hexdigest(), 'purpose': PURPOSES[name]}
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.split-build-', dir=directory.parent))
    try:
        for name, raw in output.items():
            (staging/name).write_bytes(raw)
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        os.rename(staging, directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def read_manifest(directory=None):
    path = Path(directory or DEFAULT_SPLIT_DIR)/'manifest.json'
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'match-splits/1':
        raise ValueError('Unknown dataset split manifest; development access stopped.')
    return manifest


def check_match_access(records, *, final_validation=False, directory=None):
    manifest = read_manifest(directory)
    if manifest is None:
        return {'mode': 'unpartitioned', 'partitions': []}
    roles = {manifest['match_splits'].get(match_key(record), 'unassigned') for record in records}
    if 'validation' in roles and not final_validation:
        raise ValueError('This match belongs to the unseen Validation set. Keep it out of development; use --final-validation only after the implementation and model choices are frozen.')
    return {'mode': 'final_validation' if final_validation else 'development', 'partitions': sorted(roles)}


def development_view(catalogue, directory=None):
    """Filter known held-out IDs without opening their Validation data file."""
    manifest = read_manifest(directory)
    if manifest is None:
        return catalogue
    rows = [record for record in records_from_json(catalogue)
            if manifest['match_splits'].get(match_key(record)) != 'validation']
    ids = {match_key(record) for record in rows}
    return {**catalogue, 'matches': rows,
        'provenance': {key: value for key, value in catalogue.get('provenance', {}).items() if key in ids},
        'summary': {**catalogue.get('summary', {}), 'unique_matches': len(rows)},
        'split_policy': {'visible': ['training', 'test'],
                         'validation_excluded': manifest['splits']['validation']['matches']}}


def load_development_catalogue(directory=None):
    directory = Path(directory or DEFAULT_SPLIT_DIR)
    manifest = read_manifest(directory)
    if manifest is None:
        raise ValueError('Create the dataset splits first.')
    rows, provenance, sources = [], {}, []
    for name in ('training', 'test'):
        # Fixed filenames and no directory scan: Validation is never opened here.
        raw = (directory/(name+'.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['splits'][name]['sha256']:
            raise ValueError(name.title()+' file changed after splitting. Restore the frozen dataset.')
        catalogue = json.loads(raw)
        part = records_from_json(catalogue)
        if any(manifest['match_splits'].get(match_key(record)) != name for record in part):
            raise ValueError('A match appears in the wrong development split.')
        rows.extend(part)
        provenance.update(catalogue.get('provenance', {}))
        sources = catalogue.get('sources', sources)
    return development_view({'schema_version': 'match-catalog/1', 'matches': rows,
        'provenance': provenance, 'sources': sources,
        'summary': {'unique_matches': len(rows), 'input_records': len(rows),
                    'source_files': len(sources), 'identical_duplicates_removed': 0}}, directory)
