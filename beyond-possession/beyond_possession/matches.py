"""Lossless match-catalog merge; match metadata is separate from event analytics."""
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

EXPECTED_FILES = ['matches_England.json', 'matches_France.json', 'matches_Germany.json',
                  'matches_Italy.json', 'matches_Spain.json',
                  'matches_European_Championship.json', 'matches_World_Cup.json']

def match_key(record):
    for key in ('wyId', 'match_id', 'id'):
        value = record.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value)
    raise ValueError('Every match needs a nonempty wyId, match_id or id.')

def records_from_json(data):
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get('matches'), list):
        records = data['matches']
    else:
        raise ValueError('Expected a JSON array of matches or an object containing a matches array.')
    if any(not isinstance(r, dict) for r in records):
        raise ValueError('Every match must be an object.')
    return records

def combine_matches(paths):
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError('No input files supplied.')
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('An input file was selected more than once.')
    combined, provenance, sources = {}, {}, []
    input_count = duplicate_count = 0
    for path in sorted(paths, key=lambda p: str(p)):
        raw = path.read_bytes()
        data = json.loads(raw.decode('utf-8-sig'), parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON number: '+s)))
        records = records_from_json(data)
        source_id = 'source-' + str(len(sources)+1)
        sources.append({'source_id':source_id, 'filename':path.name,
                        'sha256':hashlib.sha256(raw).hexdigest(), 'record_count':len(records)})
        for record in records:
            input_count += 1
            identity = match_key(record)
            if identity in combined:
                if record != combined[identity]:
                    raise ValueError(f'Conflicting data for match {identity} in {path.name}; merge stopped without overwriting output.')
                duplicate_count += 1
            else:
                combined[identity] = record
            provenance.setdefault(identity, [])
            if source_id not in provenance[identity]:
                provenance[identity].append(source_id)
    if not combined:
        raise ValueError('Inputs contain no matches.')
    identities = sorted(combined, key=lambda s: (not s.isdecimal(), int(s) if s.isdecimal() else s))
    return {'schema_version':'match-catalog/1', 'matches':[combined[k] for k in identities],
            'sources':sources, 'provenance':{k:provenance[k] for k in identities},
            'summary':{'input_records':input_count,'unique_matches':len(combined),
                       'identical_duplicates_removed':duplicate_count,'source_files':len(sources)},
            'usage_note':'Match catalog only. Event records and full tracking are separate inputs to tactical analytics; do not manufacture them from fixture metadata.'}

def write_catalog(paths, output):
    paths = [Path(p) for p in paths]
    output = Path(output)
    if output.resolve() in {p.resolve() for p in paths}:
        raise ValueError('Output cannot overwrite an input file.')
    result = combine_matches(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent, delete=False) as file:
            temporary = file.name
            json.dump(result, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.write('\n')
        os.replace(temporary, output)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()
    return result

def load_catalog(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8-sig') as file:data=json.load(file)
    else:data=json.loads(path.read_text(encoding='utf-8-sig'))
    records = records_from_json(data)
    keys = [match_key(r) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Catalog contains duplicate match IDs.')
    return records
