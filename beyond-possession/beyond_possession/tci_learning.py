"""Train simplex-constrained TCI weights against strictly future xG share.

No external dependencies, automatic downloads or access to Validation records.
The loss predicts chance creation, not an established definition of control.
"""
import hashlib
import json
import math
from collections import defaultdict

from .analytics import WEIGHTS, window_metrics
from .data import fingerprint, validate_events
from .dataset_splits import check_match_access, read_manifest

FEATURES = tuple(WEIGHTS)
SCHEMA = 'learned-tci/1'
HORIZON = 900


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _check_partition(rows, purpose):
    check_match_access(rows)
    manifest = read_manifest()
    if manifest:
        roles = {manifest['match_splits'].get(str(row['match_id']), 'unassigned') for row in rows}
        if roles - {purpose, 'unassigned'}:
            raise ValueError('TCI '+purpose+' input contains matches assigned to another frozen partition.')


def examples(rows, window_minutes=15):
    """Historical [start,end) features; target [end,end+900) xG share.

    Require every shot to carry xG, a fully observed future horizon, no period
    boundary, all four features and nonzero future total xG. Do not impute xG.
    """
    if isinstance(window_minutes, bool) or not isinstance(window_minutes, (int, float)) or not math.isfinite(window_minutes) or not 1 <= window_minutes <= 30:
        raise ValueError('TCI training windows must be between 1 and 30 minutes.')
    matches = defaultdict(list)
    for row in rows:
        matches[row['match_id']].append(row)
    output, skipped = [], defaultdict(int)
    for match_id, match in sorted(matches.items()):
        shots = [e for e in match if e['type'] in {'shot', 'goal'}]
        if not shots or any('xg' not in e for e in shots):
            raise ValueError('Match '+match_id+' needs provider xg on every shot/goal. Match metadata and missing xG cannot train TCI.')
        teams = sorted({e['team'] for e in match})
        periods = defaultdict(list)
        for event in match:
            periods[event['period']].append(event)
        for period, part in sorted(periods.items()):
            # Conservative coverage: stop at the last recorded event. No padded
            # future horizon or information from the next half/extra time.
            first, last = min(e['time_seconds'] for e in part), max(e['time_seconds'] for e in part)
            start = math.ceil(first / (window_minutes*60)) * window_minutes*60
            while start + window_minutes*60 + HORIZON <= last:
                end = start + window_minutes*60
                future = [e for e in part if end <= e['time_seconds'] < end+HORIZON and e['type'] in {'shot', 'goal'}]
                total_xg = sum(e['xg'] for e in future)
                for team in teams:
                    metrics = window_metrics(part, [], team, start, end)['metrics']
                    if any(metrics[key] is None for key in FEATURES):
                        skipped['missing_feature'] += 1
                    elif total_xg <= 0:
                        skipped['no_future_chances'] += 1
                    else:
                        output.append({'match_id': match_id, 'team': team, 'period': period,
                            'feature_start': start, 'feature_end': end, 'target_start': end,
                            'target_end': end+HORIZON, 'x': [metrics[key] for key in FEATURES],
                            'y': sum(e['xg'] for e in future if e['team'] == team)/total_xg})
                start = end
    return output, dict(skipped)


def _project(values):
    ordered = sorted(values, reverse=True)
    cumulative, threshold = 0.0, 0.0
    for i, value in enumerate(ordered, 1):
        cumulative += value
        candidate = (cumulative-1)/i
        if value > candidate:
            threshold = candidate
    return [max(0.0, value-threshold) for value in values]


def _scores(samples, weights):
    # A match, rather than a correlated window, is the scoring unit.
    groups = defaultdict(list)
    for example in samples:
        prediction = sum(a*b for a, b in zip(example['x'], weights))
        groups[example['match_id']].append(prediction-example['y'])
    per_match = {match: {'mse': sum(e*e for e in errors)/len(errors),
                         'mae': sum(abs(e) for e in errors)/len(errors), 'windows': len(errors)}
                 for match, errors in sorted(groups.items())}
    return {'mse': sum(m['mse'] for m in per_match.values())/len(per_match) if per_match else None,
            'mae': sum(m['mae'] for m in per_match.values())/len(per_match) if per_match else None,
            'matches': len(per_match), 'windows': len(samples), 'per_match': per_match}


def fit_tci(rows, window_minutes=15):
    rows = validate_events(rows)
    _check_partition(rows, 'training')
    samples, skipped = examples(rows, window_minutes)
    match_ids = sorted({sample['match_id'] for sample in samples})
    if len(match_ids) < 3 or len(samples) < 6:
        raise ValueError('Training needs at least three independent matches and six eligible team windows with complete xG labels. Supply a Training event dataset; no demonstration weights are fabricated.')
    counts = {match: sum(e['match_id'] == match for e in samples) for match in match_ids}
    # Four-feature sufficient statistics keep optimization independent of the
    # number of windows. Large uploads do not rescan every row 10,000 times.
    gram=[[0.0]*4 for _ in range(4)];cross=[0.0]*4
    for example in samples:
        mass=1/(len(match_ids)*counts[example['match_id']])
        for j in range(4):
            cross[j]+=mass*example['x'][j]*example['y']
            for k in range(4):gram[j][k]+=mass*example['x'][j]*example['x'][k]
    weights = list(WEIGHTS.values())
    for iteration in range(10000):
        gradient = [2*(sum(gram[j][k]*weights[k] for k in range(4))-cross[j]) for j in range(4)]
        updated = _project([w-.2*g for w, g in zip(weights, gradient)])
        if max(abs(a-b) for a, b in zip(weights, updated)) < 1e-11:
            weights = updated
            break
        weights = updated
    artifact = {'schema_version': SCHEMA, 'features': list(FEATURES),
        'weights': dict(zip(FEATURES, weights)), 'window_minutes': window_minutes,
        'horizon_seconds': HORIZON, 'target': 'next_15_minutes_xg_share',
        'optimizer': 'projected_gradient_simplex_match_balanced_mse', 'iterations': iteration+1,
        'training_match_ids': sorted({e['match_id'] for e in rows}),
        'training_data_sha256': fingerprint(rows, []), 'skipped': skipped,
        'training_scores': {'fixed': _scores(samples, list(WEIGHTS.values())), 'learned': _scores(samples, weights)},
        'warning': 'Training fit only. Evaluate disjoint Test matches. Predicting future xG share does not validate tactical control. Validation has not been opened.'}
    artifact['model_id'] = _hash(artifact)
    return artifact


def validate_model(model):
    if not isinstance(model, dict) or model.get('schema_version') != SCHEMA:
        raise ValueError('Upload a learned TCI model produced by Train TCI weights.')
    if model.get('model_id') != _hash({k: v for k, v in model.items() if k != 'model_id'}):
        raise ValueError('The learned TCI artifact fingerprint does not match its contents.')
    weights = model.get('weights', {})
    if set(weights) != set(FEATURES) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError('Learned TCI weights must be nonnegative and sum to one.')
    if model.get('features') != list(FEATURES) or model.get('horizon_seconds') != HORIZON or model.get('target') != 'next_15_minutes_xg_share':
        raise ValueError('Unsupported learned TCI target or features.')
    if not isinstance(model.get('training_match_ids'), list) or len(set(model['training_match_ids'])) < 3:
        raise ValueError('The TCI artifact must record its independent Training matches.')
    _check_partition([{'match_id':match} for match in model['training_match_ids']], 'training')
    window=model.get('window_minutes')
    if isinstance(window,bool) or not isinstance(window,(int,float)) or not math.isfinite(window) or not 1<=window<=30:
        raise ValueError('Invalid learned TCI window length.')
    return model


def evaluate_tci(model, rows):
    validate_model(model)
    rows = validate_events(rows)
    _check_partition(rows, 'test')
    if {e['match_id'] for e in rows} & set(model['training_match_ids']):
        raise ValueError('Training and Test matches overlap. Split whole matches before training.')
    samples, skipped = examples(rows, model['window_minutes'])
    if not samples:
        raise ValueError('Test data has no eligible future-xG windows. Check coverage and labels.')
    return {'model_id': model['model_id'], 'target': model['target'], 'split': 'test',
        'data_sha256': fingerprint(rows, []), 'skipped': skipped,
        'fixed': _scores(samples, list(WEIGHTS.values())),
        'learned': _scores(samples, [model['weights'][k] for k in FEATURES]),
        'warning': 'Disjoint Test comparison, not final unseen Validation. Positive results are not guaranteed.'}
