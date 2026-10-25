"""Strict normalized data contract and deterministic synthetic fixtures."""
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path

EVENT_FIELDS = ['match_id', 'event_id', 'period', 'time_seconds', 'team', 'player',
                'receiver', 'type', 'outcome', 'x', 'y', 'end_x', 'end_y']
TYPES = {'pass', 'carry', 'shot', 'goal', 'recovery', 'pressure', 'substitution', 'red_card'}
ACTIONS = {'pass', 'carry', 'shot', 'goal', 'recovery'}

class DataError(ValueError):
    pass

def number(value, field):
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise DataError(f'{field} must be a finite number.') from None
    if not math.isfinite(value):
        raise DataError(f'{field} must be a finite number.')
    return value

def validate_events(rows):
    if not rows:
        raise DataError('The event dataset is empty.')
    if len(rows) > 250000:
        raise DataError('Maximum 250,000 events per upload.')
    seen = set()
    result = []
    for i, raw in enumerate(rows, 2):
        if not isinstance(raw, dict):
            raise DataError(f'Row {i}: expected a record object.')
        missing = set(EVENT_FIELDS) - raw.keys()
        if missing:
            raise DataError(f'Row {i}: missing columns {sorted(missing)}.')
        row = {key: raw[key] for key in EVENT_FIELDS}
        for key in ['match_id', 'event_id', 'team', 'player', 'receiver', 'type', 'outcome']:
            row[key] = str(row[key]).strip()
        if not all(row[k] for k in ['match_id', 'event_id', 'team', 'player']):
            raise DataError(f'Row {i}: match_id, event_id, team and player are required.')
        identity = (row['match_id'], row['event_id'])
        if identity in seen:
            raise DataError(f'Row {i}: duplicate event ID {identity}.')
        seen.add(identity)
        period = number(row['period'], 'period')
        if period not in [1, 2, 3, 4]:
            raise DataError(f'Row {i}: period must be 1, 2, 3, or 4.')
        row['period'] = int(period)
        row['time_seconds'] = number(row['time_seconds'], 'time_seconds')
        if not 0 <= row['time_seconds'] <= 12000:
            raise DataError(f'Row {i}: time_seconds must be between 0 and 12000.')
        if row['type'] not in TYPES or row['outcome'] not in {'success', 'fail', 'na'}:
            raise DataError(f'Row {i}: unknown event type or outcome.')
        for key in ['x', 'y', 'end_x', 'end_y']:
            row[key] = number(row[key], key)
            if not 0 <= row[key] <= 100:
                raise DataError(f'Row {i}: {key} must be between 0 and 100.')
        if row['type'] == 'pass' and row['outcome'] == 'success' and not row['receiver']:
            raise DataError(f'Row {i}: completed passes require a receiver.')
        if row['type'] == 'pass' and row['outcome'] == 'success' and row['receiver'] == row['player']:
            raise DataError(f'Row {i}: a pass cannot be completed to its own actor.')
        result.append(row)
    matches = {}
    for row in result:
        matches.setdefault(row['match_id'], set()).add(row['team'])
    for match, teams in matches.items():
        if len(teams) != 2:
            raise DataError(f'{match}: expected exactly two teams, found {len(teams)}.')
    return sorted(result, key=lambda x: (x['match_id'], x['time_seconds'], x['event_id']))

def parse_csv(text):
    try:
        return validate_events(list(csv.DictReader(io.StringIO(text.lstrip('\ufeff')))))
    except csv.Error as exc:
        raise DataError(str(exc)) from exc


def parse_json_events(text):
    try:
        data=json.loads(text.lstrip('\ufeff'))
    except (ValueError, TypeError) as exc:
        raise DataError('Could not read the event JSON: '+str(exc)) from exc
    if isinstance(data,dict):
        if 'matches' in data and 'events' not in data:
            raise DataError('This is a match catalogue. Choose Match JSON files to load it. Tactical analysis needs event records, not just match details.')
        data=data.get('events',data)
    if not isinstance(data,list):
        raise DataError('Event JSON must be an array of event records or an object with an events array.')
    if data and isinstance(data[0],dict) and 'event_id' not in data[0] and any(k in data[0] for k in ['wyId','teamsData','label']):
        raise DataError('These look like match records. Choose Match JSON files instead. Passes, shots and coordinates must come from a separate events dataset.')
    return validate_events(data)

def validate_tracking(frames, rows):
    """Tracking is normalized to home-attacks-+x coordinates in every period."""
    if not isinstance(frames, list) or len(frames) > 20000:
        raise DataError('Tracking must be a list with at most 20,000 snapshots.')
    teams = {r['team'] for r in rows}
    matches = {r['match_id'] for r in rows}
    if frames and len(matches) != 1:
        raise DataError('Attach tracking to one match at a time.')
    seen = set()
    output = []
    for raw in frames:
        if not isinstance(raw, dict) or raw.get('match_id') not in matches:
            raise DataError('Tracking snapshot has an unknown match_id.')
        home, away = raw.get('home_team'), raw.get('away_team')
        if {home, away} != teams or home == away:
            raise DataError('Each snapshot must name the two event teams as home_team/away_team.')
        stamp = number(raw.get('time_seconds'), 'tracking time_seconds')
        if not 0 <= stamp <= 12000 or stamp in seen:
            raise DataError('Tracking timestamps must be unique and between 0 and 12000.')
        seen.add(stamp)
        players = raw.get('players', [])
        if not isinstance(players, list):
            raise DataError('players must be a list.')
        validated, ids = [], set()
        for p in players:
            if not isinstance(p, dict) or p.get('team') not in teams:
                raise DataError('Tracking player has an unknown team.')
            pid = str(p.get('player', '')).strip()
            identity = (p['team'], pid)
            if not pid or identity in ids:
                raise DataError('Tracking player identifiers must be nonempty and unique per team/frame.')
            ids.add(identity)
            x, y = number(p.get('x'), 'tracking x'), number(p.get('y'), 'tracking y')
            if not 0 <= x <= 100 or not 0 <= y <= 100:
                raise DataError('Tracking coordinates must be in [0, 100].')
            validated.append({'team': p['team'], 'player': pid, 'x': x, 'y': y})
        counts = [sum(p['team'] == t for p in validated) for t in teams]
        if any(n < 7 or n > 11 for n in counts):
            raise DataError('Full-team tracking requires 7 to 11 visible players per team.')
        output.append({'match_id': raw['match_id'], 'time_seconds': stamp,
                       'home_team': home, 'away_team': away, 'players': validated})
    if output and len({(f['home_team'], f['away_team']) for f in output}) != 1:
        raise DataError('Home/away identity must remain fixed across frames.')
    return sorted(output, key=lambda f: f['time_seconds'])

def fingerprint(rows, tracking):
    return hashlib.sha256(json.dumps([rows, tracking], sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def demo(seed=42, match_id='demo-001'):
    """Synthetic match: home action volume rises after 60, progression drops."""
    rng = random.Random(seed)
    home, away = 'Northbridge FC', 'Riverside United'
    events = []
    for i in range(1200):
        time = (i + 0.2) * 4.5
        late = time >= 3600
        team = home if rng.random() < (0.64 if late else 0.53) else away
        player = f'{"N" if team == home else "R"}{rng.randint(2, 11):02}'
        receiver = f'{player[0]}{rng.choice([n for n in range(2, 12) if n != int(player[1:])]):02}'
        typ = rng.choices(['pass','carry','shot','recovery','pressure'], [72,12,3,7,6])[0]
        x = max(1, min(99, rng.gauss(45 if late and team == home else 53, 22)))
        y = rng.uniform(3, 97)
        dx = rng.gauss(4 if late and team == home else 10, 12)
        end_x = max(0, min(100, x + dx))
        end_y = max(0, min(100, y + rng.gauss(0, 14)))
        if typ == 'shot': x, end_x, end_y = rng.uniform(73,99),100,50
        events.append(dict(zip(EVENT_FIELDS, [match_id, f'e{i:05}', 1 if time < 2700 else 2,
                      round(time,2), team, player, receiver if typ == 'pass' else '', typ,
                      'success' if rng.random() < .81 else 'fail', round(x,3),round(y,3),round(end_x,3),round(end_y,3)])))
    for i, (time, team, typ) in enumerate([(1830,home,'goal'),(3610,home,'substitution'),(4260,away,'goal')]):
        events.append(dict(zip(EVENT_FIELDS,[match_id,f'context-{i}',1 if time < 2700 else 2,time,
                      team, 'N09' if team==home else 'R09','',typ,'success',90,50,100,50])))
    frames = []
    for stamp in range(0, 5400, 30):
        players=[]
        for team in [home,away]:
            for j in range(11):
                base_x = 7 if j==0 else 22 + (j-1)//4 * 24
                x = base_x + rng.gauss(0,3) + (7 if stamp>=3600 and team==away else 0)
                y = 50 if j==0 else 10 + ((j-1)%4)*26 + rng.gauss(0,3)
                if team==away: x=100-x
                players.append({'team':team,'player':f'{"N" if team==home else "R"}{j+1:02}',
                                'x':round(max(0,min(100,x)),3),'y':round(max(0,min(100,y)),3)})
        frames.append({'match_id':match_id,'time_seconds':stamp,'home_team':home,'away_team':away,'players':players})
    return validate_events(events), frames

def write_demo(directory, seed=42, match_id='demo-001'):
    target=Path(directory);target.mkdir(parents=True,exist_ok=True)
    rows, frames = demo(seed, match_id)
    with (target/'events.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=EVENT_FIELDS);writer.writeheader();writer.writerows(rows)
    (target/'tracking.json').write_text(json.dumps(frames,indent=2)+'\n')
    (target/'metadata.json').write_text(json.dumps({'match_id':match_id,'synthetic':True,'seed':seed,
       'event_coordinates':'possession-oriented, attacking +x, scale 0..100',
       'tracking_coordinates':'home-attacks-+x throughout, scale 0..100',
       'warning':'Generated fixtures, not real match observations.'},indent=2)+'\n')
