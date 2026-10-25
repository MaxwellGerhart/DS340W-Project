"""Deterministic metrics. No language model computes statistics."""
import math
import statistics
from collections import Counter, defaultdict
from .data import ACTIONS

WEIGHTS = {'territory_share': .30, 'progressive_pass_rate': .25,
           'completion_rate': .20, 'shot_share': .25}
METRIC_DEFS = {
 'action_share': 'Share of recorded on-ball actions; NOT possession time.',
 'territory_share': 'Share of both teams\' on-ball actions starting at x >= 66.667 in their attacking coordinates.',
 'progressive_pass_rate': 'Completed passes gaining at least 10 normalized x units / all attempted passes.',
 'completion_rate': 'Completed passes / all attempted passes.',
 'shot_share': 'Share of shots plus goals; goal events must not duplicate shot events.',
 'tci': 'Provisional weighted event Tactical Control Index, 0 to 100; not a validated model.',
 'network_density': 'Distinct completed directed pass edges / n(n-1), using observed passer/receiver nodes.',
 'hub_share': 'Largest share of completed passes received by one player.',
 'pressure_per100': 'Pressure events / opponent pass attempts, multiplied by 100; NOT PPDA.',
 'recovery_high_rate': 'Recoveries at x >= 66.667 / all recoveries.',
 'spatial_control': 'Fraction of grid points nearest to this team, averaged across full snapshots; NOT calibrated pitch control.',
 'central_control': 'Nearest-player control in central attacking third: x >= 66.667, y in [33.333,66.667].',
 'defensive_width_m': 'Mean lateral range of outfield players in meters across snapshots; goalkeeper excluded by nearest-own-goal heuristic.',
 'defensive_depth_m': 'Mean longitudinal range of outfield players in meters across snapshots; goalkeeper excluded by nearest-own-goal heuristic.'}

def ratio(a,b):
    return a/b if b else None

def control_surface(frame, team, nx=18, ny=12):
    own=[p for p in frame['players'] if p['team']==team]
    other=[p for p in frame['players'] if p['team']!=team]
    surface=[]
    for iy in range(ny):
        for ix in range(nx):
            x,y=(ix+.5)*100/nx,(iy+.5)*100/ny
            distance=lambda p: ((p['x']-x)*1.05)**2+((p['y']-y)*.68)**2
            da,db=min(map(distance,own)),min(map(distance,other))
            value=0.5 if math.isclose(da,db,abs_tol=1e-8) else float(da<db)
            surface.append({'x':x,'y':y,'control':value})
    return surface

def spatial(frames, team):
    if not frames:
        return {}, []
    overall, central, widths, depths=[],[],[],[]
    for frame in frames:
        grid=control_surface(frame,team)
        attacking_x=lambda c: c['x'] if team==frame['home_team'] else 100-c['x']
        overall.append(statistics.mean(c['control'] for c in grid))
        central.append(statistics.mean(c['control'] for c in grid if attacking_x(c)>=100*2/3 and 100/3<=c['y']<=200/3))
        own=[p for p in frame['players'] if p['team']==team]
        goalie=min(own,key=lambda p:p['x']) if team==frame['home_team'] else max(own,key=lambda p:p['x'])
        field=[p for p in own if p is not goalie]
        widths.append((max(p['y'] for p in field)-min(p['y'] for p in field))*.68)
        depths.append((max(p['x'] for p in field)-min(p['x'] for p in field))*1.05)
    return {'spatial_control':statistics.mean(overall),'central_control':statistics.mean(central),
            'defensive_width_m':statistics.mean(widths),'defensive_depth_m':statistics.mean(depths)}, control_surface(frames[len(frames)//2],team)

def network(rows):
    edges=Counter((e['player'],e['receiver']) for e in rows if e['type']=='pass' and e['outcome']=='success' and e['player']!=e['receiver'])
    nodes=sorted(set(p for edge in edges for p in edge))
    incoming=Counter()
    positions=defaultdict(list)
    for e in rows:
        if e['type']=='pass' and e['outcome']=='success':
            incoming[e['receiver']]+=1
            positions[e['player']].append((e['x'],e['y']))
            positions[e['receiver']].append((e['end_x'],e['end_y']))
    return {'nodes':[{'id':n,'x':statistics.mean(x for x,y in positions[n]),
                     'y':statistics.mean(y for x,y in positions[n]),'received':incoming[n]} for n in nodes],
            'edges':[{'source':a,'target':b,'count':count} for (a,b),count in sorted(edges.items())],
            'density':ratio(len(edges),len(nodes)*(len(nodes)-1)),
            'hub_share':ratio(max(incoming.values(),default=0),sum(incoming.values()))}

def window_metrics(rows, frames, team, start, end, weights=None):
    selected=[e for e in rows if start<=e['time_seconds']<end]
    own=[e for e in selected if e['team']==team]
    opponent=[e for e in selected if e['team']!=team]
    passes=[e for e in own if e['type']=='pass']
    completed=[e for e in passes if e['outcome']=='success']
    actions=[e for e in own if e['type'] in ACTIONS]
    all_actions=[e for e in selected if e['type'] in ACTIONS]
    territory=[e for e in all_actions if e['x']>=200/3]
    shots=[e for e in selected if e['type'] in {'shot','goal'}]
    recoveries=[e for e in own if e['type']=='recovery']
    net=network(own)
    metrics={'action_share':ratio(len(actions),len(all_actions)),
             'territory_share':ratio(sum(e['team']==team for e in territory),len(territory)),
             'progressive_pass_rate':ratio(sum(e['end_x']-e['x']>=10 for e in completed),len(passes)),
             'completion_rate':ratio(len(completed),len(passes)),
             'shot_share':ratio(sum(e['team']==team for e in shots),len(shots)),
             'network_density':net['density'],'hub_share':net['hub_share'],
             'pressure_per100':None,'recovery_high_rate':ratio(sum(e['x']>=200/3 for e in recoveries),len(recoveries))}
    pressures=sum(e['type']=='pressure' for e in own)
    opp_passes=sum(e['type']=='pass' for e in opponent)
    pressure_observed=any(e['type']=='pressure' for e in rows)
    metrics['pressure_per100']=100*pressures/opp_passes if opp_passes and pressure_observed else None
    # Missing any component means no TCI: do not silently impute/reweight.
    use_weights=weights or WEIGHTS
    if set(use_weights)!=set(WEIGHTS) or any(not math.isfinite(w) or w<0 for w in use_weights.values()) or not math.isclose(sum(use_weights.values()),1):
        raise ValueError('TCI weights must be finite, nonnegative, sum to one, and include the four documented components.')
    metrics['tci']=100*sum(metrics[k]*w for k,w in use_weights.items()) if all(metrics[k] is not None for k in use_weights) else None
    snapshots=[f for f in frames if start<=f['time_seconds']<end]
    extra,surface=spatial(snapshots,team)
    metrics.update(extra)
    return {'team':team,'start':start,'end':end,'metrics':metrics,'network':net,
            'surface':surface,'snapshot_time':snapshots[len(snapshots)//2]['time_seconds'] if snapshots else None,
            'n_events':len(own),'n_passes':len(passes),'n_frames':len(snapshots),
            'supporting_events':[e['event_id'] for e in own[:12]],
            'spatial_evidence':[f"frame:{f['time_seconds']:g}" for f in snapshots[:12]],
            'contexts':context(rows,team,start,end)}

def context(rows,team,start,end):
    before=[e for e in rows if e['time_seconds']<start]
    score_for=sum(e['type']=='goal' and e['team']==team for e in before)
    score_against=sum(e['type']=='goal' and e['team']!=team for e in before)
    cards_for=sum(e['type']=='red_card' and e['team']==team for e in before)
    cards_against=sum(e['type']=='red_card' and e['team']!=team for e in before)
    changes=[{'event_id':e['event_id'],'time_seconds':e['time_seconds'],'type':e['type'],'team':e['team']}
             for e in rows if start<=e['time_seconds']<end and e['type'] in {'goal','substitution','red_card'}]
    return {'score_at_start':[score_for,score_against],'red_cards_at_start':[cards_for,cards_against],
            'changes':changes,'note':'Score changes and substitutions provide context, not causal proof.'}

def summarize(rows, frames=None, window_minutes=15, weights=None):
    if not isinstance(window_minutes,(int,float)) or not math.isfinite(window_minutes) or not 1<=window_minutes<=120:
        raise ValueError('Window length must be between 1 and 120 minutes.')
    if len({e['match_id'] for e in rows})!=1:
        raise ValueError('Analyze exactly one match per run.')
    frames=frames or []
    teams=sorted({e['team'] for e in rows})
    duration=max(e['time_seconds'] for e in rows)+.001
    step=window_minutes*60
    windows=[]
    for i in range(math.ceil(duration/step)):
        for team in teams:
            windows.append(window_metrics(rows,frames,team,i*step,min((i+1)*step,duration),weights))
    # Descriptive adjacent-window differences, not statistical significance tests.
    changes=[]
    for team in teams:
        own=[w for w in windows if w['team']==team]
        for a,b in zip(own,own[1:]):
            av,bv=a['metrics']['tci'],b['metrics']['tci']
            if av is not None and bv is not None and abs(bv-av)>=8:
                changes.append({'team':team,'time_seconds':b['start'],'metric':'tci','difference':bv-av,
                                'method':'adjacent-window threshold (8 points), exploratory'})
    patterns=[]
    for team in teams:
        own=[(i,w) for i,w in enumerate(windows) if w['team']==team]
        for (ai,a),(bi,b) in zip(own,own[1:]):
            ma,mb=a['metrics'],b['metrics']
            if a['n_passes']<10 or b['n_passes']<10:continue
            if any(m.get(k) is None for m in [ma,mb] for k in ['action_share','tci','progressive_pass_rate']):continue
            volume=mb['action_share']-ma['action_share']
            delta=mb['tci']-ma['tci']
            progression=mb['progressive_pass_rate']-ma['progressive_pass_rate']
            if volume>=.03 and (delta<=-3 or progression<=-.05):
                patterns.append({'team':team,'start':b['start'],'end':b['end'],
                    'kind':'volume_progression_divergence',
                    'text':f"{team}: action share rose by {volume*100:.1f} percentage points while progressive-pass rate changed by {progression*100:+.1f} points and TCI changed by {delta:+.1f} points versus the preceding interval. More recorded action did not necessarily coincide with greater tactical control by these proxies.",
                    'evidence_ids':[f'w{ai}:action_share',f'w{bi}:action_share',f'w{ai}:progressive_pass_rate',f'w{bi}:progressive_pass_rate',f'w{ai}:tci',f'w{bi}:tci'],
                    'warning':'Descriptive comparison, not a causal claim or a statistical significance test.'})
    return {'match_id':rows[0]['match_id'],'teams':teams,'duration_seconds':duration,
            'window_minutes':window_minutes,'windows':windows,'change_points':changes,'patterns':patterns,
            'tci_weights':weights or WEIGHTS,'metric_definitions':METRIC_DEFS,
            'tracking_available':bool(frames)}
