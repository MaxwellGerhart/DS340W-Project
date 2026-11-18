"""Past-only, conditional team outlooks and cited coaching hypotheses.

The projection is a transparent trend baseline, not a trained match-outcome
model. AI can explain the evidence and prioritize actions; it cannot invent
new forecasts, odds, scorelines or measured coaching effects.
"""
import copy
import math
from datetime import datetime, timezone

from .agents import catalog
from .analytics import WEIGHTS
from .narrative import NUMBER, WORDS

CORE = tuple(WEIGHTS)
PRIORITIES = {
    'progression': {'title': 'Create better forward options',
        'match_action': 'Try supporting runs around the ball carrier so a forward option is available before choosing another safe pass.',
        'practice_action': 'Rehearse receiving on the half-turn and combinations that move the ball into the next area of the pitch.',
        'watch_for': 'Watch progressive passing alongside completion, rather than rewarding extra passes alone.'},
    'ball_security': {'title': 'Improve ball security under pressure',
        'match_action': 'Offer closer supporting options and check whether rushed forward passes are losing the ball.',
        'practice_action': 'Use possession exercises that ask the receiver to scan, choose a support option and then look forward.',
        'watch_for': 'Watch completion and progression together to see whether safer play also moves the attack forward.'},
    'chance_creation': {'title': 'Turn attacking territory into attempts',
        'match_action': 'Test more varied final-third combinations instead of treating time near the opponent goal as an end in itself.',
        'practice_action': 'Rehearse combinations and cutback decisions in finishing exercises, then review the resulting attempts.',
        'watch_for': 'Watch shot share together with territory. Shot quality still needs separate xG or video evidence.'},
    'passing_variety': {'title': 'Develop alternative passing outlets',
        'match_action': 'Check whether another receiving option can share the build-up responsibility.',
        'practice_action': 'Rotate the receiving player in passing exercises so the attack can use several routes.',
        'watch_for': 'Watch receiving concentration alongside progression; concentration alone does not prove predictability.'},
    'pressing': {'title': 'Check the outcome of pressing',
        'match_action': 'Review whether pressure attempts lead to useful recoveries before committing more players forward.',
        'practice_action': 'Rehearse shared pressing triggers and the supporting positions behind the press.',
        'watch_for': 'Watch pressure frequency and high recoveries together. Pressure frequency is not pressing success.'},
    'central_space': {'title': 'Review access to central space',
        'match_action': 'Use the tracking view to check support positions around central attacking areas.',
        'practice_action': 'Rehearse compact support and coordinated movement through central areas.',
        'watch_for': 'Watch central control with progression. The spatial measure is a geometric proxy.'},
    'territory': {'title': 'Build sustained attacking territory',
        'match_action': 'Review the build-up route and support around the receiver before trying to push more actions forward.',
        'practice_action': 'Rehearse moving from build-up into attacking areas while retaining supporting options.',
        'watch_for': 'Watch territory, progression and ball security together.'},
    'balance': {'title': 'Keep productive play balanced',
        'match_action': 'Keep the useful passing options while checking the support left behind attacking moves.',
        'practice_action': 'Rehearse possession transitions and review whether forward play retains useful support.',
        'watch_for': 'Watch whether territory and progression stay productive without losing completion.'},
}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _project(values):
    level = sum(value*(i+1) for i, value in enumerate(values))/sum(range(1, len(values)+1))
    trend = .5*(values[-1]-values[0])/(len(values)-1)
    return max(0.0, min(1.0, level+trend))


def team_outlook(summary, team, interval_start):
    if team not in summary['teams']:
        raise ValueError('Choose one of the analyzed teams.')
    if not _finite(interval_start):
        raise ValueError('Choose a valid analyzed interval.')
    selected = next((w for w in summary['windows'] if w['team']==team and w['start']==interval_start), None)
    if selected is None:
        raise ValueError('This team/interval is not part of the analyzed match.')
    step = summary['window_minutes']*60
    eligible = [w for w in summary['windows'] if w['team']==team and w['end']<=selected['end']
                and w['end']-w['start']>=.99*step and w['n_events']>=10]
    eligible.sort(key=lambda w:w['start'])
    history = []
    for window in reversed(eligible):
        if history and not math.isclose(window['end'],history[-1]['start'],abs_tol=.01):break
        history.append(window)
        if len(history)==3:break
    history.reverse()
    base = history[-1] if history else selected
    windows_by_identity = {id(w):i for i,w in enumerate(summary['windows'])}
    full_catalog = catalog(summary)
    ref = lambda w,metric:f"w{windows_by_identity[id(w)]}:{metric}"
    # Explicit filtering before any AI input. No future windows/claims/context
    # or end-of-match duration are sent to the outlook model.
    allowed = {ref(w,k) for w in history for k,v in w['metrics'].items() if v is not None}
    evidence = {key:item for key,item in full_catalog.items() if key in allowed}
    projections=[]
    if len(history)>=2:
        for metric in CORE:
            values=[w['metrics'].get(metric) for w in history]
            enough_passes=metric not in {'completion_rate','progressive_pass_rate'} or all(w['n_passes']>=10 for w in history)
            if not enough_passes or not all(_finite(v) for v in values):continue
            estimate=_project(values)
            projections.append({'metric':metric,'observed':values[-1],'estimate':estimate,
                'recent_min':min(values),'recent_max':max(values),
                'direction':'rising' if estimate-values[-1]>.02 else 'falling' if estimate-values[-1]<-.02 else 'steady',
                'evidence_ids':[ref(w,metric) for w in history]})
        by_metric={p['metric']:p for p in projections}
        if all(k in by_metric for k in CORE):
            weights=summary['tci_weights']
            estimate=100*sum(weights[k]*by_metric[k]['estimate'] for k in CORE)
            observed=base['metrics'].get('tci')
            projections.insert(0,{'metric':'tci','observed':observed,'estimate':estimate,
                'recent_min':min(w['metrics']['tci'] for w in history),'recent_max':max(w['metrics']['tci'] for w in history),
                'direction':'rising' if estimate-observed>2 else 'falling' if estimate-observed<-2 else 'steady',
                'evidence_ids':list(dict.fromkeys(r for p in projections for r in p['evidence_ids']))})
    priorities=[]
    metrics=base['metrics'];previous=history[-2]['metrics'] if len(history)>1 else {}
    def add(key, score, metric_names, signal):
        refs=[ref(base,k) for k in metric_names if ref(base,k) in evidence]
        if not refs:return
        priorities.append({'priority_id':key,'rank_score':score,'signal':signal,
            'evidence_ids':refs,**PRIORITIES[key]})
    progressive,completion=metrics.get('progressive_pass_rate'),metrics.get('completion_rate')
    if _finite(progressive) and base['n_passes']>=10:
        falling=_finite(previous.get('progressive_pass_rate')) and progressive<previous['progressive_pass_rate']-.03
        if progressive<.25 or falling:
            add('progression',4+max(0,.25-progressive),'progressive_pass_rate completion_rate'.split(),
                'Forward progression is limited or has fallen in the recent complete interval.')
    if _finite(completion) and base['n_passes']>=10:
        falling=_finite(previous.get('completion_rate')) and completion<previous['completion_rate']-.05
        if completion<.8 or falling:
            add('ball_security',4+max(0,.8-completion),['completion_rate','progressive_pass_rate'],
                'Pass completion is limited or has fallen relative to the preceding interval.')
    territory,shots=metrics.get('territory_share'),metrics.get('shot_share')
    if _finite(shots) and shots<.45:
        add('chance_creation',4+(.45-shots),['shot_share','territory_share'],
            'The team has a smaller share of recorded attempts; territory alone does not establish chance quality.')
    if _finite(territory) and territory<.45:
        add('territory',3+(.45-territory),['territory_share','progressive_pass_rate'],
            'The team has a smaller share of recorded attacking-third actions.')
    if _finite(metrics.get('hub_share')) and metrics['hub_share']>.35:
        add('passing_variety',3+metrics['hub_share'],['hub_share','network_density'],
            'Completed receptions are concentrated on a receiving hub.')
    if _finite(metrics.get('pressure_per100')):
        add('pressing',2,['pressure_per100','recovery_high_rate'],
            'Pressure frequency is observed, but these metrics do not by themselves establish pressing success.')
    if base.get('n_frames',0)>=3 and _finite(metrics.get('central_control')) and metrics['central_control']<.45:
        add('central_space',3.5,['central_control','spatial_control'],
            'The distance-based proxy shows limited control of central attacking space.')
    add('balance',1,['completion_rate','progressive_pass_rate','territory_share'],
        'Keep effective parts of the current approach while checking progression and support together.')
    priorities.sort(key=lambda p:(-p['rank_score'],p['priority_id']))
    priorities=priorities[:4]
    keep={r for p in projections for r in p['evidence_ids']}|{r for p in priorities for r in p['evidence_ids']}
    evidence={ref:e for ref,e in evidence.items() if ref in keep}
    warnings=['This is an unvalidated trend scenario, not a win probability or a promised scoreline.',
              'Recent ranges show observed variation, not prediction intervals.',
              'Suggested actions are coaching hypotheses to test, not measured treatment effects.']
    if selected['end']!=base['end']:warnings.append('The selected partial or sparse interval is excluded; the outlook stops at the latest eligible complete interval.')
    if len(history)<2:warnings.append('At least two adjacent complete intervals with sufficient events are needed for numeric projections.')
    return {'schema_version':'team-outlook/1','team':team,'selected_interval_start':interval_start,
        'as_of_seconds':base['end'],'selection_end':selected['end'],'horizon_minutes':summary['window_minutes'],
        'horizon_label':'Next comparable playing phase, if similar patterns continue',
        'method':'Recency-weighted mean plus a half-strength recent trend; fractions clipped to their valid range. TCI uses the selected component weights.',
        'history_intervals':len(history),'history_start':history[0]['start'] if history else None,
        'status':'ready' if projections else 'insufficient_history','projections':projections,
        'priorities':priorities,'evidence':list(evidence.values()),'warnings':warnings,'ai':None}


def outlooks(summary):
    return [team_outlook(summary,w['team'],w['start']) for w in summary['windows']]


def ai_payload(outlook):
    evidence=[]
    for item in outlook['evidence']:
        record={k:item[k] for k in ['evidence_id','team','start','end','metric','value']}
        evidence.append(record)
    return {'role':'manager_analyst','team':outlook['team'],'as_of_seconds':outlook['as_of_seconds'],
        'horizon_minutes':outlook['horizon_minutes'],'evidence':evidence,
        'directions':[{'metric':p['metric'],'direction':p['direction']} for p in outlook['projections']],
        'priorities':[{k:p[k] for k in ['priority_id','title','signal','evidence_ids','match_action','practice_action','watch_for']} for p in outlook['priorities']]}


def validate_ai(output, outlook):
    if not isinstance(output,dict) or not isinstance(output.get('summary'),str):
        raise ValueError('The outlook model returned no summary.')
    allowed={e['evidence_id'] for e in outlook['evidence']}
    candidates={p['priority_id']:p for p in outlook['priorities']}
    def text(value):
        if not isinstance(value,str) or not 1<=len(value.strip())<=600:raise ValueError('An outlook explanation is empty or too long.')
        clean=value.replace(outlook['team'],'')
        if NUMBER.search(clean) or WORDS.search(clean):raise ValueError('AI outlook prose introduced numeric claims. Calculated values are displayed separately.')
        if any(term in clean.casefold() for term in ['guarantee','definitely win','will win','certain victory','injur','fatigu']):
            raise ValueError('The outlook contains an unsupported guarantee or player-health claim.')
        return value.strip()
    summary=text(output['summary'])
    if not any(term in summary.lower().split() for term in ['may','could','if','might','likely']):
        raise ValueError('The outlook must describe a conditional scenario.')
    refs=output.get('evidence_ids')
    if not isinstance(refs,list) or not 1<=len(refs)<=8 or any(not isinstance(ref,str) or ref not in allowed for ref in refs):
        raise ValueError('The outlook cites unknown or missing evidence.')
    recommendations=output.get('recommendations')
    if not isinstance(recommendations,list) or not 1<=len(recommendations)<=3:
        raise ValueError('The outlook must recommend a bounded set of supplied priorities.')
    seen=set();checked=[]
    for recommendation in recommendations:
        if not isinstance(recommendation,dict) or recommendation.get('priority_id') not in candidates or recommendation['priority_id'] in seen:
            raise ValueError('The outlook invented or repeated a coaching priority.')
        key=recommendation['priority_id'];seen.add(key)
        checked.append({**candidates[key],**{field:text(recommendation.get(field)) for field in ['rationale','match_action','practice_action','watch_for']}})
    return {'summary':summary,'evidence_ids':refs,'recommendations':checked,
        'verification':'Citations and output constraints checked; coaching reasoning needs human review.',
        'created_at':datetime.now(timezone.utc).isoformat()}


def explain_outlook(outlook, model, seed=42):
    if outlook['status']!='ready' or not outlook['priorities']:
        raise ValueError('More complete match intervals are needed before requesting an AI outlook.')
    start=len(getattr(model,'calls',[]))
    output=model.generate_task('outlook',ai_payload(outlook),seed)
    result=copy.deepcopy(outlook)
    result['ai']={**validate_ai(output,outlook),'model':getattr(model,'model',None),
                  'model_calls':copy.deepcopy(getattr(model,'calls',[])[start:])}
    return result
