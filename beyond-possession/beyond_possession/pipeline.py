import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from . import __version__
from .analytics import summarize
from .agents import run_agents
from .data import fingerprint, validate_events, validate_tracking
from .dataset_splits import check_match_access
from .outlook import outlooks

def analyze(rows, tracking=None, *, system='multi', backend='offline', seed=42, window_minutes=15, weights=None, model_backend=None, final_validation=False,
            tci_model=None, generation_mode='template', critic_mode='numeric', critic_model_backend=None):
    rows=validate_events(rows)
    dataset_access=check_match_access(rows,final_validation=final_validation)
    tracking=validate_tracking(tracking or [],rows)
    if tci_model is not None:
        from .tci_learning import validate_model
        validate_model(tci_model)
        if weights is not None:raise ValueError('Choose a learned TCI model or explicit fixed weights, not both.')
        if window_minutes!=tci_model['window_minutes']:raise ValueError('Use the same analysis window as the learned TCI model.')
        weights=tci_model['weights']
    summary=summarize(rows,tracking,window_minutes,weights)
    summary['tci_model']={'kind':'learned','model_id':tci_model['model_id'],'target':tci_model['target'],
        'training_match_ids':tci_model['training_match_ids']} if tci_model else {'kind':'fixed','model_id':None}
    result=run_agents(summary,system,backend,seed,model_backend,generation_mode,critic_mode,critic_model_backend)
    sha=fingerprint(rows,tracking)
    config={'system':system,'backend':backend,'seed':seed,'window_minutes':window_minutes,
            'weights':summary['tci_weights'],'model':result['model'],'version':__version__,
            'spatial_model':summary['spatial_model'],'dataset_access':dataset_access,
            'tci_model':summary['tci_model'],'generation_mode':result['generation_mode'],
            'critic_mode':result['critic_mode'],'critic_model':result['critic_model']}
    return {'schema_version':'1.0','created_at':datetime.now(timezone.utc).isoformat(),
        'run_id':hashlib.sha256((sha+json.dumps(config,sort_keys=True)).encode()).hexdigest()[:16],
        'data_sha256':sha,'config':config,'summary':summary,'analysis':result,'outlooks':outlooks(summary),
        'limitations':['Synthetic matches demonstrate software behavior only; they cannot establish research findings.',
          'Action share is not time-based possession. TCI is an unvalidated event proxy.',
          'Tracking estimates use smooth distance-based geometry with a 3 m transition scale, not calibrated arrival-time pitch control.',
          'No confidence intervals are computed: adjacent observations are dependent.',
          'Team outlooks are conditional, untrained trend scenarios. Coaching suggestions need testing; no calibrated match-result probabilities are provided.',
          'No causal claims, inferred fatigue, or passing-lane availability are supported.',
          'Automatic checks measure structural/numeric faithfulness; experts must assess tactical validity.']}

def markdown(report):
    summary,analysis=report['summary'],report['analysis']
    lines=['# Beyond Possession: tactical report','',f"Match: {summary['match_id']}",
        f"System: {analysis['system']} | Backend: {analysis['backend']} | Model: {analysis['model'] or 'none'}",
        f"Run: {report['run_id']} | Data SHA-256: {report['data_sha256']}",'',
        '## Evidence-backed observations','']
    lines.insert(6,f"Writing: {analysis.get('generation_mode','template')} | Critic: {analysis.get('critic_mode','numeric')} | TCI: {summary.get('tci_model',{}).get('kind','fixed')}")
    for claim in analysis['claims']:
        e=claim['evidence']
        lines.extend([f"- **[{claim['verification']}]** {claim['text']}",
            f"  Evidence: `{e['evidence_id']}`; {e['sample_count']} {e['sample_unit']}; examples: {', '.join(e['supporting_events']) or 'none'}.",
            f"  Alternative: {claim['alternative_explanation']}"])
        if 'context' in e:
            lines.append(f"  Score at interval start (team-opponent): {e['context']['score_at_start']}; changes: {json.dumps(e['context']['changes'])}.")
        if claim.get('ai_review'):
            lines.append('  AI review: '+claim['ai_review']['verdict']+'. '+claim['ai_review']['reason'])
        if len(claim.get('supporting_evidence',[]))>1:
            lines.append('  Additional citations: '+', '.join(item['evidence_id'] for item in claim['supporting_evidence'][1:]))
    if summary.get('patterns'):
        lines.extend(['','## Descriptive interval comparisons',''])
        for pattern in summary['patterns']:
            lines.extend(['- '+pattern['text'],'  Evidence: '+', '.join(pattern['evidence_ids'])+'. '+pattern['warning']])
    available=report.get('outlooks',[])
    latest={}
    outlook_order=lambda o:(o['as_of_seconds'],bool(o.get('ai')),o['selection_end']==o['as_of_seconds'])
    for outlook in available:
        if outlook['status']=='ready' and (outlook['team'] not in latest or outlook_order(outlook)>outlook_order(latest[outlook['team']])):
            latest[outlook['team']]=outlook
    if latest:
        lines.extend(['','## Team outlooks and actions','',
            'Conditional trend scenarios only. No win odds or guaranteed scores. Recent ranges are historical variation, not prediction intervals.',''])
        display=list(latest.values())
        shown={(o['team'],o['selected_interval_start']) for o in display}
        display.extend(o for o in available if o.get('ai') and (o['team'],o['selected_interval_start']) not in shown)
        for outlook in display:
            team=outlook['team']
            lines.append(f"### {team}: data through {outlook['as_of_seconds']/60:g} minutes")
            lines.append(outlook['method'])
            for projection in outlook['projections']:
                from .agents import format_value
                lines.append('- '+projection['metric'].replace('_',' ')+': estimated '+format_value(projection['metric'],projection['estimate'])+'; evidence '+', '.join(projection['evidence_ids']))
            ai=outlook.get('ai')
            if ai:lines.extend(['', 'AI outlook ('+str(ai['model'])+'): '+ai['summary']])
            for priority in (ai['recommendations'] if ai else outlook['priorities'][:3]):
                lines.extend(['- '+priority['title']+': '+priority.get('rationale',priority['signal']),
                    '  Match action: '+priority['match_action'],'  Practice: '+priority['practice_action'],
                    '  Monitor: '+priority['watch_for'],'  Evidence: '+', '.join(priority['evidence_ids'])])
            lines.append('')
    lines.extend(['','## Rejected candidates','',str(len(analysis['rejected'])),'','## Limitations',''])
    lines.extend('- '+s for s in report['limitations'])
    return '\n'.join(lines)+'\n'

def save_report(report, directory, database=None):
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (folder/'report.md').write_text(markdown(report))
    if database:
        path=Path(database);path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, run_id TEXT, created_at TEXT, match_id TEXT, config TEXT, report TEXT)')
            db.execute('INSERT INTO runs (run_id,created_at,match_id,config,report) VALUES (?,?,?,?,?)',
                       (report['run_id'],report['created_at'],report['summary']['match_id'],json.dumps(report['config']),json.dumps(report,allow_nan=False)))
    return folder
