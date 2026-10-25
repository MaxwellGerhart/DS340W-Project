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

def analyze(rows, tracking=None, *, system='multi', backend='offline', seed=42, window_minutes=15, weights=None, model_backend=None):
    rows=validate_events(rows)
    tracking=validate_tracking(tracking or [],rows)
    summary=summarize(rows,tracking,window_minutes,weights)
    result=run_agents(summary,system,backend,seed,model_backend)
    sha=fingerprint(rows,tracking)
    config={'system':system,'backend':backend,'seed':seed,'window_minutes':window_minutes,
            'weights':summary['tci_weights'],'model':result['model'],'version':__version__}
    return {'schema_version':'1.0','created_at':datetime.now(timezone.utc).isoformat(),
        'run_id':hashlib.sha256((sha+json.dumps(config,sort_keys=True)).encode()).hexdigest()[:16],
        'data_sha256':sha,'config':config,'summary':summary,'analysis':result,
        'limitations':['Synthetic matches demonstrate software behavior only; they cannot establish research findings.',
          'Action share is not time-based possession. TCI is an unvalidated event proxy.',
          'Tracking estimates use nearest-player geometry, not calibrated arrival-time pitch control.',
          'No confidence intervals are computed: adjacent observations are dependent.',
          'No causal claims, inferred fatigue, or passing-lane availability are supported.',
          'Automatic checks measure structural/numeric faithfulness; experts must assess tactical validity.']}

def markdown(report):
    summary,analysis=report['summary'],report['analysis']
    lines=['# Beyond Possession: tactical report','',f"Match: {summary['match_id']}",
        f"System: {analysis['system']} | Backend: {analysis['backend']} | Model: {analysis['model'] or 'none'}",
        f"Run: {report['run_id']} | Data SHA-256: {report['data_sha256']}",'',
        '## Evidence-backed observations','']
    for claim in analysis['claims']:
        e=claim['evidence']
        lines.extend([f"- **[{claim['verification']}]** {claim['text']}",
            f"  Evidence: `{e['evidence_id']}`; {e['sample_count']} {e['sample_unit']}; examples: {', '.join(e['supporting_events']) or 'none'}.",
            f"  Alternative: {claim['alternative_explanation']}"])
        if 'context' in e:
            lines.append(f"  Score at interval start (team-opponent): {e['context']['score_at_start']}; changes: {json.dumps(e['context']['changes'])}.")
    if summary.get('patterns'):
        lines.extend(['','## Descriptive interval comparisons',''])
        for pattern in summary['patterns']:
            lines.extend(['- '+pattern['text'],'  Evidence: '+', '.join(pattern['evidence_ids'])+'. '+pattern['warning']])
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
