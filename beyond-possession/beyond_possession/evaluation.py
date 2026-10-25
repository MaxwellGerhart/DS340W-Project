"""Report-level checks and blinded review packets, not fabricated research scores."""
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from .agents import CONFIGS
from .pipeline import analyze, markdown

def automatic_scores(report):
    claims=report['analysis']['claims']
    supported=sum(c['evidence']['supporting_events']!=[] for c in claims)
    faithful=sum(math.isclose(c['value'],c['evidence']['value'],abs_tol=1e-8,rel_tol=1e-9) for c in claims)
    categories={c['agent'] for c in claims}
    unique={c['evidence']['evidence_id'] for c in claims}
    # No automatic tactical validity or contradiction score is invented.
    return {'claim_count':len(claims),'numerical_faithfulness':faithful/len(claims) if claims else None,
            'reference_coverage':supported/len(claims) if claims else None,
            'unique_evidence_count':len(unique),'agent_role_count':len(categories),
            'rejected_candidates':len(report['analysis']['rejected']),
            'latency_seconds':report['analysis']['latency_seconds'],
            'input_tokens':sum(c.get('input_tokens') or 0 for c in report['analysis']['model_calls']),
            'output_tokens':sum(c.get('output_tokens') or 0 for c in report['analysis']['model_calls']),
            'tactical_validity':None,'contradiction_rate':None,'cost_usd':None}

def run_experiment(rows,tracking,directory,backend='offline',repeats=2,seed=42,systems=None,model_backend_factory=None):
    if not 1<=repeats<=20:raise ValueError('Repeats must be between 1 and 20.')
    systems=systems or CONFIGS
    if any(s not in CONFIGS for s in systems):raise ValueError('Unknown experimental system.')
    target=Path(directory)
    if target.exists() and any(target.iterdir()):raise ValueError('Use a new or empty experiment directory to preserve previous runs.')
    target.mkdir(parents=True,exist_ok=True)
    review=target/'blinded';review.mkdir()
    private=target/'private';private.mkdir()
    records=[];mapping={};review_rows=[];sets={}
    rng=random.Random(seed)
    jobs=[(repeat,system) for repeat in range(repeats) for system in systems]
    rng.shuffle(jobs)
    for index,(repeat,system) in enumerate(jobs,1):
        code=f'R{index:03}'
        config={'system':system,'repeat':repeat,'backend':backend,'seed':seed+repeat}
        try:
            model=model_backend_factory() if model_backend_factory and system!='baseline' else None
            report=analyze(rows,tracking,system=system,backend=backend,seed=seed+repeat,model_backend=model)
            (private/f'{code}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            mapping[code]={**config,'data_sha256':report['data_sha256'],'run_id':report['run_id']}
            # No system name, backend, role, hash, latency, or critic label in the reviewer packet.
            content=['# Match analysis',f"Match: {report['summary']['match_id']}",'', '## Observations','']
            for claim in report['analysis']['claims']:
                content.extend(['- '+claim['text'],'  Alternative: '+str(claim['alternative_explanation'])])
            content.extend(['','## Shared method caveats','',
                'All metrics are descriptive. Event action share is not possession time. Spatial results, when available, use nearest-player geometry. No causal validity is assumed.'])
            (review/f'{code}.md').write_text('\n'.join(content)+'\n')
            scores=automatic_scores(report)
            records.append({'review_id':code,**config,'status':'ok',**scores})
            sets.setdefault(system,[]).append({c['evidence']['evidence_id'] for c in report['analysis']['claims']})
            for reviewer in ['reviewer_1','reviewer_2','reviewer_3']:
                review_rows.append({'review_id':code,'reviewer_id':reviewer,'tactical_validity_1to5':'',
                    'usefulness_1to5':'','insight_diversity_1to5':'','contradiction_count':'','notes':''})
        except Exception as exc:
            records.append({'review_id':code,**config,'status':'error','error':str(exc)})
    stability={}
    for system,observations in sets.items():
        pairs=[(a,b) for i,a in enumerate(observations) for b in observations[i+1:]]
        stability[system]={'mean_evidence_jaccard':sum(len(a&b)/len(a|b) if a|b else 1 for a,b in pairs)/len(pairs) if pairs else None,
                           'successful_runs':len(observations)}
    result={'backend':backend,'records':records,'reproducibility':stability,
        'warning':'Workflow checks only when backend=offline. Rates do not establish tactical usefulness. Human fields remain blank. Blinding can still be inferred from report style/length.',
        'required_next_steps':['Use a documented held-out match set.','Fix prompts, model, hardware and token budgets before comparing systems.',
          'Use multiple matches as the sampling unit for uncertainty.','Have experts complete ratings without access to private mapping.',
          'Analyze tradeoffs in cost, latency and quality; report failures.']}
    (target/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (private/'review_key.json').write_text(json.dumps(mapping,indent=2))
    if review_rows:
        with (review/'ratings.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(review_rows[0]));writer.writeheader();writer.writerows(review_rows)
    return result
