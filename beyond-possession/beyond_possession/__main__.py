import argparse
import json
import os
from pathlib import Path
from .data import parse_csv, validate_tracking, parse_json_events, demo, write_demo
from .pipeline import analyze, save_report
from .evaluation import run_experiment
from .server import serve
from .agents import OllamaBackend
from .cloud_ai import DEFAULT_PROVIDER, GroqBackend, load_environment
from .gemini_ai import GeminiBackend
from .shared_ai import SharedBackend
from .matches import EXPECTED_FILES, write_catalog
from .dataset_splits import write_match_splits
from .tci_learning import fit_tci, evaluate_tci

def main():
    load_environment()
    parser=argparse.ArgumentParser(description='Beyond Possession: evidence-first soccer analysis')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor',help='Check the configured AI provider and model')
    doctor.add_argument('--backend',choices=['shared','gemini','groq','ollama','offline'],default=os.getenv('AI_PROVIDER',DEFAULT_PROVIDER))
    web=sub.add_parser('serve',help='Launch the local dashboard')
    web.add_argument('--host',default='127.0.0.1');web.add_argument('--port',type=int,default=8765)
    merge=sub.add_parser('combine',help='Combine the seven match JSON files into one catalog')
    merge.add_argument('--input-dir',default='data/raw',help='Folder containing the seven matches_*.json files')
    merge.add_argument('--output',default='data/real/combined_matches.json')
    split=sub.add_parser('split',help='Freeze 70/20/10 Training/Test/Validation match datasets')
    split.add_argument('--catalog',default='data/real/combined_matches.json.gz')
    split.add_argument('--output',default='data/real/splits')
    split.add_argument('--seed',type=int,default=42)
    split.add_argument('--ratios',type=float,nargs=3,default=[.7,.2,.1],metavar=('TRAIN','TEST','VALIDATION'))
    gen=sub.add_parser('demo',help='Write reproducible synthetic events and tracking')
    gen.add_argument('--output',default='data/demo');gen.add_argument('--seed',type=int,default=42)
    for name in ['train-tci','test-tci']:
        p=sub.add_parser(name,help='Learn TCI from Training xG labels' if name=='train-tci' else 'Compare learned and fixed TCI on disjoint Test matches')
        p.add_argument('--events',required=True,help='Normalized multi-match event JSON or CSV with xg on every shot/goal')
        p.add_argument('--output',default='artifacts/tci-model.json' if name=='train-tci' else 'artifacts/tci-test.json')
        if name=='train-tci':p.add_argument('--window',type=float,default=15)
        else:p.add_argument('--model',required=True)
    for command in ['analyze','evaluate']:
        p=sub.add_parser(command)
        p.add_argument('--events',help='Normalized event CSV or JSON; omit for synthetic demo')
        p.add_argument('--tracking',help='Optional normalized tracking JSON')
        p.add_argument('--match-id',help='Select one match from a multi-match event CSV')
        p.add_argument('--backend',choices=['offline','ollama','groq','gemini']+(['shared'] if command=='analyze' else []),default='offline')
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--final-validation',action='store_true',help='Explicitly unlock final unseen matches after development is frozen')
        p.add_argument('--output',default='artifacts/'+command)
        p.add_argument('--writing',choices=['template','free'],default='template')
        p.add_argument('--critic',choices=['numeric','ai','hybrid'],default='numeric')
        p.add_argument('--tci-model',help='Learned TCI model JSON; omit to retain the original weights')
        if command=='analyze':
            p.add_argument('--system',choices=['baseline','single','multi','no_critic','no_game_state','no_specialists'],default='multi')
            p.add_argument('--window',type=float,default=15)
            p.add_argument('--database',help='Optional SQLite history file')
        else:
            p.add_argument('--repeats',type=int,default=2)
            p.add_argument('--compare-approaches',action='store_true',help='Blind comparison of original, AI-only/hybrid critics, free writing and combined approach (single analyst)')
    args=parser.parse_args()
    try:
        if args.command in {'train-tci','test-tci'}:
            path=Path(args.events);text=path.read_text(encoding='utf-8-sig')
            rows=parse_json_events(text) if path.suffix.lower()=='.json' else parse_csv(text)
            result=fit_tci(rows,args.window) if args.command=='train-tci' else evaluate_tci(json.loads(Path(args.model).read_text()),rows)
            output=Path(args.output)
            if output.exists():raise ValueError('Use a new output path to preserve previous TCI artifacts.')
            output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print('Saved '+str(output)+'. Validation was not opened.')
            return
        if args.command=='doctor':
            providers={'shared':SharedBackend,'gemini':GeminiBackend,'groq':GroqBackend,'ollama':OllamaBackend}
            result=providers[args.backend]().check_ready() if args.backend!='offline' else {'available':True,'provider':'Built-in analysis','model':None}
            print(json.dumps(result,indent=2))
            return
        if args.command=='split':
            manifest=write_match_splits(args.catalog,args.output,args.seed,args.ratios)
            print(json.dumps({name:spec['matches'] for name,spec in manifest['splits'].items()},indent=2))
            print('Saved to '+args.output+'. Validation is excluded from development.')
            return
        if args.command=='combine':
            paths=[Path(args.input_dir)/name for name in EXPECTED_FILES]
            missing=[p.name for p in paths if not p.is_file()]
            if missing:raise ValueError('Missing inputs: '+', '.join(missing))
            result=write_catalog(paths,args.output)
            print(json.dumps(result['summary'],indent=2))
            print('Combined catalog saved to '+args.output)
            return
        if args.command=='serve':return serve(args.host,args.port)
        if args.command=='demo':
            write_demo(args.output,args.seed);print(f'Synthetic data saved to {args.output}');return
        if args.tracking and not args.events:raise ValueError('--tracking requires --events.')
        if args.events:
            text=Path(args.events).read_text(encoding='utf-8-sig')
            rows=parse_json_events(text) if Path(args.events).suffix.lower()=='.json' else parse_csv(text)
            if args.match_id: rows=[r for r in rows if r['match_id']==args.match_id]
            if not rows:raise ValueError('No rows for requested match.')
            frames=json.loads(Path(args.tracking).read_text()) if args.tracking else []
        else:rows,frames=demo(args.seed)
        tci_model=json.loads(Path(args.tci_model).read_text()) if args.tci_model else None
        if args.command=='analyze':
            report=analyze(rows,frames,system=args.system,backend=args.backend,seed=args.seed,window_minutes=args.window,final_validation=args.final_validation,
                generation_mode=args.writing,critic_mode=args.critic,tci_model=tci_model)
            save_report(report,args.output,args.database)
            print(f"Report saved to {args.output}; {len(report['analysis']['claims'])} observations.")
        else:
            result=run_experiment(rows,frames,args.output,args.backend,args.repeats,args.seed,final_validation=args.final_validation,
                generation_mode=args.writing,critic_mode=args.critic,tci_model=tci_model,compare_approaches=args.compare_approaches)
            failures=sum(r['status']=='error' for r in result['records'])
            print(f'Experiment saved to {args.output}; {failures} failed runs. Human scores are intentionally blank.')
            if failures:raise SystemExit(1)
    except (ValueError,OSError,RuntimeError) as exc:parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
