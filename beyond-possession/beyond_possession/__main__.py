import argparse
import json
from pathlib import Path
from .data import parse_csv, validate_tracking, parse_json_events, demo, write_demo
from .pipeline import analyze, save_report
from .evaluation import run_experiment
from .server import serve
from .agents import OllamaBackend
from .cloud_ai import load_environment
from .matches import EXPECTED_FILES, write_catalog

def main():
    load_environment()
    parser=argparse.ArgumentParser(description='Beyond Possession: evidence-first soccer analysis')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Check the Ollama service and configured model')
    web=sub.add_parser('serve',help='Launch the local dashboard')
    web.add_argument('--host',default='127.0.0.1');web.add_argument('--port',type=int,default=8765)
    merge=sub.add_parser('combine',help='Combine the seven match JSON files into one catalog')
    merge.add_argument('--input-dir',default='data/raw',help='Folder containing the seven matches_*.json files')
    merge.add_argument('--output',default='data/real/combined_matches.json')
    gen=sub.add_parser('demo',help='Write reproducible synthetic events and tracking')
    gen.add_argument('--output',default='data/demo');gen.add_argument('--seed',type=int,default=42)
    for command in ['analyze','evaluate']:
        p=sub.add_parser(command)
        p.add_argument('--events',help='Normalized event CSV or JSON; omit for synthetic demo')
        p.add_argument('--tracking',help='Optional normalized tracking JSON')
        p.add_argument('--match-id',help='Select one match from a multi-match event CSV')
        p.add_argument('--backend',choices=['offline','ollama','groq'],default='offline')
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--output',default='artifacts/'+command)
        if command=='analyze':
            p.add_argument('--system',choices=['baseline','single','multi','no_critic','no_game_state','no_specialists'],default='multi')
            p.add_argument('--window',type=float,default=15)
            p.add_argument('--database',help='Optional SQLite history file')
        else:
            p.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    try:
        if args.command=='doctor':
            print(json.dumps(OllamaBackend().check_ready(),indent=2))
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
        if args.command=='analyze':
            report=analyze(rows,frames,system=args.system,backend=args.backend,seed=args.seed,window_minutes=args.window)
            save_report(report,args.output,args.database)
            print(f"Report saved to {args.output}; {len(report['analysis']['claims'])} observations.")
        else:
            result=run_experiment(rows,frames,args.output,args.backend,args.repeats,args.seed)
            failures=sum(r['status']=='error' for r in result['records'])
            print(f'Experiment saved to {args.output}; {failures} failed runs. Human scores are intentionally blank.')
            if failures:raise SystemExit(1)
    except (ValueError,OSError,RuntimeError) as exc:parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
