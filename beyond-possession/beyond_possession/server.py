"""Local, dependency-free dashboard server. Binds to localhost by default."""
import gzip
import json
import mimetypes
import os
import secrets
import threading
import tempfile
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .data import DataError, demo, parse_csv, parse_json_events
from .pipeline import analyze, markdown
from .agents import OllamaBackend, ModelError
from .local_ai import DEFAULT_MODEL
from .cloud_ai import GroqBackend, CloudAIError, load_environment, key_present, DEFAULT_GROQ_MODEL, DEFAULT_PROVIDER
from .gemini_ai import GeminiBackend, gemini_key_present, DEFAULT_GEMINI_MODEL
from .shared_ai import SharedBackend
from .matches import combine_matches
from .dataset_splits import read_manifest, load_development_catalogue, development_view
from .tci_learning import fit_tci, evaluate_tci
from .outlook import explain_outlook

STATIC=Path(__file__).parent/'static'
MAX_BYTES=24*1024*1024
CATALOG=Path(__file__).resolve().parents[1]/'data/real/combined_matches.json'
if not CATALOG.is_file():CATALOG=CATALOG.with_suffix('.json.gz')

def read_saved_catalogue():
    if read_manifest() is not None:
        return load_development_catalogue()
    if CATALOG.suffix=='.gz':
        with gzip.open(CATALOG,'rt',encoding='utf-8') as file:return json.load(file)
    return json.loads(CATALOG.read_text(encoding='utf-8'))

def serve(host='127.0.0.1',port=8765,on_ready=None):
    load_environment()
    token=secrets.token_urlsafe(32)
    lock=threading.Lock()
    reports=OrderedDict()
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,body,kind='application/json'):
            if kind=='application/json': body=json.dumps(body,allow_nan=False).encode()
            elif isinstance(body,str): body=body.encode()
            self.send_response(code)
            self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY')
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/api/matches':
                if not CATALOG.is_file():return self.send(404,{'error':'No saved match catalogue. Select your match JSON files to create one.'})
                return self.send(200,{'catalog':read_saved_catalogue()})
            if path=='/api/health':return self.send(200,{'status':'ok'})
            if path=='/api/shared-status':
                try:return self.send(200,SharedBackend().check_ready())
                except CloudAIError as exc:return self.send(200,{'available':False,'error':str(exc)})
            if path=='/api/gemini-status':
                try:return self.send(200,GeminiBackend().check_ready())
                except CloudAIError as exc:return self.send(200,{'available':False,'error':str(exc)})
            if path=='/api/groq-status':
                try:return self.send(200,GroqBackend().check_ready())
                except CloudAIError as exc:return self.send(200,{'available':False,'error':str(exc)})
            if path=='/api/ollama-status':
                try:return self.send(200,OllamaBackend().check_ready())
                except ModelError as exc:return self.send(200,{'available':False,'error':str(exc)})
            if path=='/api/config':return self.send(200,{'token':token,'preferred_backend':os.getenv('BEYOND_PREFERRED_BACKEND',os.getenv('AI_PROVIDER',DEFAULT_PROVIDER)),'gemini_configured':gemini_key_present(),'gemini_model':os.getenv('GEMINI_MODEL',DEFAULT_GEMINI_MODEL),'groq_configured':key_present(),'groq_model':os.getenv('GROQ_MODEL',DEFAULT_GROQ_MODEL),'match_catalog_available':CATALOG.is_file(),'ai_ready':os.getenv('BEYOND_AI_READY')=='1','ai_setup_error':os.getenv('BEYOND_AI_SETUP_ERROR',''),'ollama_model':os.getenv('OLLAMA_MODEL',DEFAULT_MODEL)})
            allowed={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path not in allowed:return self.send(404,{'error':'Not found.'})
            file=STATIC/allowed[path]
            return self.send(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'text/plain')
        def do_POST(self):
            # Cross-origin requests cannot acquire this token: no CORS is enabled.
            if self.headers.get('X-Session-Token')!=token:return self.send(403,{'error':'Invalid session token. Reload the dashboard.'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>MAX_BYTES:return self.send(413,{'error':'Upload must be between 1 byte and 24 MiB.'})
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise DataError('Expected a JSON object.')
                if urlparse(self.path).path=='/api/outlook':
                    handle=payload.get('report_handle')
                    if not isinstance(handle,str) or handle not in reports:
                        return self.send(404,{'error':'This analysis is no longer in memory. Analyze the match again.'})
                    if not lock.acquire(blocking=False):return self.send(409,{'error':'Another analysis or outlook is running. Wait for it to finish.'})
                    try:
                        report=reports[handle]
                        outlook=next((o for o in report['outlooks'] if o['team']==payload.get('team') and o['selected_interval_start']==payload.get('interval_start')),None)
                        if outlook is None:raise DataError('Choose a team and interval from this analyzed match.')
                        cached=outlook.get('ai') is not None
                        if not cached:
                            backend=report['analysis']['backend']
                            providers={'shared':SharedBackend,'gemini':lambda:GeminiBackend(cache=True),
                                       'groq':lambda:GroqBackend(cache=True),'ollama':OllamaBackend}
                            if backend not in providers:raise DataError('AI outlooks need an AI engine. The calculated outlook is already available.')
                            outlook=explain_outlook(outlook,providers[backend](),report['config']['seed'])
                            report['outlooks']=[outlook if o['team']==outlook['team'] and o['selected_interval_start']==outlook['selected_interval_start'] else o for o in report['outlooks']]
                        return self.send(200,{'outlook':outlook,'outlooks':report['outlooks'],'markdown':markdown(report),'cached':cached})
                    finally:lock.release()
                if urlparse(self.path).path=='/api/train-tci':
                    if not lock.acquire(blocking=False):return self.send(409,{'error':'An analysis or training run is already active.'})
                    try:
                        training=parse_json_events(payload['training_json']) if 'training_json' in payload else parse_csv(payload.get('training_csv',''))
                        model=fit_tci(training,float(payload.get('window_minutes',15)))
                        testing=None
                        if 'test_json' in payload or 'test_csv' in payload:
                            test=parse_json_events(payload['test_json']) if 'test_json' in payload else parse_csv(payload['test_csv'])
                            testing=evaluate_tci(model,test)
                        return self.send(200,{'model':model,'test':testing})
                    finally:lock.release()
                if urlparse(self.path).path=='/api/combine-matches':
                    files=payload.get('files',[])
                    if not isinstance(files,list) or not 1<=len(files)<=20:
                        raise DataError('Select between 1 and 20 match JSON files.')
                    with tempfile.TemporaryDirectory() as directory:
                        paths=[];names={}
                        for i,item in enumerate(files):
                            if not isinstance(item,dict) or not isinstance(item.get('text'),str):
                                raise DataError('Each file needs a filename and JSON text.')
                            name=Path(str(item.get('name','matches.json'))).name
                            path=Path(directory)/f'{i}-{name}'
                            path.write_text(item['text'],encoding='utf-8');paths.append(path);names[path.name]=name
                        combined=combine_matches(paths)
                        for source in combined['sources']:source['filename']=names[source['filename']]
                        return self.send(200,{'catalog':development_view(combined)})
                if urlparse(self.path).path!='/api/analyze':return self.send(404,{'error':'Not found.'})
                if not lock.acquire(blocking=False):return self.send(409,{'error':'An analysis is already running. Try again when it finishes.'})
                try:
                    if payload.get('source','demo')=='demo':rows,frames=demo(int(payload.get('seed',42)))
                    else:
                        rows=parse_json_events(payload['events_json']) if 'events_json' in payload else parse_csv(payload.get('events_csv',''))
                        match=payload.get('match_id')
                        if match: rows=[e for e in rows if e['match_id']==match]
                        frames=json.loads(payload.get('tracking_json') or '[]')
                    backend=payload.get('backend','offline')
                    model=None
                    if payload.get('system','multi')!='baseline':
                        if backend=='shared':model=SharedBackend()
                        elif backend=='gemini':model=GeminiBackend(cache=True)
                        elif backend=='groq':model=GroqBackend(cache=True)
                    report=analyze(rows,frames,system=payload.get('system','multi'),backend=backend,model_backend=model,
                        seed=int(payload.get('seed',42)),window_minutes=float(payload.get('window_minutes',15)),
                        generation_mode=payload.get('generation_mode','template'),critic_mode=payload.get('critic_mode','numeric'),
                        tci_model=payload.get('tci_model'))
                    handle=secrets.token_urlsafe(18)
                    reports[handle]=report
                    while len(reports)>3:reports.popitem(last=False)
                    return self.send(200,{'report':report,'report_handle':handle,'markdown':markdown(report)})
                finally:lock.release()
            except (ValueError,TypeError,KeyError) as exc:return self.send(400,{'error':str(exc)})
            except Exception as exc:return self.send(502,{'error':str(exc)})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer((host,port),Handler)
    print(f'Beyond Possession is running at http://{host}:{server.server_address[1]}',flush=True)
    print('Synthetic demo by default. Press Ctrl+C to stop.',flush=True)
    if on_ready:
        threading.Thread(target=on_ready, args=(f'http://{host}:{server.server_address[1]}',), daemon=True).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
