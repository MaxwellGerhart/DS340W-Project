"""First-run portable local AI setup. Downloads only official Ollama release assets."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from .tls import verified_context

DEFAULT_MODEL = 'qwen2.5:3b'
RELEASE = 'v0.40.11'
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.local-ai'

class SetupError(RuntimeError):
    pass

def read_json(url, timeout=5):
    request=urllib.request.Request(url,headers={'User-Agent':'Beyond-Possession/0.1'})
    with urllib.request.urlopen(request,timeout=timeout,context=verified_context()) as response:
        return json.load(response)

def service_running(url):
    try:
        data=read_json(url+'/api/version',2)
        return isinstance(data,dict) and bool(data.get('version'))
    except (OSError,ValueError):
        return False

def asset_name(system=None, machine=None):
    system=system or platform.system();machine=(machine or platform.machine()).lower()
    arch={'x86_64':'amd64','amd64':'amd64','arm64':'arm64','aarch64':'arm64'}.get(machine)
    if not arch:raise SetupError('Unsupported processor: '+machine)
    if system=='Darwin':return 'ollama-darwin.tgz'
    if system=='Windows' and arch=='amd64':return 'ollama-windows-amd64.zip'
    if system=='Linux':return 'ollama-linux-'+arch+'.tar.zst'
    raise SetupError('Automatic setup is not supported on '+system+'/'+machine+'. Install Ollama from https://ollama.com/download instead.')

def safe_path(name):
    normalized=name.replace('\\','/')
    return not normalized.startswith('/') and not any(part=='..' or ':' in part for part in normalized.split('/'))

def extract_runtime(archive, destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if archive.name.endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            if any(not safe_path(i.filename) for i in z.infolist()):raise SetupError('Invalid path in runtime archive.')
            z.extractall(destination)
    elif archive.name.endswith('.tgz'):
        with tarfile.open(archive,'r:gz') as tar:
            tar.extractall(destination,filter='data')
    else:
        # Official Linux packages use zstd. GNU tar and zstd must be available.
        if not shutil.which('tar'):raise SetupError('Linux portable setup needs tar with zstd support. Install Ollama from its official Linux instructions, then rerun.')
        listing=subprocess.run(['tar','--zstd','-tf',str(archive)],capture_output=True,text=True)
        if listing.returncode:raise SetupError('Linux portable setup needs tar with zstd support. Install zstd or Ollama, then rerun.')
        if any(not safe_path(name) for name in listing.stdout.splitlines()):raise SetupError('Invalid path in runtime archive.')
        subprocess.run(['tar','--zstd','-xf',str(archive),'-C',str(destination)],check=True)

def download_runtime(cache=CACHE, log=print):
    cache=Path(cache);runtime=cache/'runtime'/RELEASE
    executable_name='ollama.exe' if platform.system()=='Windows' else 'ollama'
    record=runtime/'installed.json'
    if record.is_file():
        installed=json.loads(record.read_text())
        candidate=runtime/installed['executable']
        if candidate.is_file() and candidate.resolve().is_relative_to(runtime.resolve()):return candidate
    name=asset_name()
    log(f'First-run AI setup: downloading the official Ollama runtime ({RELEASE}).')
    release=read_json('https://api.github.com/repos/ollama/ollama/releases/tags/'+RELEASE,30)
    asset=next((a for a in release['assets'] if a['name']==name),None)
    if not asset:raise SetupError('No official portable runtime found for '+name+'. Install Ollama manually from its official site.')
    digest=asset.get('digest','')
    if not digest.startswith('sha256:'):raise SetupError('The official runtime has no SHA-256 digest; automatic installation stopped.')
    url=asset['browser_download_url']
    if not url.startswith('https://github.com/ollama/ollama/releases/download/'+RELEASE+'/'):
        raise SetupError('Unexpected runtime download source.')
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        archive=Path(directory)/name;sha=hashlib.sha256();done=0;last=0
        with urllib.request.urlopen(url,timeout=90,context=verified_context()) as source,archive.open('wb') as target:
            while True:
                chunk=source.read(1024*1024)
                if not chunk:break
                target.write(chunk);sha.update(chunk);done+=len(chunk)
                if time.monotonic()-last>=2:
                    log(f'Ollama runtime: {done//(1024*1024)} MiB downloaded');last=time.monotonic()
        if sha.hexdigest()!=digest.split(':',1)[1]:raise SetupError('Runtime checksum did not match the official release. Download was not installed.')
        extracted=Path(directory)/'extracted';extract_runtime(archive,extracted)
        candidates=[p for p in extracted.rglob(executable_name) if p.is_file()]
        if not candidates:raise SetupError('The runtime archive did not contain the Ollama executable.')
        executable=min(candidates,key=lambda p:len(p.parts));relative=executable.relative_to(extracted)
        executable.chmod(executable.stat().st_mode|0o111)
        (extracted/'installed.json').write_text(json.dumps({'release':RELEASE,'sha256':sha.hexdigest(),'executable':str(relative)}))
        runtime.parent.mkdir(parents=True,exist_ok=True)
        if runtime.exists():shutil.rmtree(runtime)
        shutil.move(str(extracted),runtime)
    return runtime/relative

def ensure_model(url, model, log=print):
    names=[m.get('name','') for m in read_json(url+'/api/tags').get('models',[])]
    if model in names or (model+':latest') in names:
        log('Local AI model is already downloaded: '+model)
        return
    log('Downloading '+model+'. This happens once per machine and may take several minutes.')
    request=urllib.request.Request(url+'/api/pull',data=json.dumps({'model':model,'stream':True}).encode(),headers={'Content-Type':'application/json'})
    success=False;last_status='';last_time=0
    with urllib.request.urlopen(request,timeout=300,context=verified_context()) as response:
        for line in response:
            if not line.strip():continue
            update=json.loads(line)
            if update.get('error'):raise SetupError('Model download failed: '+str(update['error']))
            status=update.get('status','Downloading model')
            if status=='success':success=True
            now=time.monotonic()
            if status!=last_status or now-last_time>=2:
                suffix=f" ({update.get('completed',0)*100/update['total']:.0f}%)" if update.get('total') else ''
                log(status+suffix);last_status=status;last_time=now
    if not success:raise SetupError('Model download ended before Ollama confirmed success. Run the app again to resume.')
    names=[m.get('name','') for m in read_json(url+'/api/tags').get('models',[])]
    if model not in names and model+':latest' not in names:raise SetupError('Downloaded model was not found during the final readiness check.')

def prepare_local_ai(cache=CACHE, log=print):
    """Return the private server process if started, otherwise None. Caller owns cleanup."""
    cache=Path(cache);model=os.environ.setdefault('OLLAMA_MODEL',DEFAULT_MODEL)
    configured=os.getenv('OLLAMA_BASE_URL')
    existing=(configured or 'http://127.0.0.1:11434').rstrip('/')
    owned=None
    try:
        if service_running(existing):
            url=existing
        else:
            if configured:raise SetupError('Configured Ollama server is not reachable: '+configured)
            url='http://127.0.0.1:11435'
            if not service_running(url):
                executable=shutil.which('ollama')
                if not executable:
                    mac=Path('/Applications/Ollama.app/Contents/Resources/ollama')
                    executable=str(mac) if mac.is_file() else str(download_runtime(cache,log))
                cache.mkdir(parents=True,exist_ok=True)
                env=os.environ.copy();env['OLLAMA_HOST']='127.0.0.1:11435';env['OLLAMA_MODELS']=str((cache/'models').resolve())
                with (cache/'server.log').open('ab') as logfile:
                    owned=subprocess.Popen([executable,'serve'],env=env,stdout=logfile,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+40
                while not service_running(url):
                    if owned.poll() is not None:raise SetupError('Local AI could not start. Check .local-ai/server.log.')
                    if time.monotonic()>=deadline:raise SetupError('Local AI took too long to start. Check .local-ai/server.log.')
                    time.sleep(.5)
            os.environ['OLLAMA_BASE_URL']=url
        ensure_model(url,model,log)
        os.environ['BEYOND_AI_READY']='1'
        log('Local AI is ready: '+model)
        return owned
    except BaseException as exc:
        stop_local_ai(owned)
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
        raise SetupError('Automatic AI setup could not finish: '+str(exc)) from exc

def stop_local_ai(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
