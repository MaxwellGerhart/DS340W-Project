import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from beyond_possession import local_ai

class LocalAiSetupTests(unittest.TestCase):
    def test_platform_assets(self):
        self.assertEqual(local_ai.asset_name('Darwin','arm64'),'ollama-darwin.tgz')
        self.assertEqual(local_ai.asset_name('Windows','AMD64'),'ollama-windows-amd64.zip')
        self.assertEqual(local_ai.asset_name('Linux','x86_64'),'ollama-linux-amd64.tar.zst')
        with self.assertRaises(local_ai.SetupError):local_ai.asset_name('Windows','arm64')
    def test_first_download_verifies_and_reuses_cached_runtime(self):
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w:gz') as t:
            content=b'fixture executable';entry=tarfile.TarInfo('bin/ollama');entry.size=len(content);t.addfile(entry,io.BytesIO(content))
        body=buf.getvalue()
        metadata={'assets':[{'name':'ollama-darwin.tgz','digest':'sha256:'+hashlib.sha256(body).hexdigest(),'browser_download_url':'https://github.com/ollama/ollama/releases/download/'+local_ai.RELEASE+'/ollama-darwin.tgz'}]}
        with tempfile.TemporaryDirectory() as tmp,patch.object(local_ai.platform,'system',return_value='Darwin'),patch.object(local_ai,'read_json',return_value=metadata),patch.object(local_ai.urllib.request,'urlopen',return_value=io.BytesIO(body)) as call:
            executable=local_ai.download_runtime(tmp,lambda _:None)
            self.assertEqual(executable.read_bytes(),b'fixture executable')
            self.assertEqual(local_ai.download_runtime(tmp,lambda _:None),executable)
            self.assertEqual(call.call_count,1)
    def test_wrong_checksum_not_installed(self):
        metadata={'assets':[{'name':'ollama-darwin.tgz','digest':'sha256:'+'0'*64,'browser_download_url':'https://github.com/ollama/ollama/releases/download/'+local_ai.RELEASE+'/ollama-darwin.tgz'}]}
        with tempfile.TemporaryDirectory() as tmp,patch.object(local_ai.platform,'system',return_value='Darwin'),patch.object(local_ai,'read_json',return_value=metadata),patch.object(local_ai.urllib.request,'urlopen',return_value=io.BytesIO(b'wrong')):
            with self.assertRaisesRegex(local_ai.SetupError,'checksum'):local_ai.download_runtime(tmp,lambda _:None)
            self.assertFalse((Path(tmp)/'runtime'/local_ai.RELEASE).exists())
    def test_model_cache_skips_download(self):
        with patch.object(local_ai,'read_json',return_value={'models':[{'name':'qwen2.5:3b'}]}),patch.object(local_ai.urllib.request,'urlopen') as call:
            local_ai.ensure_model('http://local','qwen2.5:3b',lambda _:None)
            call.assert_not_called()
    def test_streamed_pull_requires_confirmed_success_and_readiness(self):
        model='qwen2.5:3b'
        stream=io.BytesIO(b'{"status":"pulling","total":100,"completed":50}\n{"status":"success"}\n')
        with patch.object(local_ai,'read_json',side_effect=[{'models':[]},{'models':[{'name':model}]}]),patch.object(local_ai.urllib.request,'urlopen',return_value=stream):
            local_ai.ensure_model('http://local',model,lambda _:None)
    def test_failed_pull_is_not_ready(self):
        with patch.object(local_ai,'read_json',return_value={'models':[]}),patch.object(local_ai.urllib.request,'urlopen',return_value=io.BytesIO(b'{"error":"out of disk space"}\n')):
            with self.assertRaisesRegex(local_ai.SetupError,'out of disk space'):
                local_ai.ensure_model('http://local','qwen2.5:3b',lambda _:None)
    def test_existing_service_reused_without_install(self):
        with patch.dict(os.environ,{},clear=True),patch.object(local_ai,'service_running',return_value=True),patch.object(local_ai,'ensure_model') as model,patch.object(local_ai,'download_runtime') as download:
            self.assertIsNone(local_ai.prepare_local_ai(log=lambda _:None))
            download.assert_not_called();model.assert_called_once()
            self.assertEqual(os.environ['BEYOND_AI_READY'],'1')
    def test_owned_server_stopped_on_model_failure(self):
        process=Mock();process.poll.return_value=None
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True),patch.object(local_ai,'service_running',side_effect=[False,False,True]),patch.object(local_ai.shutil,'which',return_value='/usr/bin/ollama'),patch.object(local_ai.subprocess,'Popen',return_value=process),patch.object(local_ai,'ensure_model',side_effect=local_ai.SetupError('download failed')):
            with self.assertRaises(local_ai.SetupError):local_ai.prepare_local_ai(tmp,lambda _:None)
            process.terminate.assert_called_once()
    def test_explicit_unavailable_endpoint_is_not_overridden(self):
        with patch.dict(os.environ,{'OLLAMA_BASE_URL':'http://localhost:9999'},clear=True),patch.object(local_ai,'service_running',return_value=False),patch.object(local_ai,'download_runtime') as download:
            with self.assertRaises(local_ai.SetupError):local_ai.prepare_local_ai(log=lambda _:None)
            download.assert_not_called()
    def test_archive_paths(self):
        for name in ['../bad','/absolute','C:\\bad','x/../bad']:
            self.assertFalse(local_ai.safe_path(name))
        self.assertTrue(local_ai.safe_path('lib/ollama/model.dll'))
