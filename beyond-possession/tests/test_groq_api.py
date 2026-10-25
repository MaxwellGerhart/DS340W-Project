import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from beyond_possession.cloud_ai import GroqBackend,CloudAIError,load_environment
from beyond_possession.data import demo
from beyond_possession.pipeline import analyze

class Response:
    def __init__(self,data):self.data=data
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return json.dumps(self.data).encode()

class GroqTests(unittest.TestCase):
    def test_environment_is_data_and_exported_values_take_priority(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'GROQ_MODEL':'openai/gpt-oss-20b'},clear=True):
            file=Path(tmp)/'.env';file.write_text('GROQ_API_KEY="fixture-key"\nGROQ_MODEL=openai/gpt-oss-120b\nHOME=do-not-change\nUNKNOWN=$(false)\n')
            load_environment(file)
            self.assertEqual(os.environ['GROQ_API_KEY'],'fixture-key')
            self.assertEqual(os.environ['GROQ_MODEL'],'openai/gpt-oss-20b')
            self.assertNotIn('HOME',os.environ)
            self.assertNotIn('UNKNOWN',os.environ)
    def test_missing_key_has_instructions(self):
        with patch.dict(os.environ,{},clear=True),patch('beyond_possession.cloud_ai.load_environment'):
            with self.assertRaisesRegex(CloudAIError,'GROQ_API_KEY'):GroqBackend()
    def test_auth_schema_and_token_logging(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-secret','GROQ_MODEL':'openai/gpt-oss-20b'},clear=True),patch('urllib.request.urlopen',return_value=Response({'choices':[{'message':{'content':'{"claims":[]}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':20,'completion_tokens':5}})) as call:
            model=GroqBackend();self.assertEqual(model.select('spatial',[],42),[])
            req=call.call_args.args[0];body=json.loads(req.data)
            self.assertEqual(req.get_header('Authorization'),'Bearer fixture-secret')
            self.assertTrue(body['response_format']['json_schema']['strict'])
            self.assertEqual(model.calls[0]['input_tokens'],20)
            self.assertNotIn('fixture-secret',json.dumps(model.calls))
    def test_rate_limit_is_actionable_and_no_retry(self):
        error=urllib.error.HTTPError('https://api.groq.com',429,'quota',{},io.BytesIO(b'{"error":{"message":"quota"}}'))
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-secret'},clear=True),patch('urllib.request.urlopen',side_effect=error) as call:
            with self.assertRaisesRegex(CloudAIError,'HTTP 429'):GroqBackend().select('spatial',[],42)
            self.assertEqual(call.call_count,1)
    def test_auth_errors_do_not_reveal_key(self):
        error=urllib.error.HTTPError('https://api.groq.com',401,'bad',{},io.BytesIO(b'{"error":{"message":"fixture-secret"}}'))
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-secret'},clear=True),patch('urllib.request.urlopen',side_effect=error):
            try:GroqBackend().select('spatial',[],42)
            except CloudAIError as exc:self.assertNotIn('fixture-secret',str(exc));self.assertIn('API key',str(exc))
            else:self.fail('Expected rejected key')
    def test_truncation_is_not_an_offline_success(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-secret'},clear=True),patch('urllib.request.urlopen',return_value=Response({'choices':[{'message':{'content':'{"claims":[]}'},'finish_reason':'length'}]})):
            with self.assertRaisesRegex(CloudAIError,'complete list'):GroqBackend().select('spatial',[],42)
    def test_pipeline_supports_groq_with_evidence_checks(self):
        class Fake:
            model='openai/gpt-oss-20b';calls=[]
            def select(self,role,evidence,seed):
                e=evidence[0]
                return [{'evidence_id':e['evidence_id'],'value':e['value'],'alternative_explanation':'No causal inference is supported.'}]
        rows,_=demo();report=analyze(rows,backend='groq',model_backend=Fake())
        self.assertEqual(report['analysis']['backend'],'groq')
        self.assertTrue(report['analysis']['claims'])
        self.assertTrue(all(c['verification']=='verified' for c in report['analysis']['claims']))

class GroqConnectionTests(unittest.TestCase):
    def test_connection_check_does_not_request_inference(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-secret'},clear=True),patch('urllib.request.urlopen',return_value=Response({'data':[{'id':'openai/gpt-oss-20b'}]})) as call:
            self.assertTrue(GroqBackend().check_ready()['available'])
            self.assertTrue(call.call_args.args[0].full_url.endswith('/models'))
            self.assertIsNone(call.call_args.args[0].data)
    def test_certificate_verification_stays_enabled(self):
        import ssl
        from beyond_possession.tls import verified_context
        context=verified_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode,ssl.CERT_REQUIRED)

class ApiStartupTests(unittest.TestCase):
    def test_api_startup_does_not_install_or_download_ollama(self):
        import run
        with patch.dict(os.environ,{'AI_PROVIDER':'groq'},clear=True),patch('sys.argv',['run.py','--no-browser']),patch('beyond_possession.local_ai.prepare_local_ai') as install,patch('beyond_possession.server.serve') as serve,patch('builtins.print'):
            self.assertEqual(run.main(),0)
            install.assert_not_called()
            serve.assert_called_once()

class ApiQuotaProtectionTests(unittest.TestCase):
    def test_prompt_is_bounded_and_covers_each_team_interval(self):
        from beyond_possession.agents import catalog
        from beyond_possession.analytics import summarize
        from beyond_possession.cloud_ai import compact_evidence
        rows,_=demo();all_evidence=list(catalog(summarize(rows)).values())
        selected=compact_evidence(all_evidence)
        self.assertLessEqual(len(selected),24)
        expected={(e['team'],e['start']) for e in all_evidence}
        self.assertEqual({(e['team'],e['start']) for e in selected},expected)
        self.assertLess(len(json.dumps(selected).encode()),12000)
    def test_dashboard_cache_avoids_second_inference_request(self):
        from beyond_possession.cloud_ai import _CACHE
        _CACHE.clear()
        response=Response({'choices':[{'message':{'content':'{"claims":[]}'},'finish_reason':'stop'}]})
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-cache-key'},clear=True),patch('urllib.request.urlopen',return_value=response) as call:
            backend=GroqBackend(cache=True)
            backend.select('spatial',[],42);backend.select('spatial',[],42)
            self.assertEqual(call.call_count,1)
            self.assertTrue(backend.calls[-1]['cached'])
            self.assertEqual(backend.calls[-1]['input_tokens'],0)
        _CACHE.clear()
    def test_research_mode_does_not_cache(self):
        response=Response({'choices':[{'message':{'content':'{"claims":[]}'},'finish_reason':'stop'}]})
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-cache-key'},clear=True),patch('urllib.request.urlopen',return_value=response) as call:
            backend=GroqBackend()
            backend.select('spatial',[],42);backend.select('spatial',[],42)
            self.assertEqual(call.call_count,2)
    def test_specialists_stop_after_first_failure(self):
        from beyond_possession.agents import run_agents
        from beyond_possession.analytics import summarize
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-key'},clear=True):
            model=GroqBackend()
        rows,_=demo()
        with patch.object(model,'select',side_effect=CloudAIError('HTTP 429')) as select:
            with self.assertRaises(CloudAIError):run_agents(summarize(rows),backend='groq',system='multi',model_backend=model)
            self.assertEqual(select.call_count,1)
    def test_quota_error_preserves_reason_and_retry_after(self):
        error=urllib.error.HTTPError('https://api.groq.com',429,'quota',{'Retry-After':'20'},io.BytesIO(b'{"error":{"message":"Request too large for tokens per minute: fixture-key"}}'))
        with patch.dict(os.environ,{'GROQ_API_KEY':'fixture-key'},clear=True),patch('urllib.request.urlopen',side_effect=error):
            try:GroqBackend().select('spatial',[],42)
            except CloudAIError as exc:
                self.assertIn('Request too large',str(exc));self.assertIn('20 seconds',str(exc));self.assertNotIn('fixture-key',str(exc))
            else:self.fail('Expected a rejected request')
