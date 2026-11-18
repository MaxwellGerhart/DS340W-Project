import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch
from beyond_possession.cloud_ai import CloudAIError, load_environment
from beyond_possession.gemini_ai import GeminiBackend, DEFAULT_GEMINI_MODEL, _CACHE
from beyond_possession.data import demo
from beyond_possession.pipeline import analyze


class Response:
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.data).encode()


def completion(claims=None, reason='STOP'):
    return {'candidates': [{'finishReason': reason, 'content': {'parts': [
        {'text': json.dumps({'claims': claims or []})}]}}],
        'usageMetadata': {'promptTokenCount': 30, 'candidatesTokenCount': 12, 'thoughtsTokenCount': 3}}


class GeminiTests(unittest.TestCase):
    def setUp(self):
        _CACHE.clear()
        self.env = patch.dict(os.environ, {'GEMINI_API_KEY': 'fixture-google-secret'}, clear=True)
        self.env.start()
        self.loader = patch('beyond_possession.gemini_ai.load_environment')
        self.loader.start()

    def tearDown(self):
        self.loader.stop()
        self.env.stop()
        _CACHE.clear()

    def test_native_google_request_contract_and_usage(self):
        with patch('urllib.request.urlopen', return_value=Response(completion())) as call:
            model = GeminiBackend()
            self.assertEqual(model.select('spatial', [], 42), [])
        req = call.call_args.args[0]
        self.assertEqual(req.get_header('X-goog-api-key'), 'fixture-google-secret')
        self.assertEqual(req.full_url, 'https://generativelanguage.googleapis.com/v1beta/models/'+DEFAULT_GEMINI_MODEL+':generateContent')
        self.assertNotIn('fixture-google-secret', req.full_url)
        payload = json.loads(req.data)
        self.assertEqual(payload['generationConfig']['responseMimeType'], 'application/json')
        self.assertEqual(payload['generationConfig']['responseJsonSchema']['properties']['claims']['maxItems'], 8)
        self.assertIn('spatial', payload['systemInstruction']['parts'][0]['text'])
        self.assertEqual(model.calls[0]['input_tokens'], 30)
        self.assertEqual(model.calls[0]['output_tokens'], 12)
        self.assertEqual(model.calls[0]['thinking_tokens'], 3)
        self.assertNotIn('fixture-google-secret', json.dumps(model.calls))
        self.assertTrue(call.call_args.kwargs['context'].check_hostname)

    def test_missing_key_does_not_substitute_offline_findings(self):
        os.environ.pop('GEMINI_API_KEY')
        rows, _ = demo()
        with patch('beyond_possession.agents.load_environment'), self.assertRaisesRegex(CloudAIError, 'GEMINI_API_KEY'):
            analyze(rows, backend='gemini', system='single')

    def test_google_key_alias_and_model_validation(self):
        os.environ.pop('GEMINI_API_KEY')
        os.environ['GOOGLE_API_KEY'] = 'fixture-alias'
        self.assertEqual(GeminiBackend()._key, 'fixture-alias')
        os.environ['GEMINI_MODEL'] = 'https://example.invalid/steal'
        with self.assertRaisesRegex(CloudAIError, 'model ID'):
            GeminiBackend()

    def test_model_lookup_check_never_generates_text(self):
        info = {'name': 'models/'+DEFAULT_GEMINI_MODEL, 'supportedGenerationMethods': ['generateContent']}
        with patch('urllib.request.urlopen', return_value=Response(info)) as call:
            self.assertTrue(GeminiBackend().check_ready()['available'])
        req = call.call_args.args[0]
        self.assertIsNone(req.data)
        self.assertNotIn(':generateContent', req.full_url)

    def test_lookup_does_not_claim_success_for_non_generation_model(self):
        with patch('urllib.request.urlopen', return_value=Response({'name': 'models/'+DEFAULT_GEMINI_MODEL})):
            with self.assertRaisesRegex(CloudAIError, 'text-generation access'):
                GeminiBackend().check_ready()

    def test_quota_is_not_retried_and_key_is_redacted(self):
        error = urllib.error.HTTPError('https://generativelanguage.googleapis.com', 429, 'quota',
            {'Retry-After': '20'}, io.BytesIO(b'{"error":{"message":"fixture-google-secret daily quota"}}'))
        with patch('urllib.request.urlopen', side_effect=error) as call:
            with self.assertRaisesRegex(CloudAIError, 'HTTP 429') as caught:
                GeminiBackend().select('spatial', [], 42)
        self.assertEqual(call.call_count, 1)
        self.assertIn('20 seconds', str(caught.exception))
        self.assertIn('daily quota', str(caught.exception))
        self.assertNotIn('fixture-google-secret', str(caught.exception))

    def overload(self,retry=None):
        return urllib.error.HTTPError('https://generativelanguage.googleapis.com',503,'busy',
            {'Retry-After':retry} if retry else {},io.BytesIO(b'{"error":{"message":"fixture-google-secret high demand"}}'))

    def test_503_recovers_with_bounded_backoff_and_logs_real_attempts(self):
        with patch('urllib.request.urlopen',side_effect=[self.overload(),self.overload(),Response(completion())]) as call,patch('beyond_possession.gemini_ai.time.sleep') as wait,patch('beyond_possession.gemini_ai.random.uniform',return_value=0):
            model=GeminiBackend();self.assertEqual(model.select('spatial',[],42),[])
        self.assertEqual(call.call_count,3)
        self.assertEqual([entry.args[0] for entry in wait.call_args_list],[1,2])
        self.assertEqual(model.calls[0]['http_attempts'],3)
        self.assertNotIn('fixture-google-secret',json.dumps(model.calls))

    def test_persistent_503_stops_at_three_and_redacts(self):
        with patch('urllib.request.urlopen',side_effect=[self.overload() for _ in range(3)]) as call,patch('beyond_possession.gemini_ai.time.sleep'):
            with self.assertRaisesRegex(CloudAIError,'after 3 attempt') as caught:GeminiBackend().select('spatial',[],42)
        self.assertEqual(call.call_count,3)
        self.assertNotIn('fixture-google-secret',str(caught.exception))

    def test_long_retry_after_and_non_inference_checks_do_not_loop(self):
        with patch('urllib.request.urlopen',side_effect=self.overload('60')) as call,patch('beyond_possession.gemini_ai.time.sleep') as wait:
            with self.assertRaises(CloudAIError):GeminiBackend().select('spatial',[],42)
        self.assertEqual(call.call_count,1);wait.assert_not_called()
        with patch('urllib.request.urlopen',side_effect=self.overload()) as call,patch('beyond_possession.gemini_ai.time.sleep') as wait:
            with self.assertRaises(CloudAIError):GeminiBackend().check_ready()
        self.assertEqual(call.call_count,1);wait.assert_not_called()

    def test_retry_stops_immediately_if_google_returns_quota_or_bad_request(self):
        for code in [400,401,403,404,429]:
            error=urllib.error.HTTPError('https://generativelanguage.googleapis.com',code,'rejected',{},io.BytesIO(b'{"error":{"message":"fixture-google-secret"}}'))
            with self.subTest(code=code),patch('urllib.request.urlopen',side_effect=[self.overload(),error]) as call,patch('beyond_possession.gemini_ai.time.sleep') as wait:
                with self.assertRaises(CloudAIError):GeminiBackend().select('spatial',[],42)
            self.assertEqual(call.call_count,2);self.assertEqual(wait.call_count,1)

    def test_auth_and_model_errors_include_correct_next_step(self):
        for code, hint in [(403, 'GEMINI_API_KEY'), (404, 'GEMINI_MODEL')]:
            error = urllib.error.HTTPError('https://generativelanguage.googleapis.com', code, 'rejected', {},
                io.BytesIO(b'{"error":{"message":"fixture-google-secret"}}'))
            with self.subTest(code=code), patch('urllib.request.urlopen', side_effect=error):
                with self.assertRaisesRegex(CloudAIError, hint) as caught:
                    GeminiBackend().check_ready()
                self.assertNotIn('fixture-google-secret', str(caught.exception))

    def test_blocked_truncated_and_malformed_results_fail(self):
        responses = [completion(reason='MAX_TOKENS'), {'promptFeedback': {'blockReason': 'SAFETY'}},
            {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': 'not JSON'}]}}]},
            completion([{'evidence_id': 'e', 'value': True, 'alternative_explanation': 'No causal inference is supported.'}])]
        for result in responses:
            with self.subTest(result=result), patch('urllib.request.urlopen', return_value=Response(result)):
                with self.assertRaisesRegex(CloudAIError, 'no built-in result was substituted'):
                    GeminiBackend().select('spatial', [], 42)

    def test_dashboard_cache_avoids_repeat_inference_and_copies_results(self):
        claim = {'evidence_id': 'e', 'value': 1, 'alternative_explanation': 'No causal inference is supported.'}
        with patch('urllib.request.urlopen', return_value=Response(completion([claim]))) as call:
            model = GeminiBackend(cache=True)
            first = model.select('spatial', [], 42)
            first[0]['value'] = 999
            second = model.select('spatial', [], 42)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(second[0]['value'], 1)
            self.assertTrue(model.calls[-1]['cached'])

    def test_research_repetitions_do_not_cache(self):
        with patch('urllib.request.urlopen', return_value=Response(completion())) as call:
            model = GeminiBackend()
            model.select('spatial', [], 42)
            model.select('spatial', [], 42)
            self.assertEqual(call.call_count, 2)

    def test_specialists_stop_on_first_api_failure(self):
        rows, _ = demo()
        with patch('beyond_possession.agents.load_environment'), patch.object(GeminiBackend, 'select', side_effect=CloudAIError('HTTP 429')) as call:
            with self.assertRaisesRegex(CloudAIError, 'HTTP 429'):
                analyze(rows, backend='gemini', system='multi')
            self.assertEqual(call.call_count, 1)

    def test_http_analysis_reaches_google_and_runs_numerical_critic(self):
        from beyond_possession import server
        captured = {}
        class FakeServer:
            def __init__(self, address, handler): captured['handler'] = handler; self.server_address = address
            def serve_forever(self): pass
            def server_close(self): pass
        with patch.object(server, 'ThreadingHTTPServer', FakeServer), patch.object(server, 'load_environment'), patch('builtins.print'):
            server.serve()
        handler = captured['handler']
        def request(method, path, payload=None, token=''):
            obj = object.__new__(handler); obj.path = path
            encoded = json.dumps(payload or {}).encode()
            obj.headers = {'Content-Length': str(len(encoded)), 'X-Session-Token': token}
            obj.rfile = io.BytesIO(encoded)
            obj.send = lambda status, body, kind='application/json': (status, body)
            return getattr(obj, 'do_'+method)()
        _, config = request('GET', '/api/config')
        self.assertEqual(config['preferred_backend'], 'shared')
        self.assertTrue(config['gemini_configured'])
        self.assertNotIn('fixture-google-secret', json.dumps(config))
        def transport(req, **kwargs):
            payload = json.loads(req.data)
            evidence = json.loads(payload['contents'][0]['parts'][0]['text'])
            self.assertLessEqual(len(evidence), 24)
            item = evidence[0]
            return Response(completion([{'evidence_id': item['evidence_id'], 'value': item['value'],
                'alternative_explanation': 'No causal inference is supported.'}]))
        with patch('urllib.request.urlopen', side_effect=transport) as call, patch('beyond_possession.agents.load_environment'):
            status, body = request('POST', '/api/analyze', {'source': 'demo', 'backend': 'gemini', 'system': 'single'}, config['token'])
            self.assertEqual(status, 200)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(body['report']['analysis']['backend'], 'gemini')
            self.assertTrue(body['report']['analysis']['claims'])
            self.assertTrue(all(c['verification']=='verified' for c in body['report']['analysis']['claims']))
            self.assertNotIn('fixture-google-secret', json.dumps(body))
        with patch.object(server.GeminiBackend, 'check_ready', return_value={'available': True}) as check:
            self.assertEqual(request('GET', '/api/gemini-status')[1], {'available': True})
            check.assert_called_once()


class GeminiStartupTests(unittest.TestCase):
    def test_default_and_explicit_api_startup_never_prepare_ollama(self):
        import run
        cases = [({}, []), ({'AI_PROVIDER': 'ollama'}, ['--api']), ({'AI_PROVIDER': 'groq'}, ['--api'])]
        for env, args in cases:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), patch('sys.argv', ['run.py', '--no-browser']+args), patch('beyond_possession.cloud_ai.load_environment'), patch('beyond_possession.local_ai.prepare_local_ai') as install, patch('beyond_possession.local_ai.stop_local_ai') as stop, patch('beyond_possession.server.serve') as serve, patch('builtins.print'):
                self.assertEqual(run.main(), 0)
                self.assertEqual(os.environ['BEYOND_PREFERRED_BACKEND'], 'shared')
                install.assert_not_called(); stop.assert_not_called(); serve.assert_called_once()

    def test_conflicting_api_and_local_flags_are_rejected(self):
        import run
        with patch('sys.argv', ['run.py', '--api', '--local-ai']), patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as caught:
                run.main()
            self.assertEqual(caught.exception.code, 2)

    def test_google_dotenv_values_load_without_execution(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp)/'.env'
            path.write_text('AI_PROVIDER=gemini\nGEMINI_API_KEY="fixture-google"\nGEMINI_MODEL='+DEFAULT_GEMINI_MODEL+'\nUNKNOWN=$(false)\n')
            load_environment(path)
            self.assertEqual(os.environ['GEMINI_API_KEY'], 'fixture-google')
            self.assertEqual(os.environ['GEMINI_MODEL'], DEFAULT_GEMINI_MODEL)
            self.assertNotIn('UNKNOWN', os.environ)


if __name__ == '__main__':
    unittest.main()
