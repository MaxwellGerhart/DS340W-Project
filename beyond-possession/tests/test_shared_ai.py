import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch
from beyond_possession.shared_ai import SharedBackend, DEFAULT_SHARED_URL
from beyond_possession.cloud_ai import CloudAIError
from beyond_possession.data import demo
from beyond_possession.pipeline import analyze
from test_groq_api import Response


class SharedAiTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'GEMINI_API_KEY': 'fixture-personal-google-key'}, clear=True)
        self.environment.start()
        self.loader = patch('beyond_possession.shared_ai.load_environment')
        self.loader.start()

    def tearDown(self):
        self.loader.stop()
        self.environment.stop()

    def test_public_health_needs_no_key_and_does_not_generate(self):
        with patch('urllib.request.urlopen', return_value=Response({'service':'beyond-possession-shared-ai','available':True,'model':'fixture-model'})) as call:
            status = SharedBackend().check_ready()
            self.assertTrue(status['available'])
            request = call.call_args.args[0]
            self.assertEqual(request.full_url, DEFAULT_SHARED_URL+'/api/health')
            self.assertIsNone(request.data)
            self.assertNotIn('fixture-personal-google-key', str(request.header_items()))

    def test_scoped_session_is_reused_and_google_key_is_never_forwarded(self):
        calls=[]
        def transport(request, **kwargs):
            calls.append(request)
            self.assertNotIn('fixture-personal-google-key', str(request.header_items()))
            if request.full_url.endswith('/api/session'):
                self.assertEqual(json.loads(request.data), {})
                return Response({'token':'fixture-session-token-at-least-20-characters'})
            self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-session-token-at-least-20-characters')
            return Response({'claims':[],'model':'fixture-model','cached':True,'usage':{'input_tokens':0,'output_tokens':0}})
        with patch('urllib.request.urlopen', side_effect=transport):
            model=SharedBackend()
            self.assertEqual(model.select('spatial', [], 42), [])
            self.assertEqual(model.select('passing', [], 42), [])
            self.assertEqual(len(calls), 3)
            self.assertTrue(model.calls[-1]['cached'])
            self.assertEqual(model.model, 'fixture-model')
            self.assertNotIn('fixture-session', json.dumps(model.calls))

    def test_quota_is_not_retried_and_session_is_redacted(self):
        token='fixture-session-token-at-least-20-characters'
        error=urllib.error.HTTPError(DEFAULT_SHARED_URL,429,'quota',{},io.BytesIO(json.dumps({'error':'Quota '+token}).encode()))
        with patch('urllib.request.urlopen', side_effect=error) as call:
            model=SharedBackend();model._token=token;model._expires=float('inf')
            with self.assertRaisesRegex(CloudAIError,'HTTP 429') as caught:
                model.select('spatial', [], 42)
            self.assertEqual(call.call_count,1)
            self.assertNotIn(token,str(caught.exception))

    def test_public_service_cannot_be_an_insecure_or_credential_url(self):
        for url in ['http://example.com','https://name:password@example.com','https://example.com?key=value']:
            with self.subTest(url=url), self.assertRaises(CloudAIError):
                SharedBackend(url)

    def test_shared_claims_reach_the_existing_numerical_critic(self):
        def transport(request, **kwargs):
            if request.full_url.endswith('/api/session'):
                return Response({'token':'fixture-session-token-at-least-20-characters'})
            evidence=json.loads(request.data)['evidence']
            self.assertLessEqual(len(evidence),24)
            item=evidence[0]
            return Response({'claims':[{'evidence_id':item['evidence_id'],'value':item['value']+1,
                'alternative_explanation':'No causal inference is supported.'}],'model':'fixture-model','usage':{}})
        rows,_=demo()
        with patch('urllib.request.urlopen',side_effect=transport),patch('beyond_possession.agents.load_environment'):
            report=analyze(rows,backend='shared',system='single')
        self.assertEqual(report['analysis']['backend'],'shared')
        self.assertEqual(len(report['analysis']['claims']),0)
        self.assertEqual(len(report['analysis']['rejected']),1)


if __name__=='__main__':unittest.main()
