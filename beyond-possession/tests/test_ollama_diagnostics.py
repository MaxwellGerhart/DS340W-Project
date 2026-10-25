import io
import json
import unittest
import urllib.error
from unittest.mock import patch
from beyond_possession.agents import OllamaBackend, ModelError

class Response:
    def __init__(self,data):self.data=data
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return json.dumps(self.data).encode()

class DiagnosticsTests(unittest.TestCase):
    def test_service_missing(self):
        with patch('urllib.request.urlopen',side_effect=urllib.error.URLError('connection refused')):
            with self.assertRaisesRegex(ModelError,'Cannot reach Ollama'):
                OllamaBackend().check_ready()
    def test_model_missing(self):
        with patch('urllib.request.urlopen',return_value=Response({'models':[]})):
            with self.assertRaisesRegex(ModelError,'ollama pull qwen2.5:3b'):
                OllamaBackend().check_ready()
    def test_installed_model(self):
        with patch('urllib.request.urlopen',return_value=Response({'models':[{'name':'qwen2.5:3b'}]})):
            self.assertTrue(OllamaBackend().check_ready()['available'])
    def test_memory_error_body_preserved(self):
        error=urllib.error.HTTPError('http://localhost',500,'failure',{},io.BytesIO(b'{"error":"model requires more system memory"}'))
        with patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(ModelError,'model requires more system memory'):
                OllamaBackend().select('spatial',[],42)
    def test_invalid_model_output(self):
        with patch('urllib.request.urlopen',return_value=Response({'message':{'content':'not json'}})):
            with self.assertRaisesRegex(ModelError,'invalid structured output'):
                OllamaBackend().select('spatial',[],42)
