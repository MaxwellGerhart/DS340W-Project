"""Public demo Gemini access. The provider key stays on the hosted service."""
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from .cloud_ai import CloudAIError, compact_evidence, load_environment
from .gemini_ai import DEFAULT_GEMINI_MODEL
from .tls import verified_context

DEFAULT_SHARED_URL = 'https://beyond-possession-ds340w-ai.arnavbedi123.chatgpt.site'


class SharedBackend:
    def __init__(self, base_url=None):
        load_environment()
        self.base_url = (base_url or os.getenv('SHARED_AI_BASE_URL') or DEFAULT_SHARED_URL).rstrip('/')
        url = urlsplit(self.base_url)
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise CloudAIError('The shared AI service must use a valid HTTPS URL.')
        self.model = DEFAULT_GEMINI_MODEL
        self.calls = []
        self._token = None
        self._expires = 0

    def _request(self, path, payload=None, authenticated=False):
        if authenticated and (not self._token or time.monotonic() >= self._expires):
            session = self._request('/api/session', {})
            token = session.get('token')
            if not isinstance(token, str) or not 20 <= len(token) <= 2000:
                raise CloudAIError('The shared AI service returned an invalid session.')
            self._token = token
            self._expires = time.monotonic() + 840
        headers = {'Content-Type': 'application/json', 'User-Agent': 'Beyond-Possession/0.1'}
        if authenticated:
            headers['Authorization'] = 'Bearer '+self._token
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.base_url+path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120, context=verified_context()) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError('Invalid service response')
            return value
        except urllib.error.HTTPError as error:
            try:
                body = error.read(32768).decode('utf-8', errors='replace')
                try: message = str(json.loads(body).get('error', 'Shared AI request failed.'))
                except (ValueError, TypeError): message = 'Shared AI request failed.'
                if self._token: message = message.replace(self._token, '[redacted]')
                code = error.code
            finally:
                error.close()
            raise CloudAIError(f'Shared Gemini service returned HTTP {code}: {message[:1000]} No automatic retry was made.') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise CloudAIError('Could not reach shared Gemini AI. Check internet access, retry later, or select your own Google Gemini API key.') from None
        except (ValueError, TypeError):
            raise CloudAIError('The shared AI service returned an unreadable response.') from None

    def check_ready(self):
        status = self._request('/api/health')
        if status.get('service') != 'beyond-possession-shared-ai' or not status.get('available'):
            raise CloudAIError('Shared Gemini AI is not available yet. Try again later or use your own Google API key.')
        self.model = status.get('model') or self.model
        return {'available': True, 'provider': 'Shared Google Gemini', 'model': self.model,
                'note': 'The service keeps the API key on its server. Google availability and shared demo limits still apply.'}

    def select(self, role, evidence, seed):
        start = time.monotonic()
        result = self._request('/api/select', {'role': role, 'evidence': compact_evidence(evidence), 'seed': seed}, authenticated=True)
        claims = result.get('claims')
        if not isinstance(claims, list) or len(claims) > 8:
            raise CloudAIError('Shared Gemini AI returned invalid structured observations.')
        self.model = result.get('model') or self.model
        usage = result.get('usage') or {}
        self.calls.append({'role': role, 'seconds': time.monotonic()-start,
                           'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
                           'cached': result.get('cached') is True})
        return claims
