"""Bounded narrative/review calls using existing provider credentials and TLS."""
import json
import os
import re
import time
import urllib.error
import urllib.request

from .cloud_ai import CloudAIError
from .narrative import task_spec
from .tls import verified_context


def request_task(backend, provider, task, payload, seed):
    prompt, schema, key = task_spec(task, payload)
    encoded = json.dumps(payload, allow_nan=False, separators=(',', ':'))
    if len(encoded.encode()) > 24000:
        raise CloudAIError('Narrative or critic input exceeds the bounded 24 KB task budget.')
    model = backend.model
    if provider == 'gemini' and task == 'review':
        model = (os.getenv('GEMINI_CRITIC_MODEL') or model).removeprefix('models/')
        if not re.fullmatch(r'gemini-[A-Za-z0-9._-]+', model):
            raise CloudAIError('GEMINI_CRITIC_MODEL must be a valid Gemini model ID.')
    start = time.perf_counter()
    if provider == 'gemini':
        data = backend._request('/models/'+model+':generateContent', {
            'systemInstruction': {'parts': [{'text': prompt}]},
            'contents': [{'role': 'user', 'parts': [{'text': encoded}]}],
            'generationConfig': {'seed': seed, 'maxOutputTokens': 4096,
                'responseMimeType': 'application/json', 'responseJsonSchema': schema}})
        try:
            candidate = data['candidates'][0]
            if candidate.get('finishReason') != 'STOP':
                raise ValueError('Incomplete response')
            text = ''.join(p['text'] for p in candidate['content']['parts'] if 'text' in p and not p.get('thought'))
            usage = data.get('usageMetadata') or {}
            tokens = (usage.get('promptTokenCount'), usage.get('candidatesTokenCount'))
        except (KeyError, TypeError, ValueError, IndexError):
            raise CloudAIError('Google did not complete the '+task+' task. No template result was substituted.') from None
    elif provider == 'groq':
        data = backend._request('/chat/completions', {'model': model, 'temperature': 0, 'seed': seed,
            'max_completion_tokens': 4096, 'stream': False,
            'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': encoded}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'soccer_'+task, 'strict': True, 'schema': schema}}})
        try:
            if data['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('Incomplete response')
            text = data['choices'][0]['message']['content']
            usage = data.get('usage') or {}
            tokens = (usage.get('prompt_tokens'), usage.get('completion_tokens'))
        except (KeyError, TypeError, ValueError, IndexError):
            raise CloudAIError('Groq did not complete the '+task+' task.') from None
    elif provider == 'ollama':
        request = urllib.request.Request(backend.base_url+'/api/chat', data=json.dumps({
            'model': model, 'stream': False, 'format': schema,
            'options': {'temperature': 0, 'seed': seed, 'num_ctx': 16384},
            'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': encoded}]}).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=300, context=verified_context()) as response:
                data = json.load(response)
            text = data['message']['content']
            tokens = (data.get('prompt_eval_count'), data.get('eval_count'))
        except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError):
            raise CloudAIError('Ollama could not complete the '+task+' task. Check the local model and connection.') from None
    else:
        raise ValueError('Unsupported task provider.')
    try:
        output = json.loads(text)
        if not isinstance(output, dict) or not isinstance(output.get(key), list) or len(output[key]) > 8:
            raise ValueError('Invalid output list')
    except (ValueError, TypeError):
        raise CloudAIError('The '+task+' task returned malformed JSON. No unchecked output was substituted.') from None
    backend.calls.append({'role': payload.get('role', 'tactical_critic')+'_'+task,
        'model': model, 'seconds': time.perf_counter()-start, 'input_tokens': tokens[0], 'output_tokens': tokens[1], 'cached': False})
    return output
