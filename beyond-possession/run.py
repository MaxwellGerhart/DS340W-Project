"""Open this file in VS Code and click Run Python File, or press F5."""
import argparse
import os
import sys
import webbrowser


def main():
    if sys.version_info < (3, 11):
        raise SystemExit('Select Python 3.11 or newer using VS Code: Python: Select Interpreter.')
    from beyond_possession.server import serve

    parser = argparse.ArgumentParser(description='Start the Beyond Possession dashboard')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--api', action='store_true', help='Use Google Gemini API without installing or starting Ollama')
    mode.add_argument('--local-ai', action='store_true', help='Use and prepare the optional Ollama model')
    mode.add_argument('--no-ai', action='store_true', help='Run built-in analysis without calling an AI model')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser automatically')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    from beyond_possession.cloud_ai import DEFAULT_PROVIDER, load_environment, key_present
    from beyond_possession.gemini_ai import DEFAULT_GEMINI_MODEL, gemini_key_present
    load_environment()
    process=None
    provider='offline' if args.no_ai else ('gemini' if args.api else ('ollama' if args.local_ai else os.getenv('AI_PROVIDER',DEFAULT_PROVIDER)))
    if provider not in {'gemini','groq','ollama','offline'}:
        parser.error('AI_PROVIDER must be gemini, groq, ollama, or offline.')
    os.environ['BEYOND_PREFERRED_BACKEND']=provider
    os.environ.pop('BEYOND_AI_SETUP_ERROR',None)
    if provider=='ollama':
        from beyond_possession.local_ai import prepare_local_ai, stop_local_ai, SetupError
        try:process=prepare_local_ai()
        except SetupError as error:
            print(str(error), file=sys.stderr)
            print('Starting the dashboard with built-in analysis.', file=sys.stderr)
            os.environ['BEYOND_AI_SETUP_ERROR']=str(error)
    elif provider=='gemini':
        print('Google Gemini API mode. Model: '+os.getenv('GEMINI_MODEL',DEFAULT_GEMINI_MODEL))
        print('AI runs through Google. No Ollama installation or model download is needed.')
        if not gemini_key_present():print('Add GEMINI_API_KEY to .env, then restart. Create a key at https://aistudio.google.com/apikey. Match browsing and built-in analysis are available meanwhile.')
    elif provider=='groq':
        print('Groq API mode. No Ollama installation or model download is needed.')
        if not key_present():print('Add your free Groq key to .env, then restart. The dashboard can browse matches and run built-in analysis without it.')
    try:
        serve(port=args.port, on_ready=None if args.no_browser else webbrowser.open)
    except OSError as error:
        print(f'Could not start the dashboard: {error}', file=sys.stderr)
        print('If another copy is running, stop it first. You can also run: python run.py --port 8766', file=sys.stderr)
        return 1
    finally:
        if process is not None:stop_local_ai(process)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
