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
    parser.add_argument('--local-ai', action='store_true', help='Use and prepare Ollama instead of the Groq API')
    parser.add_argument('--no-ai', action='store_true', help='Skip all AI downloads and run the built-in analysis only')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser automatically')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    from beyond_possession.local_ai import prepare_local_ai, stop_local_ai, SetupError
    from beyond_possession.cloud_ai import load_environment, key_present
    load_environment()
    process=None
    provider='offline' if args.no_ai else ('ollama' if args.local_ai else os.getenv('AI_PROVIDER','ollama'))
    if provider not in {'groq','ollama','offline'}:
        parser.error('AI_PROVIDER must be groq, ollama, or offline.')
    os.environ['BEYOND_PREFERRED_BACKEND']=provider
    os.environ.pop('BEYOND_AI_SETUP_ERROR',None)
    if provider=='ollama':
        try:process=prepare_local_ai()
        except SetupError as error:
            print(str(error), file=sys.stderr)
            print('Starting the dashboard with built-in analysis.', file=sys.stderr)
            os.environ['BEYOND_AI_SETUP_ERROR']=str(error)
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
        stop_local_ai(process)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
