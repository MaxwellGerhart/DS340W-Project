"""Start the public download with shared Gemini access and no local key setup."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def launch(app=None, arguments=None):
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11 or newer is needed. On Windows, double-click START_WINDOWS.bat to prepare Python automatically.')
    app = (Path(app) if app is not None else ROOT/'beyond-possession').resolve()
    if not (app/'run.py').is_file():
        raise RuntimeError('Extract the entire ZIP before starting. Keep beyond-possession beside start.py.')
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments == ['--check']:
        print('Public download is ready. Shared Gemini access requires no local API key.')
        return 0
    print('Starting Beyond Possession with shared Google Gemini AI.', flush=True)
    print('Keep this window open. Press Ctrl+C to stop.', flush=True)
    return subprocess.run([sys.executable, str(app/'run.py'), '--api', *arguments], cwd=app, check=False).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(launch())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except (OSError, RuntimeError) as error:
        print('Could not start: '+str(error), file=sys.stderr)
        raise SystemExit(1)
