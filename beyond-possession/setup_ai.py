"""Optional: prepare the local AI cache without starting the dashboard."""
from beyond_possession.local_ai import prepare_local_ai, stop_local_ai, SetupError

if __name__ == '__main__':
    try:
        process=prepare_local_ai()
        stop_local_ai(process)
    except SetupError as error:
        raise SystemExit(str(error))
