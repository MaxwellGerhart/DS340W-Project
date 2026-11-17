#!/bin/sh
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
for PYTHON in /usr/local/bin/python3 /opt/homebrew/bin/python3 python3 python3.14 python3.13 python3.12 python3.11; do
    if command -v "$PYTHON" >/dev/null 2>&1 && "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
        "$PYTHON" "$ROOT/start.py" "$@"
        RESULT=$?
        if [ "$RESULT" -ne 0 ]; then
            printf '\nPress Enter to close this window.'
            read -r ANSWER
        fi
        exit "$RESULT"
    fi
done
printf 'Install Python 3.11 or newer from https://www.python.org/downloads/macos/ and open this launcher again.\n'
printf 'Press Enter to close this window.'
read -r ANSWER
exit 1
