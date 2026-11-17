#!/bin/sh
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$ROOT/start.py" "$@"
fi
printf 'Install Python 3.11 or newer, then run this launcher again.\n' >&2
exit 1
