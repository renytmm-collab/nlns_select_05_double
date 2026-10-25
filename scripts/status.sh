#!/usr/bin/env bash
set -eu
if [ "$#" -ne 3 ]; then echo 'Usage: status.sh PROJECT_ROOT RUN_DIR PYTHON' >&2; exit 2; fi
cd "$1"
"$3" -B -m relation_transfer.pipeline status --run-dir "$2/pipeline"
if [ -f "$2/exit_code.txt" ]; then printf 'launcher_exit_code='; cat "$2/exit_code.txt"; fi
