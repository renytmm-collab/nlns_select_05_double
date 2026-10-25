#!/usr/bin/env bash
set -u
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo 'Usage: run_server.sh PROJECT_ROOT DATA_DIR RUN_DIR PYTHON [run|resume]' >&2
  exit 2
fi
PROJECT_ROOT="$1"
DATA_DIR="$2"
RUN_DIR="$3"
PYTHON="$4"
MODE="${5:-run}"
case "$PROJECT_ROOT:$DATA_DIR:$RUN_DIR" in /*:/*:/*) ;; *) echo 'Use absolute project/data/run paths' >&2; exit 2;; esac
case "$MODE" in run|resume) ;; *) exit 2;; esac
cd "$PROJECT_ROOT" || exit 2
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Logs live next to the actual pipeline directory within the user-selected envelope.
if [ "$MODE" = run ]; then
  (umask 077; mkdir "$RUN_DIR") || exit 2
else
  [ -d "$RUN_DIR/pipeline" ] || exit 2
fi
LOG="$RUN_DIR/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
"$PYTHON" -B -m relation_transfer.pipeline "$MODE" \
  --config "$PROJECT_ROOT/configs/xml100_default.json" \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR/pipeline" --confirm-real-run >"$LOG" 2>&1
CODE=$?
printf '%s\n' "$CODE" >"$RUN_DIR/exit_code.pending"
mv "$RUN_DIR/exit_code.pending" "$RUN_DIR/exit_code.txt"
exit "$CODE"
