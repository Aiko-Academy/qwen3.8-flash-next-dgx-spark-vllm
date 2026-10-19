#!/usr/bin/env bash
# Every witness that needs neither Docker nor a server: serve.sh --print, tool-call classification, load bench
# request bodies, reasoning modes, sha256 of the copied files, credits. Any machine with bash and python3.
#   tests/offline.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0
run() { printf '== %s\n' "$1"; shift; "$@" || FAIL=$((FAIL + 1)); }
run "serve.sh --print" bash "$REPO/tests/serve-print.sh"
run "tool-call classification" python3 -I "$REPO/bench/toolcall.py" --self-check
run "load bench request bodies" python3 -I "$REPO/bench/load_bench.py" --self-check
run "reasoning modes" python3 -I "$REPO/bench/reasoning_modes.py"
if command -v sha256sum >/dev/null; then
  run "MANIFEST.sha256" sh -c "cd '$REPO' && sha256sum -c --quiet MANIFEST.sha256 && echo 'copied files as published'"
else
  run "MANIFEST.sha256" sh -c "cd '$REPO' && shasum -a 256 -c --quiet MANIFEST.sha256 && echo 'copied files as published'"
fi
run "credits" python3 -I "$REPO/tools/check_credits.py"
printf '\n%s\n' "$([ $FAIL -eq 0 ] && echo 'offline witnesses: all green' || echo "offline witnesses: $FAIL red")"
[ $FAIL -eq 0 ]
