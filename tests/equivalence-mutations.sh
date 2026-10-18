#!/usr/bin/env bash
# Witness of serve/verify.sh against the running container: green as is, red on two mutations of serve.sh
# (one vLLM argument removed, one mount removed). A comparison that cannot fail proves nothing.
# Read-only for the container; run it on the Spark with the configuration of the running service:
#   SERVE_CONFIG=~/flashnext.env tests/equivalence-mutations.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'OK    %s\n' "$1"; }
non() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }

"$REPO/serve/verify.sh" > "$TMP/as-is.log" 2>&1; rc=$?
[ $rc -eq 0 ] && ok "verify.sh green on the recipe as published ($(tail -1 "$TMP/as-is.log"))" \
  || { non "verify.sh red on the recipe as published"; grep FAIL "$TMP/as-is.log"; }

mutate() {  # $1 = name, $2 = sed expression applied to serve.sh in a copy of the repository
  local copy="$TMP/$1"
  mkdir -p "$copy"; cp -R "$REPO/serve" "$copy/"
  ln -s "$REPO/patches" "$copy/patches"
  sed -e "$2" "$REPO/serve/serve.sh" > "$copy/serve/serve.sh"
  if cmp -s "$REPO/serve/serve.sh" "$copy/serve/serve.sh"; then non "$1: the mutation changed nothing"; return; fi
  "$copy/serve/verify.sh" > "$copy/verify.log" 2>&1
  if [ $? -ne 0 ]; then ok "$1: verify.sh red ($(grep -c '^FAIL' "$copy/verify.log") FAIL lines)"
  else non "$1: verify.sh green on a mutated recipe"; fi
}
mutate "argument removed (--no-enable-flashinfer-autotune)" '/^  --no-enable-flashinfer-autotune$/d'
mutate "mount removed (qsa.py of mtp-kv-bf16)" '/-v "\$MTP_KV\/vllm-overlay\/models\/qwen4_exp\/nvidia\/qsa.py/s/^  -v "\$MTP_KV[^)]*)/  )/'
printf '\n%d OK, %d FAIL\n' "$PASS" "$FAIL"; [ "$FAIL" -eq 0 ]
