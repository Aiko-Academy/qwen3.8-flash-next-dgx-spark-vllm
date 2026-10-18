#!/usr/bin/env bash
# Offline witness of serve/serve.sh --print (no Docker, no Spark; bash 3.2 or later):
# the printed docker run equals tests/serve-print.golden (repository path replaced by <repo>), with 41 vLLM
# arguments, 25 mounts and 13 variables, bound to 127.0.0.1 without a key file by default; a configured bind and
# key file appear; and a serve.sh with one argument removed is caught.
#   tests/serve-print.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'OK    %s\n' "$1"; }
non() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }
clean() { env -u SERVE_CONFIG -u BIND -u API_KEY_FILE -u PATCHES -u BUILD -u MODEL_DIR -u CACHE_DIR -u CPUSET -u CONTAINER \
  -u IMAGE -u PLE_OVERLAY -u FP8_HYBRID -u CACHE_DET -u DET_LIB -u DRAFT_HEAD -u DRAFT_VOCAB -u DRAFT_VOCAB_DIR -u MTP_KV \
  -u TOOL_PARSER -u PORT "$@"; }

clean bash "$REPO/serve/serve.sh" --print | sed "s#$REPO#<repo>#g" > "$TMP/out"
if cmp -s "$TMP/out" "$REPO/tests/serve-print.golden"; then ok "output equals tests/serve-print.golden"
else non "output differs from the golden file"; diff "$REPO/tests/serve-print.golden" "$TMP/out" | head -10; fi
after=$(sed -n '/^--entrypoint$/,$p' "$TMP/out" | tail -n +4 | wc -l | tr -d ' ')
[ "$after" = 41 ] && ok "41 vLLM arguments after the image" || non "vLLM arguments: $after"
[ "$(grep -cx -- '-v' "$TMP/out")" = 25 ] && ok "25 mounts" || non "mounts: $(grep -cx -- '-v' "$TMP/out")"
[ "$(grep -cx -- '-e' "$TMP/out")" = 13 ] && ok "13 environment variables" || non "variables: $(grep -cx -- '-e' "$TMP/out")"
grep -qx '127.0.0.1:8000:8000' "$TMP/out" && ! grep -qx -- '--env-file' "$TMP/out" \
  && ok "default: bound to 127.0.0.1, no key file" || non "default bind or key file"
clean env BIND=10.0.0.5 API_KEY_FILE=/k.env bash "$REPO/serve/serve.sh" --print > "$TMP/conf"
grep -qx '10.0.0.5:8000:8000' "$TMP/conf" && grep -A1 -x -- '--env-file' "$TMP/conf" | grep -qx /k.env \
  && ok "configured bind and key file appear" || non "configured bind or key file missing"
mkdir -p "$TMP/m/serve"; sed '/^  --no-enable-flashinfer-autotune$/d' "$REPO/serve/serve.sh" > "$TMP/m/serve/serve.sh"
clean bash "$TMP/m/serve/serve.sh" --print | sed "s#$TMP/m#<repo>#g" > "$TMP/mut"
cmp -s "$TMP/mut" "$REPO/tests/serve-print.golden" && non "mutation not caught" || ok "a removed argument is caught"
printf '\n%d OK, %d FAIL\n' "$PASS" "$FAIL"; [ "$FAIL" -eq 0 ]
