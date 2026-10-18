#!/usr/bin/env bash
# A community protocol: llama-benchy (eugr/llama-benchy, MIT), run ON the Spark so that the network is not counted.
# Installed at a pinned version in a throwaway container of the vLLM image: bridge network, no capability, no new
# privilege, 4 GB, pinned on the Cortex-A725 cores (0-4) so that the client does not compete with the engine's
# cores. The tokenizer is read from the checkpoint, mounted read-only. The key comes from the docker --env-file of
# serve.sh and is passed inside the container only.
#   bench/community/llama-benchy.sh <label>       writes results/llama-benchy-<label>.md and .json
# Same configuration as serve.sh (SERVE_CONFIG): BIND, PORT, API_KEY_FILE, SOURCE_DIR.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -n "${SERVE_CONFIG:-}" ]; then
  # shellcheck disable=SC1090
  . "$SERVE_CONFIG"
fi
LABEL="${1:?usage: $(basename "$0") <label>}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
BIND="${BIND:-127.0.0.1}"; PORT="${PORT:-8000}"
SOURCE_DIR="${SOURCE_DIR:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia}"
VERSION=0.4.0
WHEEL_SHA=9d380f97bd5ef39b03ed2f319f1ee3bdf5c983463f6ce1479f432e90db079410
ARGS="--pp 2048 --tg 32 128 512 --depth 0 8192 --runs 3 --latency-mode generation"
host="$BIND"; [ "$host" = 0.0.0.0 ] && host=127.0.0.1
net=(--network bridge); [ "$host" = 127.0.0.1 ] && net=(--network host)
KEYENV=(); [ -n "${API_KEY_FILE:-}" ] && KEYENV=(--env-file "$API_KEY_FILE")
mkdir -p "$REPO/results"
OUT="$REPO/results/llama-benchy-$LABEL"
start="$(date +%Y-%m-%dT%H:%M:%S%z)"
docker run --rm "${net[@]}" --cap-drop ALL --security-opt no-new-privileges --memory 4g --cpuset-cpus 0-4 \
  -e CUDA_VISIBLE_DEVICES= ${KEYENV[@]+"${KEYENV[@]}"} -v "$SOURCE_DIR":/tok:ro --entrypoint bash "$IMAGE" -c "
    set -e
    pip download -q --no-deps -d /tmp/w llama-benchy==$VERSION >/dev/null
    echo '$WHEEL_SHA  /tmp/w/llama_benchy-$VERSION-py3-none-any.whl' | sha256sum -c --quiet
    pip install -q /tmp/w/llama_benchy-$VERSION-py3-none-any.whl >/dev/null 2>&1
    llama-benchy --base-url http://$host:$PORT/v1 --api-key \"\${VLLM_API_KEY:-EMPTY}\" --model qwen3.8-flash-next \
      --tokenizer /tok $ARGS" 2>&1 | grep -v -i -E 'warning|notice' | tee "$OUT.md"
python3 -I - "$OUT" "$start" "$VERSION" "$ARGS" "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)" <<'PY'
import json, re, sys, time
out, start, version, args, commit = sys.argv[1:6]
table = [l for l in open(out + ".md", encoding="utf-8") if l.startswith("|")]
rows = []
for l in table[2:]:
    cells = [c.strip() for c in l.strip().strip("|").split("|")]
    if len(cells) >= 3:
        rows.append({"test": cells[1], "t_s": cells[2], "peak_t_s": cells[3] if len(cells) > 3 else "",
                     "ttfr_ms": cells[4] if len(cells) > 4 else ""})
json.dump({"kind": "llama-benchy", "label": out.rsplit("llama-benchy-", 1)[1], "measured_at": start,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo_commit": commit,
           "protocol": {"tool": f"llama-benchy {version}", "arguments": args, "client": "on the Spark, cores 0-4"},
           "rows": rows, "requests": {"sent": None, "errors": None}}, open(out + ".json", "w"), indent=1)
print(f"written: results/llama-benchy-{out.rsplit('llama-benchy-', 1)[1]}.json ({len(rows)} rows)")
PY
