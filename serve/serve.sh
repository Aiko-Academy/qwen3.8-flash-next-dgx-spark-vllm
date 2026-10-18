#!/usr/bin/env bash
# serve.sh: run Qwen3.8-Flash-Next on one DGX Spark with this recipe, and only this recipe.
#
#   serve/serve.sh            check every input, then (re)create the container
#   serve/serve.sh --print    print the docker run, one argument per line; start nothing, check nothing
#   serve/serve.sh --check    run every check below and stop there; the running container is not touched
#
# Image vllm/vllm-openai:v0.31.0 (pinned by digest), the NVIDIA NVFP4 checkpoint with its dense layers in block FP8
# (serve/prepare.sh), 524 288 tokens of context (YaRN factor 2), MTP 2 with a 65 536-row NVFP4 draft head, 16
# sequences on the ten Cortex-X925 cores, and the overlays of patches/ mounted read-only on the image.
#
# Configuration by environment variables, or by a file named in SERVE_CONFIG (serve/config.example.env lists them
# all). The API key is never an argument: API_KEY_FILE names a docker --env-file that holds VLLM_API_KEY=<key>.
# Without it, the server answers anyone who reaches the port, so it only binds 127.0.0.1.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${SERVE_CONFIG:-}" ]; then
  # shellcheck disable=SC1090
  . "$SERVE_CONFIG"
fi

IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3}"
CONTAINER="${CONTAINER:-vllm-flashnext}"
MODEL_DIR="${MODEL_DIR:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia-fp8hybrid}"
CACHE_DIR="${CACHE_DIR:-/var/tmp/qwen38fn-vllm-cache}"
BIND="${BIND:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY_FILE="${API_KEY_FILE:-}"
CPUSET="${CPUSET:-5-9,15-19}"
PATCHES="${PATCHES:-$REPO/patches}"
BUILD="${BUILD:-$REPO/build}"
PLE_OVERLAY="${PLE_OVERLAY:-$PATCHES/ple-checkpoint-mapped/vllm-overlay}"
FP8_HYBRID="${FP8_HYBRID:-$PATCHES/fp8-hybrid}"
CACHE_DET="${CACHE_DET:-$PATCHES/cache-det}"
DET_LIB="${DET_LIB:-$BUILD/_C_det.so}"
DRAFT_HEAD="${DRAFT_HEAD:-$PATCHES/draft-nvfp4-head}"
DRAFT_VOCAB="${DRAFT_VOCAB:-$BUILD/draft_vocab_public_65536.npy}"
DRAFT_VOCAB_DIR="${DRAFT_VOCAB_DIR:-/opt/draft-vocab}"
MTP_KV="${MTP_KV:-$PATCHES/mtp-kv-bf16}"
TOOL_PARSER="${TOOL_PARSER:-$PATCHES/tool-parser}"
VERIFIED_MARKER="${VERIFIED_MARKER:-.verified}"

VP=/usr/local/lib/python3.12/dist-packages/vllm   # the vllm package inside the image
CTX=524288
VOCAB_IN="$DRAFT_VOCAB_DIR/$(basename "$DRAFT_VOCAB")"

ENVS=(-e VLLM_USE_DEEP_GEMM=0 -e VLLM_FP8_HYBRID=1 -e VLLM_MTP_DRAFT_NVFP4=1
  -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/kernel-det/_C_det.so
  -e VLLM_MTP_DRAFT_VOCAB="$VOCAB_IN")

# Mounts, in the order of the reference container: dense FP8 shim, prefix cache and deterministic top-k, draft
# head and its vocabulary, MTP-layer KV in BF16, repaired tool parser; then checkpoint, cache, and the PLE overlay.
MOUNTS=(-v "$FP8_HYBRID/vllm_fp8_hybrid_modelopt.py:${VP%/vllm}/vllm_fp8_hybrid_modelopt.py:ro"
  -v "$FP8_HYBRID/vllm-overlay/model_executor/layers/quantization/modelopt.py:$VP/model_executor/layers/quantization/modelopt.py:ro"
  -v "$CACHE_DET/vllm-overlay/v1/worker/gpu/model_states/mamba_hybrid.py:$VP/v1/worker/gpu/model_states/mamba_hybrid.py:ro"
  -v "$CACHE_DET/vllm-overlay/v1/core/sched/scheduler.py:$VP/v1/core/sched/scheduler.py:ro"
  -v "$CACHE_DET/vllm-overlay/models/qwen4_exp/nvidia/ops/qsa_indexer.py:$VP/models/qwen4_exp/nvidia/ops/qsa_indexer.py:ro"
  -v "$DET_LIB:/opt/kernel-det/_C_det.so:ro"
  -v "$DRAFT_HEAD/vllm-overlay/models/qwen4_exp/nvidia/mtp.py:$VP/models/qwen4_exp/nvidia/mtp.py:ro"
  -v "$DRAFT_VOCAB:$VOCAB_IN:ro"
  -v "$MTP_KV/vllm-overlay/models/qwen4_exp/nvidia/qsa.py:$VP/models/qwen4_exp/nvidia/qsa.py:ro")
for f in entrypoints/openai/responses/protocol.py entrypoints/openai/responses/serving.py parser/abstract_parser.py \
         parser/engine/adapters.py parser/engine/parser_engine.py parser/engine/parser_engine_config.py \
         parser/engine/streaming_parser_engine.py parser/nemotron_v3.py parser/qwen3.py; do
  MOUNTS+=(-v "$TOOL_PARSER/vllm-overlay/$f:$VP/$f:ro")
done
MOUNTS+=(-v "$MODEL_DIR:/models/flashnext:ro" -v "$CACHE_DIR:/root/.cache")
for f in config/engram.py models/qwen4_exp/common/ngram_embedding.py models/qwen4_exp/nvidia/model_state.py \
         models/qwen4_exp/nvidia/ngram_embedding.py models/qwen4_exp/nvidia/ple_pageable.py; do
  MOUNTS+=(-v "$PLE_OVERLAY/vllm/$f:$VP/$f:ro")
done

KEYFILE=()
[ -n "$API_KEY_FILE" ] && KEYFILE=(--env-file "$API_KEY_FILE")

ARGS=(serve /models/flashnext --served-model-name qwen3.8-flash-next
  --trust-remote-code --quantization modelopt --tensor-parallel-size 1
  --max-model-len "$CTX" --gpu-memory-utilization 0.80 --kv-cache-dtype fp8
  --max-num-seqs 16 --max-num-batched-tokens 4096
  --engram-config '{"checkpoint_mapped": true}'
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --default-chat-template-kwargs '{"enable_thinking": false}'
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2,\"max_model_len\":$CTX}"
  --no-enable-flashinfer-autotune
  --host 0.0.0.0 --port 8000
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 2.0, "original_max_position_embeddings": 262144}}}'
  --kv-cache-dtype-skip-layers 48
  --mamba-ssm-cache-dtype bfloat16)

RUN=(docker run -d --name "$CONTAINER" --restart unless-stopped
  --gpus all --ipc host --shm-size 32g --ulimit memlock=-1:-1 --ulimit stack=67108864
  -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  --cpuset-cpus "$CPUSET" "${ENVS[@]}" "${MOUNTS[@]}"
  -p "$BIND:$PORT:8000" ${KEYFILE[@]+"${KEYFILE[@]}"}
  --entrypoint vllm "$IMAGE" "${ARGS[@]}")

if [ "${1:-}" = --print ]; then
  printf '%s\n' "${RUN[@]}"
  exit 0
fi
[ $# -eq 0 ] || [ "$1" = --check ] || { echo "usage: $(basename "$0") [--print|--check]" >&2; exit 2; }

# Checks before anything is stopped: every mounted file, the converted checkpoint, the image digest, the key.
fail() { echo "serve.sh: $*" >&2; exit 3; }
i=0
while [ $i -lt ${#MOUNTS[@]} ]; do
  src="${MOUNTS[$((i + 1))]%%:*}"
  case "$src" in
    "$CACHE_DIR") mkdir -p "$CACHE_DIR" ;;
    "$MODEL_DIR") [ -d "$src" ] || fail "checkpoint absent: $src (serve/prepare.sh)" ;;
    *) [ -s "$src" ] || fail "missing file: $src (serve/prepare.sh)" ;;
  esac
  i=$((i + 2))
done
[ -f "$MODEL_DIR/.prepared" ] && [ -f "$MODEL_DIR/$VERIFIED_MARKER" ] \
  || fail "checkpoint not converted or not verified: $MODEL_DIR (serve/prepare.sh convert)"
digests="$(docker image inspect "$IMAGE" --format '{{join .RepoDigests " "}}' 2>/dev/null)" \
  || fail "image absent: $IMAGE (docker pull $IMAGE)"
case " $digests " in
  *"@$IMAGE_DIGEST "*) ;;
  *) [ "${ALLOW_UNPINNED_IMAGE:-0}" = 1 ] || fail "$IMAGE is not $IMAGE_DIGEST (got: $digests); ALLOW_UNPINNED_IMAGE=1 to override" ;;
esac
if [ -n "$API_KEY_FILE" ]; then
  grep -q '^VLLM_API_KEY=.' "$API_KEY_FILE" || fail "$API_KEY_FILE holds no VLLM_API_KEY=<key> line"
elif [ "$BIND" != 127.0.0.1 ]; then
  fail "no API_KEY_FILE: refusing to bind $BIND without a key"
fi

if [ "${1:-}" = --check ]; then
  echo "serve.sh: every check passed; nothing started"; exit 0
fi
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
"${RUN[@]}"
echo "serve.sh: $CONTAINER started on $BIND:$PORT; the first completion takes 12 to 18 minutes (docker logs -f $CONTAINER)"
