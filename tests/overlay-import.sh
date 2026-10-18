#!/usr/bin/env bash
# Witness that every overlay of serve.sh compiles and imports in the pinned image, mounted where serve.sh mounts it,
# in a throwaway CPU container without network (a serving container is not touched). Then, under VLLM_FP8_HYBRID=1:
# the shim hooks modelopt.py, the QSA indexer projection is kept apart, and the draft-vocabulary hook is active.
#   tests/overlay-import.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.32.0}"
M=()
while IFS= read -r line; do M+=("$line"); done < <(env -u SERVE_CONFIG PATCHES="$REPO/patches" bash "$REPO/serve/serve.sh" --print \
  | awk 'prev == "-v" && $0 ~ /\.py:/ {print "-v"; print} {prev = $0}')
docker run -i --rm --network none -e CUDA_VISIBLE_DEVICES= -e VLLM_FP8_HYBRID=1 --memory 6g "${M[@]}" \
  -e VLLM_MTP_DRAFT_VOCAB=/v/draft_vocab_65536.npy -v "$REPO/patches/draft-vocab":/v:ro \
  --entrypoint python3 "$IMAGE" - <<'PY' 2>&1 | grep -v -i -E 'warning|vllm\._C|cuda runtime|cuda_home'
import importlib, inspect, os, py_compile, sys
ok = ko = 0
def check(cond, msg):
    global ok, ko
    print(("OK    " if cond else "FAIL  ") + msg, flush=True); ok += bool(cond); ko += (not cond)
root = "/usr/local/lib/python3.12/dist-packages/"
# qsa.py and model.py import each other: the model goes first, as vLLM loads it (true of the upstream file too).
mods = ["vllm.model_executor.layers.quantization.modelopt", "vllm.models.qwen4_exp.nvidia.model",
        "vllm.models.qwen4_exp.nvidia.qsa", "vllm.models.qwen4_exp.nvidia.mtp", "vllm.v1.core.sched.scheduler", "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
        "vllm.entrypoints.openai.responses.protocol", "vllm.entrypoints.openai.responses.serving",
        "vllm.parser.abstract_parser", "vllm.parser.engine.adapters", "vllm.parser.engine.parser_engine",
        "vllm.parser.engine.parser_engine_config", "vllm.parser.engine.streaming_parser_engine",
        "vllm.parser.nemotron_v3", "vllm.parser.qwen3", "vllm.config.engram",
        "vllm.models.qwen4_exp.common.ngram_embedding", "vllm.models.qwen4_exp.nvidia.model_state",
        "vllm.models.qwen4_exp.nvidia.ngram_embedding", "vllm.models.qwen4_exp.nvidia.ple_pageable",
        "vllm_fp8_hybrid_modelopt"]
for m in mods:
    try:
        importlib.import_module(m); check(True, f"import {m}")
    except Exception as e:
        check(False, f"import {m}: {type(e).__name__}: {e}")
import vllm.models.qwen4_exp.nvidia.qsa as qsa
check(qsa._fp8_hybrid(), "VLLM_FP8_HYBRID=1 seen by qsa.py")
src = inspect.getsource(qsa.Qwen4ExpQSAAttention.__init__)
check("and not _fp8_hybrid()" in src and "indexer.index_qk_proj" in src, "QSA indexer projection kept apart under VLLM_FP8_HYBRID")
import vllm.models.qwen4_exp.nvidia.mtp as mtp
check(mtp.Qwen4ExpMTP.compute_logits.__name__ == "_dv_compute_logits", "draft vocabulary hook active")
import vllm.model_executor.layers.quantization.modelopt as mo
check("vllm_fp8_hybrid_modelopt" in inspect.getsource(mo), "modelopt.py imports the FP8 shim")
import vllm.parser.qwen3 as q3
check("guard" in inspect.getsource(q3).lower() or "preamble" in inspect.getsource(q3).lower(), "repaired qwen3 parser mounted")
print(f"\n{ok} OK, {ko} FAIL"); sys.exit(1 if ko else 0)
PY
exit "${PIPESTATUS[0]}"
