#!/usr/bin/env bash
# Witness that every overlay of serve.sh compiles and imports in the pinned image, mounted where serve.sh mounts it,
# in a throwaway CPU container without network (a serving container is not touched). Then, under VLLM_FP8_HYBRID=1:
# the shim hooks modelopt.py, the QSA indexer projection is kept apart, and the draft-vocabulary hook is active; the
# drafter's M-RoPE cache is floored on max_model_len while the YaRN target is unchanged; the key guards /invocations.
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
        "vllm.model_executor.layers.rotary_embedding.mrope", "vllm.entrypoints.serve.middleware.authenticate",
        "vllm.model_executor.utils", "vllm.model_executor.parameter", "vllm.model_executor.layers.linear",
        "vllm.model_executor.layers.vocab_parallel_embedding", "vllm.model_executor.model_loader.weight_utils",
        "vllm.model_executor.layers.fused_moe.routed_experts", "vllm_fp8_hybrid_modelopt"]
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
import types, warnings
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
base = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000}
rows = {}
for name, rp in (("drafter", dict(base, rope_type="default")),
                 ("target", dict(base, rope_type="yarn", factor=2.0, original_max_position_embeddings=262144))):
    vc = VllmConfig(device_config=DeviceConfig(device="cpu"))
    object.__setattr__(vc, "model_config", types.SimpleNamespace(max_model_len=524288, multimodal_config=None))
    with set_current_vllm_config(vc), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows[name] = get_rope(head_size=256, max_position=262144, rope_parameters=rp, dtype=torch.float32,
                              mrope_positions_factor=1).cos_sin_cache.shape[0]
check(rows["drafter"] == 524288, f"drafter M-RoPE cache floored on max_model_len: {rows['drafter']} rows")
check(rows["target"] == 524288, f"YaRN target M-RoPE cache unchanged: {rows['target']} rows")
import asyncio
import vllm.entrypoints.serve.middleware.authenticate as au
async def _app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []}); await send({"type": "http.response.body", "body": b""})
def status(method, path, key):
    got = {}
    async def send(m):
        if m["type"] == "http.response.start": got["s"] = m["status"]
    async def receive(): return {"type": "http.request", "body": b"", "more_body": False}
    hdr = [(b"authorization", b"Bearer " + key)] if key else []
    asyncio.run(au.AuthenticationMiddleware(_app, ["k"])({"type": "http", "method": method, "path": path, "root_path": "",
                                                         "headers": hdr, "query_string": b""}, receive, send))
    return got["s"]
check((status("POST", "/invocations", None), status("POST", "/invocations", b"k"), status("GET", "/health", None))
      == (401, 200, 200), "key required on /invocations, GET /health open")
import torch, vllm.model_executor.utils as wu
sites = {m: open(importlib.import_module(m).__file__).read().count("copy_weight_(") for m in
         ("vllm.model_executor.parameter", "vllm.model_executor.layers.linear", "vllm.model_executor.layers.vocab_parallel_embedding",
          "vllm.model_executor.model_loader.weight_utils", "vllm.model_executor.layers.fused_moe.routed_experts")}
check(sum(sites.values()) == 26, f"weight loaders: 26 copies through copy_weight_ ({sites})")
d, s = torch.zeros(4096, dtype=torch.uint8), torch.arange(4096).to(torch.uint8)
wu.copy_weight_(d, s); check(torch.equal(d, s), "copy_weight_ copies byte for byte (CPU, the touch is off)")
print(f"\n{ok} OK, {ko} FAIL"); sys.exit(1 if ko else 0)
PY
exit "${PIPESTATUS[0]}"
