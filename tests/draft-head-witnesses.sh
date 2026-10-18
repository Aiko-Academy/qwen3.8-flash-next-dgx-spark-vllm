#!/usr/bin/env bash
# Witnesses of the served draft head (patches/draft-nvfp4-head/mtp.py) in the pinned image, throwaway containers
# without network. Any 65 536-id set works; blazux's published set is used here.
# CPU: the reduced-vocabulary hook is active; on a BF16 head the set's logits equal the reference and the others are
#      -inf; on an FP8 head without VLLM_MTP_DRAFT_HEAD_CKPT the hook refuses explicitly, and with it the set's rows
#      are the BF16 rows of the checkpoint, bit for bit.
# GPU (--gpu, a few seconds, can run beside a serving model): NVFP4 Marlin product of the 65 536 rows close to BF16
#      (cosine above 0.99) and faster than cuBLAS BF16 at M = 1.
#   tests/draft-head-witnesses.sh [--gpu]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
N=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
W="$(mktemp -d /var/tmp/test-draft-head.XXXXXX)"
trap 'docker run --rm -v "$W":/w --entrypoint rm "$IMAGE" -rf /w/ckpt >/dev/null 2>&1; rm -rf "$W"' EXIT
MOUNTS=(-v "$REPO/patches/draft-nvfp4-head/vllm-overlay/models/qwen4_exp/nvidia/mtp.py:$N/mtp.py:ro"
  -v "$REPO/patches/draft-vocab":/v:ro -v "$W":/w)
filter() { grep -v -i -E 'warning|vllm\._C|cuda runtime|cuda_home'; }

docker run -i --rm --network none -e CUDA_VISIBLE_DEVICES= --memory 6g "${MOUNTS[@]}" \
  -e VLLM_MTP_DRAFT_VOCAB=/v/draft_vocab_65536.npy --entrypoint python3 "$IMAGE" - <<'PY' 2>&1 | filter
import json, os, sys, types
import numpy as np
import torch
from safetensors.torch import save_file
ok = ko = 0
def check(cond, msg):
    global ok, ko
    print(("OK    " if cond else "FAIL  ") + msg, flush=True); ok += bool(cond); ko += (not cond)

import vllm.models.qwen4_exp.nvidia.mtp as m
f = m.Qwen4ExpMTP.compute_logits
check(f.__name__ == "_dv_compute_logits", "draft vocabulary hook active")
V_, H_ = 248320, 128
torch.manual_seed(0)
head = torch.randn(V_, H_, dtype=torch.bfloat16)
os.makedirs("/w/ckpt")
save_file({"lm_head.weight": head}, "/w/ckpt/m-9.safetensors", metadata={"format": "pt"})
json.dump({"metadata": {}, "weight_map": {"lm_head.weight": "m-9.safetensors"}}, open("/w/ckpt/model.safetensors.index.json", "w"))
ids = torch.from_numpy(np.load(os.environ["VLLM_MTP_DRAFT_VOCAB"]).astype(np.int64))
hs = torch.randn(3, H_, dtype=torch.bfloat16)
outside = torch.ones(V_, dtype=torch.bool); outside[ids] = False
bf = types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=head), logits_processor=types.SimpleNamespace(org_vocab_size=V_))
out = f(bf, hs)
check(torch.equal(out[:, ids], torch.nn.functional.linear(hs, head.index_select(0, ids))), "BF16 head: logits of the set equal the reference")
check(bool(torch.isinf(out[:, outside]).all()), "BF16 head: logits outside the set at -inf")
fp8 = types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=head.to(torch.float8_e4m3fn)), logits_processor=types.SimpleNamespace(org_vocab_size=V_))
try:
    f(fp8, hs); check(False, "FP8 head without VLLM_MTP_DRAFT_HEAD_CKPT: refused")
except RuntimeError as e:
    check("VLLM_MTP_DRAFT_HEAD_CKPT" in str(e), "FP8 head without VLLM_MTP_DRAFT_HEAD_CKPT: explicit refusal")
os.environ["VLLM_MTP_DRAFT_HEAD_CKPT"] = "/w/ckpt"
out2 = f(fp8, hs)
check(torch.equal(fp8._dv_state[1], head.index_select(0, ids)), "FP8 head: rows of the set = BF16 rows of the checkpoint, bit for bit")
check(torch.equal(out2, out), "FP8 head: same logits as the BF16 head")
print(f"\nCPU: {ok} OK, {ko} FAIL"); sys.exit(1 if ko else 0)
PY
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || exit "$rc"
[ "${1:-}" = --gpu ] || exit 0

docker run -i --rm --network none --gpus all --memory 8g "${MOUNTS[@]}" \
  -e VLLM_MTP_DRAFT_VOCAB=/v/draft_vocab_65536.npy -e VLLM_MTP_DRAFT_NVFP4=1 --entrypoint python3 "$IMAGE" - <<'PY' 2>&1 | filter
import sys
import torch
ok = ko = 0
def check(cond, msg):
    global ok, ko
    print(("OK    " if cond else "FAIL  ") + msg, flush=True); ok += bool(cond); ko += (not cond)
import vllm.models.qwen4_exp.nvidia.mtp as m
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
torch.manual_seed(0)
K, N = 2560, 65536
wk = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
q = m._dv_nvfp4(wk)
nv = lambda x: apply_fp4_marlin_linear(x, q.weight, q.weight_scale, q.weight_global_scale, q.workspace, N, K)
for M in (1, 3):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, wk).float(); got = nv(x).float()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    check(cos > 0.99, f"M={M}: NVFP4 against BF16 cosine {cos:.4f}")
def timed(fn, x, n=200):
    for _ in range(20): fn(x)
    torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n): fn(x)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n
x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
tb = timed(lambda y: torch.nn.functional.linear(y, wk), x); tn = timed(nv, x)
print(f"INFO  M=1: cuBLAS BF16 {tb * 1000:.0f} us, Marlin NVFP4 {tn * 1000:.0f} us", flush=True)
check(tn < tb, "M=1: Marlin NVFP4 faster than cuBLAS BF16")
print(f"\nGPU: {ok} OK, {ko} FAIL"); sys.exit(1 if ko else 0)
PY
exit "${PIPESTATUS[0]}"
