#!/usr/bin/env bash
# Witness of the streaming converter (patches/fp8-hybrid/convertir_denses_fp8.py) on a SYNTHETIC checkpoint of a
# few MB: its output must equal, tensor by tensor and bit for bit, that of fp8_convert.py (blazux) applied in place
# on a copy. Then `verifier` must be green, and red when one byte of an untouched tensor is flipped (a verification
# that cannot fail proves nothing). The converter prints in French: VERIFICATION VERTE / ROUGE.
# Run on a machine with Docker and the image (a throwaway CPU container, no network, no GPU):
#   tests/convert-dense-fp8-synthetic.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.32.0}"
P="$REPO/patches/fp8-hybrid"
W="$(mktemp -d /var/tmp/test-fp8conv.XXXXXX)"
trap 'docker run --rm -v "$W":/w --entrypoint rm "$IMAGE" -rf /w/src /w/a /w/dst >/dev/null 2>&1; rm -rf "$W"' EXIT

docker run -i --rm --network none -e CUDA_VISIBLE_DEVICES= --memory 4g \
  -v "$P":/p:ro -v "$W":/w --entrypoint python3 "$IMAGE" - <<'PY' 2>&1 | grep -v -i warning
import json, os, shutil, struct, subprocess, sys
import torch
from safetensors.torch import save_file, load_file
torch.manual_seed(0)
L = "model.language_model.layers"
src = "/w/src"; os.makedirs(src)
s1 = {f"{L}.0.linear_attn.in_proj_qkv.weight": torch.randn(256, 384, dtype=torch.bfloat16),
      f"{L}.0.linear_attn.in_proj_b.weight": torch.randn(16, 384, dtype=torch.bfloat16),   # not a target
      f"{L}.0.mlp.shared_expert.down_proj.weight": torch.randn(384, 128, dtype=torch.bfloat16),
      f"{L}.0.input_layernorm.weight": torch.randn(384, dtype=torch.bfloat16)}
s2 = {f"{L}.1.self_attn.q_proj.weight": torch.randn(512, 384, dtype=torch.bfloat16) * 3,
      "lm_head.weight": torch.randn(1000, 384, dtype=torch.bfloat16)}                    # not a target
s3 = {"model.language_model.embed_tokens.weight": torch.randn(1000, 384, dtype=torch.bfloat16)}
shards = (("m-1.safetensors", s1), ("m-2.safetensors", s2), ("m-3.safetensors", s3))
for n, d in shards:
    save_file(d, f"{src}/{n}", metadata={"format": "pt"})
json.dump({"metadata": {}, "weight_map": {k: n for n, d in shards for k in d}}, open(f"{src}/model.safetensors.index.json", "w"))
open(f"{src}/config.json", "w").write("{}")

ok = ko = 0
def check(cond, msg):
    global ok, ko
    print(("OK    " if cond else "FAIL  ") + msg); ok += bool(cond); ko += (not cond)

shutil.copytree(src, "/w/a")
r = subprocess.run([sys.executable, "/p/fp8_convert.py", "/w/a"], capture_output=True, text=True)
check(r.returncode == 0, "fp8_convert.py (reference) runs")
r = subprocess.run([sys.executable, "/p/convertir_denses_fp8.py", "convertir", src, "/w/dst"], capture_output=True, text=True)
check(r.returncode == 0, "streaming converter runs" + ("" if r.returncode == 0 else ": " + r.stderr[-400:]))
ia = json.load(open("/w/a/model.safetensors.index.json"))["weight_map"]
ib = json.load(open("/w/dst/model.safetensors.index.json"))["weight_map"]
check(ia == ib, f"index equal to the reference ({len(ib)} keys)")
as_bytes = lambda t: t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
for f in sorted(set(ib.values())):
    A, B = load_file(f"/w/a/{f}"), load_file(f"/w/dst/{f}")
    check(A.keys() == B.keys() and all(A[k].dtype == B[k].dtype and torch.equal(as_bytes(A[k]), as_bytes(B[k])) for k in A),
          f"{f}: tensors identical to the reference, bit for bit")
check(os.stat("/w/dst/m-3.safetensors").st_ino == os.stat(f"{src}/m-3.safetensors").st_ino, "shard without target = hard link")
check(os.path.exists("/w/dst/.prepared") and not os.path.exists("/w/dst.partial"), ".prepared written, no .partial left")
r = subprocess.run([sys.executable, "/p/convertir_denses_fp8.py", "verifier", src, "/w/dst"], capture_output=True, text=True)
check(r.returncode == 0 and "VERIFICATION VERTE" in r.stdout, "verifier green on a sound output")
p = "/w/dst/m-1.safetensors"
n = struct.unpack("<Q", open(p, "rb").read(8))[0]
h = json.loads(open(p, "rb").read(8 + n)[8:])
off = h[f"{L}.0.linear_attn.in_proj_b.weight"]["data_offsets"][0]
with open(p, "r+b") as fh:
    fh.seek(8 + n + off); b = fh.read(1); fh.seek(8 + n + off); fh.write(bytes([b[0] ^ 0xFF]))
r = subprocess.run([sys.executable, "/p/convertir_denses_fp8.py", "verifier", src, "/w/dst"], capture_output=True, text=True)
check(r.returncode != 0 and "tenseur non vise modifie" in r.stdout, "verifier red on one flipped byte")
r = subprocess.run([sys.executable, "/p/convertir_denses_fp8.py", "convertir", src, "/w/dst"], capture_output=True, text=True)
check(r.returncode != 0, "convertir refuses an existing output")
print(f"\n{ok} OK, {ko} FAIL"); sys.exit(1 if ko else 0)
PY
exit "${PIPESTATUS[0]}"
