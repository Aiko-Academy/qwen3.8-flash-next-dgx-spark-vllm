#!/usr/bin/env bash
# verify.sh: read back, on the running container, that it is this recipe. Read-only; run it on the Spark.
#
#   serve/verify.sh
#
# 1. Equivalence: the container's arguments (exact list and order), environment, mounts (source, target,
#    read-only), CPU set, IPC, shared memory, ulimits, restart policy, GPUs and published port equal what
#    `serve.sh --print` would run with the same configuration. Environment that comes from the image or from the
#    key file is set aside; a value is never printed for a name that looks secret.
# 2. Content: each mounted overlay file has the sha256 of the file this repository publishes for that target.
# 3. The 27 settings of the recipe, one line each.
# Exit 0 only if every line is OK. Same configuration as serve.sh (SERVE_CONFIG or environment).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${SERVE_CONFIG:-}" ]; then
  # shellcheck disable=SC1090
  . "$SERVE_CONFIG"
fi
CONTAINER="${CONTAINER:-vllm-flashnext}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.32.0}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

"$REPO/serve/serve.sh" --print > "$TMP/expected"
env -u SERVE_CONFIG -u PATCHES -u BUILD -u PLE_OVERLAY -u FP8_HYBRID -u CACHE_DET -u DET_LIB -u DRAFT_HEAD \
  -u DRAFT_VOCAB -u TOOL_PARSER -u MROPE_FLOOR -u AUTH_ALL CONTAINER="$CONTAINER" "$REPO/serve/serve.sh" --print > "$TMP/published"
docker inspect "$CONTAINER" > "$TMP/inspect.json" || { echo "FAIL  container $CONTAINER absent"; exit 1; }
docker image inspect "$IMAGE" --format '{{json .Config.Env}}' > "$TMP/image-env.json"
KEYS=""
[ -n "${API_KEY_FILE:-}" ] && KEYS="$(sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$API_KEY_FILE" | tr '\n' ' ')"

python3 -I - "$TMP" "$KEYS" "${MODEL_DIR:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia-fp8hybrid}" \
  "${VERIFIED_MARKER:-.verified}" "${IMAGE_DIGEST:-sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34}" <<'PY'
import hashlib, json, os, re, subprocess, sys

tmp, keyfile_names, model_dir, marker, digest = sys.argv[1], sys.argv[2].split(), sys.argv[3], sys.argv[4], sys.argv[5]
SECRET = re.compile(r"SECRET|KEY|PASSWORD|TOKEN|URL", re.I)
rows = []

def line(name, ok, detail=""):
    rows.append(ok)
    print(("OK    " if ok else "FAIL  ") + name + (f" ({detail})" if detail and not ok else ""))

def parse(path):
    """Options of a docker run printed one argument per line by serve.sh --print."""
    a = open(path).read().split("\n")[:-1]
    assert a[:3] == ["docker", "run", "-d"], a[:3]
    o = {"env": [], "mounts": [], "ulimit": [], "envfile": None}
    i = 3
    while i < len(a):
        t = a[i]
        if t == "--entrypoint":
            o["entrypoint"], o["image"], o["cmd"] = a[i + 1], a[i + 2], a[i + 3:]
            break
        v = a[i + 1] if i + 1 < len(a) else None
        if t == "-e": o["env"].append(v)
        elif t == "-v": o["mounts"].append(v)
        elif t == "--ulimit": o["ulimit"].append(v)
        elif t == "--env-file": o["envfile"] = v
        elif t in ("--name", "--restart", "--gpus", "--ipc", "--shm-size", "--cpuset-cpus", "-p"): o[t] = v
        else: raise SystemExit(f"verify.sh: option not understood in serve.sh --print: {t}")
        i += 2
    return o

def mounts(o):
    out = set()
    for m in o["mounts"]:
        p = m.split(":")
        out.add((p[0], p[1], len(p) > 2 and p[2] == "ro"))
    return out

e, pub = parse(f"{tmp}/expected"), parse(f"{tmp}/published")
d = json.load(open(f"{tmp}/inspect.json"))[0]
image_env = set(json.load(open(f"{tmp}/image-env.json")) or [])
hc = d["HostConfig"]

# 1. Equivalence
line("image " + e["image"], d["Config"]["Image"] == e["image"], d["Config"]["Image"])
line("entrypoint vllm", d["Config"]["Entrypoint"] == [e["entrypoint"]], str(d["Config"]["Entrypoint"]))
cmd = d["Config"]["Cmd"]
if cmd == e["cmd"]:
    line(f"arguments: the same {len(cmd)}, in the same order", True)
else:
    miss = [x for x in e["cmd"] if x not in cmd]; extra = [x for x in cmd if x not in e["cmd"]]
    line("arguments", False, f"missing {miss[:4]}, extra {extra[:4]}, or order differs")
cenv = {x for x in d["Config"]["Env"] if x not in image_env and x.split("=", 1)[0] not in keyfile_names}
eenv = set(e["env"])
mask = lambda s: {x.split("=", 1)[0] + ("=<value>" if SECRET.search(x.split("=", 1)[0]) else "=" + x.split("=", 1)[1]) for x in s}
line(f"environment: the same {len(eenv)} variables", cenv == eenv,
     f"missing {sorted(mask(eenv - cenv))}, extra {sorted(mask(cenv - eenv))}")
cm = {(m["Source"], m["Destination"], not m["RW"]) for m in d["Mounts"]}
em = mounts(e)
line(f"mounts: the same {len(em)} (source, target, read-only)", cm == em,
     f"missing {sorted(em - cm)[:3]}, extra {sorted(cm - em)[:3]}")
line("CPU set " + e["--cpuset-cpus"], hc.get("CpusetCpus") == e["--cpuset-cpus"], hc.get("CpusetCpus"))
line("IPC host, shared memory 32 GiB", hc.get("IpcMode") == "host" and hc.get("ShmSize") == 32 * 2**30)
ul = {u["Name"]: (u["Soft"], u["Hard"]) for u in hc.get("Ulimits") or []}
line("ulimits memlock unlimited, stack 64 MiB", ul == {"memlock": (-1, -1), "stack": (67108864, 67108864)}, str(ul))
line("restart unless-stopped", (hc.get("RestartPolicy") or {}).get("Name") == "unless-stopped")
gpus = hc.get("DeviceRequests") or []
line("all GPUs", len(gpus) == 1 and gpus[0].get("Count") == -1 and ["gpu"] in gpus[0].get("Capabilities", []))
bind, port, inner = e["-p"].split(":")
pb = (hc.get("PortBindings") or {}).get(f"{inner}/tcp") or [{}]
line(f"port {port} published on {bind}", pb[0].get("HostIp") == bind and pb[0].get("HostPort") == port,
     f"{pb[0].get('HostIp')}:{pb[0].get('HostPort')}")
line("key file given" if e["envfile"] else "no key file (bound to 127.0.0.1 only)",
     bool(e["envfile"]) or bind == "127.0.0.1")

# 2. Content of the mounted overlays against the files this repository publishes for the same target
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
published = {t: s for s, t, _ in mounts(pub)}
skip = ("/models/flashnext", "/root/.cache", "/opt/kernel-det/_C_det.so")
same, diff = 0, []
for s, t, _ in cm:
    if t in skip or t.endswith(".npy"):  # checkpoint, cache, locally built kernel, draft vocabulary (reported below)
        continue
    p = published.get(t)
    if p and os.path.exists(p) and os.path.exists(s) and sha(p) == sha(s):
        same += 1
    else:
        diff.append(t)
line(f"overlay files equal to the published ones: {same}", not diff, f"differ: {diff[:3]}")
vocab = [x.split("=", 1)[1] for x in e["env"] if x.startswith("VLLM_MTP_DRAFT_VOCAB=")]
vsrc = {t: s for s, t, _ in cm}.get(vocab[0]) if vocab else None
if vsrc and os.path.exists(vsrc):
    print(f"INFO  draft vocabulary mounted: {os.path.basename(vsrc)}, sha256 {sha(vsrc)[:16]}")

# 3. The 27 settings
args = cmd
val = lambda f: args[args.index(f) + 1] if f in args else None
spec = json.loads(val("--speculative-config") or "{}")
yarn = json.loads(val("--hf-overrides") or "{}").get("text_config", {}).get("rope_parameters", {})
env = set(d["Config"]["Env"])
src = {m["Destination"]: m["Source"] for m in d["Mounts"]}
ends = lambda suffix: any(t.endswith(suffix) for t in src)
digests = subprocess.run(["docker", "image", "inspect", d["Image"], "--format", "{{json .RepoDigests}}"],
                         capture_output=True, text=True).stdout
parser_files = ["parser/qwen3.py", "parser/abstract_parser.py", "parser/nemotron_v3.py", "parser/engine/adapters.py",
                "parser/engine/parser_engine.py", "parser/engine/parser_engine_config.py",
                "parser/engine/streaming_parser_engine.py", "responses/protocol.py", "responses/serving.py"]
settings = [
    ("context 524288", val("--max-model-len") == "524288"),
    ("YaRN factor 2.0 over 262144", yarn.get("rope_type") == "yarn" and yarn.get("factor") == 2.0
     and yarn.get("original_max_position_embeddings") == 262144),
    ("MTP 2, max_model_len in the speculative config", spec.get("method") == "mtp"
     and spec.get("num_speculative_tokens") == 2 and spec.get("max_model_len") == 524288),
    ("GPU memory 0.80", val("--gpu-memory-utilization") == "0.80"),
    ("KV cache fp8", val("--kv-cache-dtype") == "fp8"),
    ("image digest " + digest[:19], digest in digests),
    ("16 sequences, 4096 tokens per batch", val("--max-num-seqs") == "16" and val("--max-num-batched-tokens") == "4096"),
    ("pinned on 5-9,15-19 (Cortex-X925)", hc.get("CpusetCpus") == "5-9,15-19"),
    ("recurrent state in bfloat16", val("--mamba-ssm-cache-dtype") == "bfloat16"),
    ("quantization modelopt, TP 1", val("--quantization") == "modelopt" and val("--tensor-parallel-size") == "1"),
    ("PLE checkpoint_mapped", json.loads(val("--engram-config") or "{}").get("checkpoint_mapped") is True),
    ("PLE prefetch with MADV_WILLNEED (ple_pageable.py mounted)", ends("qwen4_exp/nvidia/ple_pageable.py")),
    ("tool parser qwen3_xml, reasoning parser qwen3", val("--tool-call-parser") == "qwen3_xml"
     and val("--reasoning-parser") == "qwen3"),
    ("FlashInfer autotune off", "--no-enable-flashinfer-autotune" in args),
    ("converted FP8 checkpoint mounted, marked prepared and verified", src.get("/models/flashnext") == model_dir
     and os.path.exists(os.path.join(model_dir, ".prepared")) and os.path.exists(os.path.join(model_dir, marker))),
    ("VLLM_USE_DEEP_GEMM=0 and VLLM_FP8_HYBRID=1", "VLLM_USE_DEEP_GEMM=0" in env and "VLLM_FP8_HYBRID=1" in env),
    ("FP8 shim, modelopt.py and qsa.py (indexer projection kept apart) mounted", ends("vllm_fp8_hybrid_modelopt.py")
     and ends("quantization/modelopt.py") and ends("qwen4_exp/nvidia/qsa.py")),
    ("reduced draft vocabulary: VLLM_MTP_DRAFT_VOCAB", any(x.startswith("VLLM_MTP_DRAFT_VOCAB=/") for x in env)),
    ("draft head mtp.py and vocabulary mounted", ends("qwen4_exp/nvidia/mtp.py") and bool(vsrc)),
    ("NVFP4 draft rows: VLLM_MTP_DRAFT_NVFP4=1", "VLLM_MTP_DRAFT_NVFP4=1" in env),
    ("target head in BF16 (no VLLM_FP8_HEAD, no model.py mounted)", not any(x.startswith("VLLM_FP8_HEAD") for x in env)
     and not ends("qwen4_exp/nvidia/model.py")),
    ("prefix cache repaired (scheduler.py; the worker half is in vLLM v0.32.0)", ends("core/sched/scheduler.py")
     and not ends("model_states/mamba_hybrid.py")),
    ("deterministic QSA top-k (qsa_indexer.py, _C_det.so, VLLM_QSA_DET_TOPK=1)", "VLLM_QSA_DET_TOPK=1" in env
     and ends("ops/qsa_indexer.py") and "/opt/kernel-det/_C_det.so" in src),
    ("tool parser repaired: nine files mounted", sum(any(t.endswith(p) for t in src) for p in parser_files) == 9),
    ("MTP-layer KV in BF16 (--kv-cache-dtype-skip-layers 48, honoured by vLLM v0.32.0)",
     val("--kv-cache-dtype-skip-layers") == "48"),
    ("drafter M-RoPE cache floored on the served window (mrope.py mounted)", ends("rotary_embedding/mrope.py")),
    ("API key on every route but GET /health (authenticate.py mounted)", ends("serve/middleware/authenticate.py")),
]
assert len(settings) == 27
for n, (name, ok) in enumerate(settings, 1):
    line(f"setting {n:2d}: {name}", ok)
print(f"\n{sum(rows)} OK, {len(rows) - sum(rows)} FAIL")
sys.exit(0 if all(rows) else 1)
PY
