#!/usr/bin/env bash
# prepare.sh: everything serve.sh needs, built on the Spark itself. Each step is idempotent and can run while
# another model is being served (CPU containers, bounded memory, no network except where noted).
#
#   serve/prepare.sh image          pull the image by digest, check the 19 image files the overlays replace
#   serve/prepare.sh download       checkpoint nvidia/Qwen3.8-Flash-Next-NVFP4 at a pinned revision (network,
#                                   about 124 GB, resumable), then check-download
#   serve/prepare.sh check-download every file of the checkpoint against serve/checkpoint.sha (sha256 for LFS files,
#                                   git blob sha1 for the small ones); reads 124 GB, which evicts the page cache
#   serve/prepare.sh convert        dense BF16 layers to block FP8 (patches/fp8-hybrid), then verify; about 4 min
#                                   and 70 GB written, the other files hard-linked
#   serve/prepare.sh kernel         compile the deterministic top-k kernel to $BUILD/_C_det.so (no network)
#   serve/prepare.sh draft-vocab    fetch the frozen public corpus (network), build the 65 536-id draft
#                                   vocabulary from it (no network), write $BUILD/draft_vocab_public_65536.npy
#   serve/prepare.sh all            the six steps in order
# Same configuration as serve.sh (SERVE_CONFIG or environment; serve/config.example.env).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${SERVE_CONFIG:-}" ]; then
  # shellcheck disable=SC1090
  . "$SERVE_CONFIG"
fi
IMAGE="${IMAGE:-vllm/vllm-openai:v0.32.0}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34}"
HF_REPO="nvidia/Qwen3.8-Flash-Next-NVFP4"
HF_REVISION="fc694b54fb0174e0913e6adf86691ef85a4ead47"
SOURCE_DIR="${SOURCE_DIR:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia}"
MODEL_DIR="${MODEL_DIR:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia-fp8hybrid}"
VERIFIED_MARKER="${VERIFIED_MARKER:-.verified}"
PATCHES="${PATCHES:-$REPO/patches}"
BUILD="${BUILD:-$REPO/build}"
NVME_DEVICE="${NVME_DEVICE:-/dev/nvme0n1}"
P="$PATCHES/fp8-hybrid"
# sha256 of the draft vocabulary built from the frozen public corpus (rebuilt to the byte on 2026-10-18)
DRAFT_VOCAB_SHA="19a4415d6146a0fc91355674a11d9929d0e726da56656096a9a52c3d168368a8"
V="$PATCHES/draft-vocab"

say() { echo "== $*"; }
fail() { echo "prepare.sh: $*" >&2; exit 3; }
[ "$(dirname "$SOURCE_DIR")" = "$(dirname "$MODEL_DIR")" ] \
  || fail "SOURCE_DIR and MODEL_DIR must share a parent directory (the conversion hard-links into it)"

step_image() {
  say "image $IMAGE@$IMAGE_DIGEST"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "${IMAGE%:*}@$IMAGE_DIGEST"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker tag "${IMAGE%:*}@$IMAGE_DIGEST" "$IMAGE"
  docker image inspect "$IMAGE" --format '{{join .RepoDigests " "}}' | grep -q "@$IMAGE_DIGEST" \
    || fail "$IMAGE is not $IMAGE_DIGEST"
  docker run --rm -i --network none --entrypoint bash "$IMAGE" \
    -c 'cd /usr/local/lib/python3.12/dist-packages && sha256sum -c --quiet' < "$REPO/serve/image-base.sha" \
    || fail "an image file the overlays replace has changed: the overlays are only valid for $IMAGE_DIGEST"
  echo "   digest and the $(wc -l < "$REPO/serve/image-base.sha" | tr -d ' ') base files of the overlays as expected"
}

step_download() {
  say "checkpoint $HF_REPO@$HF_REVISION to $SOURCE_DIR (about 124 GB)"
  mkdir -p "$SOURCE_DIR"
  docker run --rm --entrypoint python3 -e HF_REPO="$HF_REPO" -e HF_REVISION="$HF_REVISION" \
    -v "$SOURCE_DIR":/dst "$IMAGE" -c \
    'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["HF_REPO"], revision=os.environ["HF_REVISION"], local_dir="/dst")'
  step_check_download
}

step_check_download() {
  say "check of $SOURCE_DIR against serve/checkpoint.sha"
  python3 -I - "$SOURCE_DIR" "$REPO/serve/checkpoint.sha" <<'PY'
import hashlib, os, sys
root, listing = sys.argv[1], sys.argv[2]
ko = 0
for line in open(listing):
    want, name = line.split()
    p = os.path.join(root, name)
    if not os.path.exists(p):
        print(f"   ABSENT {name}"); ko += 1; continue
    if len(want) == 64:
        h = hashlib.sha256()
    else:  # a small file: its git blob id
        h = hashlib.sha1(); h.update(b"blob %d\0" % os.path.getsize(p))
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        print(f"   DIFFERS {name}"); ko += 1
n = sum(1 for _ in open(listing))
print(f"   {n - ko} / {n} files as published at the pinned revision")
sys.exit(1 if ko else 0)
PY
}

conversion_container() {  # $1 = bash command, played in a bounded CPU container
  docker rm -f fp8-convert >/dev/null 2>&1 || true
  docker run --rm --name fp8-convert --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONUNBUFFERED=1 \
    --memory 6g --cpus 4 --device-read-bps "$NVME_DEVICE:400mb" --device-write-bps "$NVME_DEVICE:400mb" \
    -v "$(dirname "$SOURCE_DIR")":"$(dirname "$SOURCE_DIR")" -v "$P":/p:ro \
    --entrypoint bash "$IMAGE" -c "$1"
}

step_convert() {
  if [ -f "$MODEL_DIR/.prepared" ] && [ -f "$MODEL_DIR/$VERIFIED_MARKER" ]; then
    say "converted checkpoint already prepared and verified: $MODEL_DIR"; return
  fi
  local verify="python3 /p/convertir_denses_fp8.py verifier $SOURCE_DIR $MODEL_DIR && date -u +%FT%TZ > $MODEL_DIR/$VERIFIED_MARKER"
  if [ ! -f "$MODEL_DIR/.prepared" ]; then
    [ -e "$MODEL_DIR" ] && fail "$MODEL_DIR exists but is not prepared: remove it by hand"
    say "dense layers to block FP8: $MODEL_DIR (CPU, 6 GB, disk capped at 400 MB/s)"
    conversion_container "set -e
      python3 /p/convertir_denses_fp8.py etat-source $SOURCE_DIR > /tmp/before
      python3 /p/convertir_denses_fp8.py convertir $SOURCE_DIR $MODEL_DIR
      python3 /p/convertir_denses_fp8.py etat-source $SOURCE_DIR > /tmp/after
      cmp /tmp/before /tmp/after && echo 'source untouched: size, date, inode and small-file hashes identical'
      $verify"
  else
    say "verification of $MODEL_DIR"
    conversion_container "set -e; $verify"
  fi
}

step_kernel() {
  say "deterministic top-k kernel to $BUILD/_C_det.so (CPU container, no network)"
  mkdir -p "$BUILD"
  docker rm -f cache-det-build >/dev/null 2>&1 || true
  docker run --rm --name cache-det-build --network none -e CUDA_VISIBLE_DEVICES= --memory 6g --cpus 4 \
    -v "$PATCHES/cache-det/kernel-det":/k:ro -v "$BUILD":/out --entrypoint bash "$IMAGE" -c \
    "set -e; cp -r /k /tmp/kd; cd /tmp/kd; DET_BUILD_DIR=/tmp/kd/build DET_ARCH=121a python3 build_det.py 2>&1 | tail -2
     cp /tmp/kd/build/_C_det.so /out/_C_det.so; chown $(id -u):$(id -g) /out/_C_det.so"
  [ -s "$BUILD/_C_det.so" ] || fail "_C_det.so absent after the build"
  sha256sum "$BUILD/_C_det.so" | sed 's/^/   /'
}

step_draft_vocab() {
  local corpus="$BUILD/public-corpus" out="$BUILD/draft_vocab_public_65536.npy"
  say "frozen public corpus to $corpus (network: Wikipedia revisions and repository files at fixed commits)"
  python3 -I "$V/fetch_public_corpus.py" "$V/public-corpus.tsv" "$corpus"
  say "draft vocabulary from the public corpus (no network, tokenizer of the checkpoint)"
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --memory 8g --cpus 4 \
    -v "$SOURCE_DIR":/tok:ro -v "$corpus":/corpus:ro -v "$V":/v:ro -v "$BUILD":/out \
    --entrypoint python3 "$IMAGE" /v/build_draft_vocab_public.py /tok /out/draft_vocab_public_65536.npy \
    --corpus /corpus --complement /v/draft_vocab_65536.npy
  docker run --rm --network none -v "$BUILD":/out --entrypoint chown "$IMAGE" "$(id -u):$(id -g)" /out/draft_vocab_public_65536.npy
  sha256sum "$out" | sed 's/^/   /'
  [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$DRAFT_VOCAB_SHA" ] \
    || fail "draft vocabulary differs from the published one ($DRAFT_VOCAB_SHA): corpus or tokenizer changed"
  echo "   equal to the published draft vocabulary"
}

case "${1:-}" in
  image) step_image ;;
  download) step_download ;;
  check-download) step_check_download ;;
  convert) step_convert ;;
  kernel) step_kernel ;;
  draft-vocab) step_draft_vocab ;;
  all) step_image; step_download; step_convert; step_kernel; step_draft_vocab ;;
  *) echo "usage: $(basename "$0") {image|download|check-download|convert|kernel|draft-vocab|all}" >&2; exit 2 ;;
esac
