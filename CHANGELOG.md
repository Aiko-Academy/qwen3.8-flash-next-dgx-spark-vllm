# Changelog

## v0.32.0-1 (2026-10-18)

The recipe moves to vLLM v0.32.0 (image pinned by digest `sha256:befccf0e...`). Same checkpoint, same converted
dense layers, same settings.

- Dropped, because v0.32.0 carries the fix: the MTP-layer KV overlay (vllm-project/vllm#60023) and the worker half of
  the prefix-cache fix (vllm-project/vllm#55601). 24 mounts instead of 25.
- New: `patches/fp8-hybrid/.../qsa.py` keeps the QSA indexer projection apart under `VLLM_FP8_HYBRID=1`; v0.32.0 merges
  it with q, k and v (vllm-project/vllm#59533) and the converted checkpoint holds them in different formats.
- Replayed on the v0.32.0 files: the FP8 shim hook, the scheduler half of the prefix-cache fix, the deterministic
  top-k hook, the draft head, the tool parser (two Responses API hunks ported by hand), the PLE files of
  vllm-project/vllm#58439 (now its head of 2026-10-13).
- `tests/overlay-import.sh`: every overlay imports in the pinned image.

## v0.31.0-1 (2026-10-18)

First version, on vLLM v0.31.0.

- `serve/serve.sh`: the reference service as one docker run, pinned image digest, 25 read-only mounts.
- `serve/prepare.sh`: checkpoint at a pinned revision and its check, dense layers to block FP8, deterministic top-k
  kernel, draft vocabulary from a frozen public corpus.
- `serve/verify.sh`: equivalence with `serve.sh --print`, published overlay files, 25 settings.
- `patches/`: seven overlays with their provenance; `CREDITS.md` lists all 45 third-party files.
- `bench/`: probe, prefill and load bench, prefix cache, tool-call witness, NVMe reads, reasoning and long-context
  bench, code gate and its judge, warm-up, clean-pass wrapper.
- `results/`: measurements of 2026-10-18 (README table).

Tag `v0.31.0-1`.
