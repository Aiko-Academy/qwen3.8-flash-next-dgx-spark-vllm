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
- Measured on 2026-10-18 (README table, `results/*-v0320-*`), with the public set, v0.31.0-1 in brackets: French
  prose 41.1 tok/s (40.9), English 45.3 (44.7), 8 / 16 clients 133.0 to 133.6 / 194.6 to 195.1 (131.0 to 134.4 /
  194.1 to 197.5); prefix cache green; short reasoning 15 / 15, code gate 4 / 4. With our private set: long context
  at 449 365 tokens, 3 / 3 with the needle.
- Found on v0.32.0: to a code prompt with no tool declared, the model often writes calls to tools that do not exist
  (6 of 12 draws with the public set, 11 of 12 with our private set; 0 of 12 on French prose); the parser drops them
  and the client receives a sentence of intent. The probe's code line has no value on this version.
- With our private draft set, French prose is level with the public set on v0.32.0 (41.2 against 41.1), where it was
  ahead on v0.31.0.

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
