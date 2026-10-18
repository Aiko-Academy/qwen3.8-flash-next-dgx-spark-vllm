# Changelog

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

Waiting for: vLLM v0.32.0 (then the overlays are to be redone or dropped where upstream has merged them:
vllm-project/vllm#58439, #55122, #60023, #59973).
