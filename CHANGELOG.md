# Changelog

## Since v0.32.0-2 (2026-10-20)

- `serve.sh`: `"disable_eagle_block_drop": true` in `--speculative-config`. On a prefix-cache hit vLLM drops the last
  cached block for the MTP drafter and recomputes it; on this model a block is 1 648 tokens. Measured in A-B-A on
  2026-10-20 with `bench/prefix_cache.py`: the cache serves 36 256 tokens of the 18 517-token prompt instead of
  32 960, the cached passes take 3.21 and 3.13 s instead of 3.48 to 3.73 s, outputs identical; greedy probe texts
  and MTP acceptance identical in the three arms. `verify.sh` checks 28 settings.
- `patches/tool-parser/`: rebuilt on blazux's commit `427907ac` (issue #56). With the first port of patch 19, a call
  to a tool the request did not offer became, on Chat Completions, an empty turn ending in `stop`; it now reaches the
  client unchanged, known aliases still resolve, Responses stays strict. Two of the nine files change
  (`parser/qwen3.py`, `parser/engine/parser_engine.py`); blazux's Chat tests (10) pass on the rebuilt overlay, the
  unoffered-call test failed on the first port.
- `docs/negative-results.md`: MTP 3 on v0.32.0, and the long-prefill threshold.

## v0.32.0-2 (2026-10-19)

- `patches/mrope-draft-floor/`: on v0.32.0 the MTP drafter's rotary cache stops at 262 144 positions while 524 288
  are served (vllm-project/vllm#62383). The overlay floors it on `max_model_len`, after vllm-project/vllm#62409 by
  @wtogami, for rotary embeddings without YaRN only: the pull request as it stands also moves 8 of the 32 YaRN
  frequencies of the target. Measured in A-B-A on 2026-10-19: greedy probe texts identical; MTP acceptance at
  position 1 for a 449 977-token prompt from 0.746 to 0.898 (a reasoning item) and 0.521 to 0.864 (the needle).
- `patches/auth-all-routes/`: the API key on every route but `GET /health` (vllm-project/vllm#58144: on v0.32.0,
  `/invocations` serves inference without the key). `serve.sh` mounts 26 files; `verify.sh` checks 27 settings.
- `bench/deep_acceptance.py`: MTP acceptance by position for prompts sized in real tokens past 262 144; measured on
  the served recipe on 2026-10-19 (README table, `results/deep-acceptance-v0320-private-set-1019`).
- `serve/image-base.sha`: the v0.32.0 files under the two new overlays (20 base files checked by `prepare.sh image`).
- `bench/load_bench.py`: `--temperature`, `--top-p`, `--top-k`, `--presence-penalty`, `--seed`, and the MTP
  acceptance of each batch. Measured with it: Qwen's non-thinking sampling is not faster than the checkpoint's
  `generation_config` for one client, and sampled throughput is 7 to 9 % under the greedy figures
  (`docs/negative-results.md`).

- `bench/toolcall.py`: `--system TEXT` puts a system message before the prompt; a `code_with_tool` case sends the
  code prompt with the Bash tool declared; every draw records the names called outside code against the declared
  tools, and the summary counts the draws with an undeclared call. Seven new self-check fixtures (27 in all).
- Measured on 2026-10-19 with our private draft set (README table, `results/toolcall-*-1019`), to the code prompt
  that gave invented calls: with a system message, 0 of 12 draws call a tool and 12 return code (cut at the
  3 000-token budget); with the Bash tool declared, 12 call Bash and none an undeclared tool; with reasoning low,
  0 of 12 call a tool and 12 return code. The same tests on v0.31.0 are not measured.
- `bench/probe.py`: `--system TEXT` puts a system message before each generation prompt, so that the code line has a
  value on v0.32.0; not yet measured.

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
