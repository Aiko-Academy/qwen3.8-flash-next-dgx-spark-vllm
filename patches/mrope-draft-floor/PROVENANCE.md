# Provenance: M-RoPE cache of the MTP drafter floored on the served window

On vLLM v0.32.0, with 524 288 tokens served through YaRN, the MTP drafter of Qwen3.8-Flash-Next builds its rotary
embedding from the checkpoint's `rope_parameters` (`rope_type` default, 262 144 positions): our `--hf-overrides`
do not reach it. Its cos/sin cache then has 262 144 rows, while the positions it is asked for go up to 524 288
(vllm-project/vllm#62383). This overlay floors that cache on the engine's `max_model_len`, for rotary embeddings
WITHOUT YaRN only.

## Sources

| item | value |
|---|---|
| upstream issue | vllm-project/vllm#62383 (2026-10-19), a crash past 262 144 tokens under CUDA graphs with MTP |
| idea and code | vllm-project/vllm#62409 by @wtogami, "[Bugfix] Floor the M-RoPE cache on the served window", open; read at head `2c2c57288bbb` on 2026-10-19 |
| base file | `vllm/model_executor/layers/rotary_embedding/mrope.py` of the v0.32.0 tag (sha256 `922e8ba6`), Apache-2.0, vLLM contributors |
| licence | Apache-2.0 |

## What differs from the pull request

The pull request floors `cache_max_position_num` for every M-RoPE. In `MRotaryEmbedding`, that number is also
passed as `max_position_embeddings`, the base of the YaRN correction range: for our YaRN target, the pull request as
it stands raises it from 262 144 to 524 288, the target cache grows to 1 048 576 rows, and 8 of the 32 rotary
frequencies change, so greedy outputs would change. This file applies the floor only when `scaling_factor` is None:
the drafter gets 524 288 rows, and the target is left as v0.32.0 builds it.

## Checks

| check | result |
|---|---|
| construction in the v0.32.0 image, CPU, no network (2026-10-19, 11:44 to 11:47) | v0.32.0: drafter 262 144 rows, target 524 288. Pull request as it stands: drafter 524 288, target 1 048 576 and 8 frequencies of 32 changed. This file: drafter 524 288 with no frequency changed, target identical to the byte |
| served, A-B-A on 2026-10-19 (12:09 to 13:04), same probe | greedy texts identical in the three arms; no crash and no Xid up to 449 977 tokens with or without this file |
| MTP acceptance at 449 977 tokens, position 1 | 0.746 (a reasoning item) and 0.521 (the needle question) without this file; 0.898 and 0.864 with it; 0.921 and 0.963 at 200 013 tokens either way |

sha256 of the overlay: `96c2cd7301e38b89428ccdb92f22df16742a341a9231becda1b630ae9e7037e6`.
