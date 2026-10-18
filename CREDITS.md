# Credits

This recipe is mostly other people's work, assembled, measured and kept in one place. Every third-party file of
this repository is listed below with its author, its repository, the commit or pull request it comes from and its
licence; `tools/check_credits.py` checks that the list is complete. Each patch directory has a `PROVENANCE.md`
that says what was taken and what was changed.

## Model and engine

| what | who | licence |
|---|---|---|
| Qwen3.8-Flash-Next, the model | Qwen team | Qwen Community License 1.0 (`qwen-community-1.0` on the model card of `Qwen/Qwen3.8-Flash-Next`) |
| checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` (downloaded by `serve/prepare.sh`, never redistributed here) | NVIDIA | NVIDIA Open Model License (`nvidia-open-model-license` on its model card) |
| vLLM v0.32.0, image `vllm/vllm-openai:v0.32.0` (digest `sha256:befccf0e...`, tag commit `5df63a72`), and every vLLM file modified in `patches/*/vllm-overlay/` | the vLLM project | Apache-2.0 |

## Data

| what | where it comes from | licence |
|---|---|---|
| public corpus of the draft vocabulary (`patches/draft-vocab/public-corpus.tsv`): only revision ids, commits and hashes are published; the `.npy` holds token ids | 213 Wikipedia featured articles at fixed revisions (French and English); vllm-project/vllm v0.31.0, microsoft/TypeScript v5.9.3, tokio-rs/tokio 1.47.1 at fixed commits | Wikipedia text CC BY-SA 4.0, not redistributed; vLLM and TypeScript Apache-2.0; tokio MIT |
| French text of the prefill and prefix-cache prompts (`bench/prompts/long-fr-verne.txt`) | Jules Verne, 1873, Project Gutenberg eBook #800 | public domain |

## Code used

| contribution | author | where it comes from | licence |
|---|---|---|---|
| dense layers in block FP8 (shim, conversion formula), repaired prefix cache, reduced draft vocabulary hook and builder, and the public draft set used as complement | blazux | blazux/qwen3.8-Flash-DGX, commit `7d0798e` (2026-10-09) | Apache-2.0 (abbreviated `LICENSE`, see below) |
| repaired Qwen tool parser: patches 12, 13 and 19 | blazux; patch 19 from fixes and fixtures by Chaos2522 (blazux issue #31) | blazux/qwen3.8-Flash-DGX, commit `a9ec989` (2026-10-16) | Apache-2.0 |
| original `fp8_convert.py` | Saren Arterius (@Saren-Arterius) | Saren-Arterius/qwen3.8-Flash-DGX-AutoRound, `tools/fp8_convert.py`, as vendored by blazux | Apache-2.0 per its header |
| checkpoint-mapped PLE storage, the change that makes the model fit one Spark | Jürgen Schmied (@jschmied) | vllm-project/vllm#58439 (open), PR head `047fc1ce` merged onto v0.32.0 | Apache-2.0 |
| PLE prefetch with `MADV_WILLNEED` | Mike Waychison (@waych) | commit `b2cc913` in jschmied/vllm#4 | Apache-2.0 |
| deterministic persistent top-k kernel (it carries its own attribution comments to sglang and flashinfer) | Jürgen Schmied (@jschmied) | jschmied/qwen38-flash-next-gb10, commit `e0ef69d`; proposed as vllm-project/vllm#55122 (open) | Apache-2.0 |
| `--kv-cache-dtype-skip-layers` honoured in the QSA attention (an overlay on v0.31.0; part of vLLM since v0.32.0) | @stecasta | vllm-project/vllm#60023, merged 2026-10-05, commit `4ff028d7` | Apache-2.0 |
| NVFP4 quantization of the draft head rows (`_quantize_nvfp4`) | @stecasta | vllm-project/vllm#59973, as carried by salient-data/qwen38-next-dgx-recipe@faf5cb30 (patch 0003) | Apache-2.0 |

## Settings taken from others

| setting | from |
|---|---|
| vLLM pinned on the Cortex-X925 cores (`--cpuset-cpus 5-9,15-19`), 16 sequences, recurrent state in BF16 (`--mamba-ssm-cache-dtype bfloat16`) | @aphex3k, blazux/qwen3.8-Flash-DGX#54 ("Tweak CPUSET and MAMBA_SSM_DTYPE"), merged 2026-10-14 |
| KV cache of the MTP layer in BF16 (`--kv-cache-dtype-skip-layers 48`) as a lever worth trying | salient-data/qwen38-next-dgx-recipe@faf5cb30 and its NVIDIA forum thread (385463) |

## Inspiration and comparison

Ideas and figures that we read, compared against or tested; no code from these sources is in this repository.

| source | what it gave us |
|---|---|
| salient-data/qwen38-next-dgx-recipe (@jeremy-newhouse), commit `faf5cb30` | the FP8 output head and NVFP4 draft head ideas, the warm-up note on Triton kernels of the rejection sampler (`bench/warmup.py`) |
| vllm-project/vllm#59753 and #59740 (@stecasta), measured on local-inference-lab/Qwen3.8-Flash-Next-NVFP4 | SM121 skinny-GEMM plans and the sparse draft head: studied, not used (`docs/negative-results.md`); the public reference figure of #59740 does not serve the NVIDIA checkpoint |
| MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark | published throughput figures we compare with, at a different protocol |
| tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark (Tony DeAngelo) | an earlier reduced draft vocabulary (`single-spark-vllm-tp1/patch/mtp_draft_vocab.py`, first committed on 2026-09-05, commit `63fa63b`); the hook used here is blazux's. Throughput figures measured with llama-benchy (issue #3), at a different protocol |
| eugr/spark-vllm-docker and the image `eugr/spark-vllm-b12x` | the recipe of the NVIDIA FP4-Local checkpoint, tested and not kept (`docs/negative-results.md`) |
| christopherowen/spark-ds41f and NVIDIA forum thread 384438 (post 29) | the NVMe APST experiment, replayed in A-B-A and not kept (`docs/negative-results.md`) |
| NVIDIA developer forum, DGX Spark category: threads 385463, 385994, 386117, 386124 | measurements, settings and warnings shared by the community |
| eugr/llama-benchy (Eugene Rakhmatulin, MIT), version 0.4.0 | the community benchmark we also run, installed in a throwaway container at measurement time (`results/`) |

## Open question for blazux

The `LICENSE` of blazux/qwen3.8-Flash-DGX is an abbreviated Apache-2.0 header ("Copyright 2026 blazux", then a
pointer to the full text), so GitHub shows NOASSERTION. We ship it as published (`LICENSE-blazux` in each patch
directory) and add the full Apache-2.0 text with that copyright line in `LICENSES/blazux-Apache-2.0.txt`. Adding the
full text upstream would remove the ambiguity.

## Third-party files

One row per file. `modified` says whether the file differs from its source, and by whom.

| file | author | repository | commit or PR | licence | modified |
|---|---|---|---|---|---|
| `LICENSES/blazux-Apache-2.0.txt` | Apache Software Foundation; copyright line of blazux | apache.org, blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | yes, by Aiko Academy: copyright line filled in |
| `bench/prompts/long-fr-verne.txt` | Jules Verne (1873) | Project Gutenberg eBook #800 | release of 2013-10-05 | public domain | yes, by Aiko Academy: Project Gutenberg header and licence removed, cut after 80 519 characters |
| `patches/fp8-hybrid/LICENSE-blazux` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/fp8-hybrid/fp8_convert.py` | Saren Arterius, vendored by blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/fp8-hybrid/vllm_fp8_hybrid_modelopt.py` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/fp8-hybrid/vllm-overlay/model_executor/layers/quantization/modelopt.py` | the vLLM project; hook by blazux | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `7d0798e` | Apache-2.0 | yes, by blazux's hook (three lines) |
| `patches/fp8-hybrid/vllm-overlay/models/qwen4_exp/nvidia/qsa.py` | the vLLM project | vllm-project/vllm | v0.32.0 | Apache-2.0 | yes, by Aiko Academy: indexer projection kept apart under `VLLM_FP8_HYBRID=1` |
| `patches/ple-checkpoint-mapped/pr-58439.diff` | Jürgen Schmied | vllm-project/vllm | #58439 | Apache-2.0 | no |
| `patches/ple-checkpoint-mapped/willneed-jschmied-b2cc913.diff` | Mike Waychison | jschmied/vllm | #4, `b2cc913` | Apache-2.0 | no |
| `patches/ple-checkpoint-mapped/vllm-overlay/vllm/config/engram.py` | the vLLM project; Jürgen Schmied | vllm-project/vllm | #58439 at `047fc1ce` | Apache-2.0 | no (the three-way merge gives the PR file) |
| `patches/ple-checkpoint-mapped/vllm-overlay/vllm/models/qwen4_exp/common/ngram_embedding.py` | the vLLM project; Jürgen Schmied | vllm-project/vllm | #58439 at `047fc1ce` | Apache-2.0 | no (the three-way merge gives the PR file) |
| `patches/ple-checkpoint-mapped/vllm-overlay/vllm/models/qwen4_exp/nvidia/model_state.py` | the vLLM project; Jürgen Schmied | vllm-project/vllm | #58439 at `047fc1ce` | Apache-2.0 | no (the three-way merge gives the PR file) |
| `patches/ple-checkpoint-mapped/vllm-overlay/vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | the vLLM project; Jürgen Schmied | vllm-project/vllm | #58439 at `047fc1ce` | Apache-2.0 | no (the three-way merge gives the PR file) |
| `patches/ple-checkpoint-mapped/vllm-overlay/vllm/models/qwen4_exp/nvidia/ple_pageable.py` | Jürgen Schmied; Mike Waychison | vllm-project/vllm, jschmied/vllm | #58439 at `047fc1ce`, #4 `b2cc913` | Apache-2.0 | yes, by Aiko Academy: `b2cc913` applied |
| `patches/cache-det/LICENSE-blazux` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/cache-det/LICENSE-jschmied` | Jürgen Schmied | jschmied/qwen38-flash-next-gb10 | `e0ef69d` | Apache-2.0 | no |
| `patches/cache-det/kernel-det/bindings_det.cpp` | Jürgen Schmied | jschmied/qwen38-flash-next-gb10 | `e0ef69d` | Apache-2.0 | no |
| `patches/cache-det/kernel-det/build_det.py` | Jürgen Schmied | jschmied/qwen38-flash-next-gb10 | `e0ef69d` | Apache-2.0 | no |
| `patches/cache-det/kernel-det/persistent_topk.cuh` | Jürgen Schmied (from vLLM, sglang and flashinfer code, per its comments) | jschmied/qwen38-flash-next-gb10 | `e0ef69d` | Apache-2.0 | no |
| `patches/cache-det/kernel-det/topk_det.cu` | Jürgen Schmied | jschmied/qwen38-flash-next-gb10 | `e0ef69d` | Apache-2.0 | no |
| `patches/cache-det/kernel-det/torch_utils.h` | Jürgen Schmied | jschmied/qwen38-flash-next-gb10 | `e0ef69d` | Apache-2.0 | no |
| `patches/cache-det/patch_mamba_block_size.py` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/cache-det/patch_qsadet.py` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/cache-det/vllm-overlay/models/qwen4_exp/nvidia/ops/qsa_indexer.py` | the vLLM project; hook by blazux | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `7d0798e` | Apache-2.0 | yes, by blazux's `patch_qsadet.py` |
| `patches/cache-det/vllm-overlay/v1/core/sched/scheduler.py` | the vLLM project; hook by blazux | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `7d0798e` | Apache-2.0 | yes, by blazux's `patch_mamba_block_size.py` |
| `patches/draft-vocab/LICENSE-blazux` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/draft-vocab/build_draft_vocab.py` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/draft-vocab/draft_vocab_65536.npy` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/draft-vocab/patch_mtp_draft_vocab.py` | blazux | blazux/qwen3.8-Flash-DGX | `7d0798e` | Apache-2.0 | no |
| `patches/draft-vocab/vllm-overlay/models/qwen4_exp/nvidia/mtp.py` | the vLLM project; hook by blazux | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `7d0798e` | Apache-2.0 | yes, by blazux's `patch_mtp_draft_vocab.py` |
| `patches/draft-nvfp4-head/vllm-overlay/models/qwen4_exp/nvidia/mtp.py` | the vLLM project; hook by blazux; NVFP4 code by @stecasta | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `7d0798e`, #59973 | Apache-2.0 | yes, by Aiko Academy: NVFP4 draft rows and FP8-head path |
| `patches/tool-parser/LICENSE-blazux` | blazux | blazux/qwen3.8-Flash-DGX | `a9ec989` | Apache-2.0 | no |
| `patches/tool-parser/upstream-patches/qwen-tool-preamble.patch` | blazux | blazux/qwen3.8-Flash-DGX | `a9ec989` | Apache-2.0 | no |
| `patches/tool-parser/upstream-patches/qwen-tool-marker-guard.patch` | blazux | blazux/qwen3.8-Flash-DGX | `a9ec989` | Apache-2.0 | no |
| `patches/tool-parser/upstream-patches/qwen-safe-output.patch` | Chaos2522, ported by blazux | blazux/qwen3.8-Flash-DGX | `a9ec989`, issue #31 | Apache-2.0 | no |
| `patches/tool-parser/vllm-overlay/entrypoints/openai/responses/protocol.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |
| `patches/tool-parser/vllm-overlay/entrypoints/openai/responses/serving.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches; two hunks of patch 19 ported by hand by Aiko Academy |
| `patches/tool-parser/vllm-overlay/parser/abstract_parser.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |
| `patches/tool-parser/vllm-overlay/parser/engine/adapters.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |
| `patches/tool-parser/vllm-overlay/parser/engine/parser_engine.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |
| `patches/tool-parser/vllm-overlay/parser/engine/parser_engine_config.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |
| `patches/tool-parser/vllm-overlay/parser/engine/streaming_parser_engine.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |
| `patches/tool-parser/vllm-overlay/parser/nemotron_v3.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |
| `patches/tool-parser/vllm-overlay/parser/qwen3.py` | the vLLM project; patches by blazux and Chaos2522 | vllm-project/vllm, blazux/qwen3.8-Flash-DGX | v0.32.0, `a9ec989` | Apache-2.0 | yes, by the three patches |

## Our own files

Everything else is written by Aiko Academy and licensed Apache-2.0 (`LICENSE`): the files matched by these patterns.

- `README.md`, `CHANGELOG.md`, `CREDITS.md`, `NOTICE`, `LICENSE`, `MANIFEST.sha256`, `.gitignore`
- `serve/*`, `bench/*.py`, `bench/quality/*`, `bench/community/*`, `bench/prompts/*.json`, `tests/*`, `tools/*`, `docs/*`, `results/*`
- `patches/*/PROVENANCE.md`, `patches/fp8-hybrid/convertir_denses_fp8.py`, `patches/draft-vocab/build_draft_vocab_public.py`,
  `patches/draft-vocab/fetch_public_corpus.py`, `patches/draft-vocab/public-corpus.tsv`, `patches/draft-vocab/temoin_crochet.py`,
  `patches/tool-parser/EMPREINTES.sha256`, `patches/tool-parser/marker-guard-witness/*`
