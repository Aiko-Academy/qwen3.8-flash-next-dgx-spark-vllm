# Provenance: per-layer embeddings read in place from the checkpoint

What makes the model fit one 128 GB Spark: with `--engram-config '{"checkpoint_mapped": true}'`, the per-layer
embedding (PLE) table (51.2 GB, 129 tensors of `model-fp8-mtp-ple.safetensors`, which also holds the 2.5 GB MTP
layer) is memory-mapped read-only from the checkpoint and gathered
on demand, instead of being copied to host memory. The rows a step needs are first passed to
`madvise(MADV_WILLNEED)`, which queues all their reads at once.

## Sources

| item | value |
|---|---|
| upstream PR | vllm-project/vllm#58439 by Jürgen Schmied (@jschmied), "[Qwen4Exp] Checkpoint-mapped PLE storage for unified-memory GPUs (DGX Spark)", open; files taken at the PR head of 2026-10-07, commit `6af74baa5daa` (author commit of 2026-09-30) |
| prefetch | commit `b2cc913b6e69` by Mike Waychison (@waych), "[Qwen4Exp] Checkpoint-mapped PLE: queue a step's page reads with MADV_WILLNEED", in jschmied/vllm#4 ("PLE prefetch MADV_WILLNEED + layer ahead") |
| licence | Apache-2.0 (vLLM and its fork jschmied/vllm) |

## Files

| file | origin |
|---|---|
| `vllm-overlay/vllm/models/qwen4_exp/nvidia/ple_pageable.py` | PR head `6af74baa5daa`, plus commit `b2cc913` (new file of the PR) |
| `vllm-overlay/vllm/models/qwen4_exp/nvidia/model_state.py` | PR head `6af74baa5daa` |
| `vllm-overlay/vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | PR head `6af74baa5daa` |
| `vllm-overlay/vllm/models/qwen4_exp/common/ngram_embedding.py` | three-way merge of v0.31.0 and the PR, by Aiko Academy, no conflict marker |
| `vllm-overlay/vllm/config/engram.py` | three-way merge of v0.31.0 and the PR, by Aiko Academy, no conflict marker |
| `pr-58439.diff` | the PR diff as retrieved on 2026-10-07 |
| `willneed-jschmied-b2cc913.diff` | commit `b2cc913` as a diff (the file name says where it was found, not who wrote it) |

Checked on 2026-10-18: the three PR-head files equal the files of jschmied/vllm at `6af74baa5daa` byte for byte
(`ple_pageable.py` after reversing `b2cc913`). Image hashes of the files the overlay replaces:
`../../serve/image-base.sha` (`ple_pageable.py` is new and absent from the image). sha256: `../../MANIFEST.sha256`.

## Review (2026-10-07)

`ple_pageable.py`: read-only mmap and a Triton gather; no network, no shell, no `eval`.
