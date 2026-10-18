# Provenance: dense layers in block FP8

The NVIDIA checkpoint keeps its dense layers (attention projections, shared experts, linear-attention
projections) in BF16. This patch serves them in FP8 e4m3 with 128x128 block scales: the 300 converted tensors weigh
5.4 GiB in BF16 and 2.7 GiB in FP8, so about 2.7 GiB fewer bytes are read per decoding step. The experts stay NVFP4, the target `lm_head` stays BF16.

## Source

| item | value |
|---|---|
| repository | https://github.com/blazux/qwen3.8-Flash-DGX |
| commit | `7d0798e8c3aaacee8c72afb302aa4247fe4277a7` (2026-10-09) |
| licence | Apache-2.0: `LICENSE-blazux` as published (abbreviated, so GitHub shows NOASSERTION); full text with its copyright line in `../../LICENSES/blazux-Apache-2.0.txt` |
| origin of `fp8_convert.py` | its own header: Saren-Arterius/qwen3.8-Flash-DGX-AutoRound, `tools/fp8_convert.py`, by @Saren-Arterius, Apache-2.0 |
| retrieved | 2026-10-12, files only, through the GitHub API |

## Files

| file | origin | role |
|---|---|---|
| `vllm_fp8_hybrid_modelopt.py` | blazux, `src/`, verbatim | under `VLLM_FP8_HYBRID=1`, sends to vLLM's block `Fp8Config` every `LinearBase` whose checkpoint weight is F8_E4M3 with a `weight_scale_inv` |
| `fp8_convert.py` | blazux, `tools/`, verbatim | target regex and `block_quant` (e4m3, fp32 scale, relative error asserted below 0.10). Its `main()` is not called: it loads a whole shard in memory |
| `vllm-overlay/model_executor/layers/quantization/modelopt.py` | `modelopt.py` of `vllm/vllm-openai:v0.32.0` (`serve/image-base.sha`) plus the three lines that blazux's Dockerfile adds (replayed without fuzz on the v0.32.0 file) | imports and applies the shim; without `VLLM_FP8_HYBRID=1`, it does nothing. A modified file of vLLM (Apache-2.0) |
| `vllm-overlay/models/qwen4_exp/nvidia/qsa.py` | `qsa.py` of `vllm/vllm-openai:v0.32.0` plus 13 lines by Aiko Academy | under `VLLM_FP8_HYBRID=1`, keeps the QSA indexer projection apart from `qkv_proj` (see below). A modified file of vLLM (Apache-2.0) |
| `convertir_denses_fp8.py` | Aiko Academy, Apache-2.0 | streaming conversion: one tensor in memory at a time, output header computed first, untouched files hard-linked, index rewritten as a real copy; `verifier` and `etat-source` sub-commands (French messages) |
| `LICENSE-blazux` | blazux | licence, as published |

sha256 of every file: `../../MANIFEST.sha256`.

## What was adapted

- Conversion: streamed instead of `load_file`, so it runs beside a serving model (6 GB container).
- Hook: the patched `modelopt.py` is mounted read-only on the image instead of rebuilding the image.
- Layout: a flat directory with hard links instead of the symbolic links of the Hugging Face cache.
- blazux's hook on `qsa.py` is not used: for `modelopt_mixed`, `without_modelopt_fp4` returns the configuration
  unchanged and the shim in `modelopt.py` already sees it.

## Review (2026-10-12)

- No network, no `subprocess`, no `eval` or `exec`, no write outside the output directory, in either third-party
  file. The shim only reads safetensors headers of the model directory.
- Hook points present in v0.31.0, and again in v0.32.0 (2026-10-18): `ModelOptMixedPrecisionConfig`, `apply_vllm_mapper`, `get_quant_method`,
  `get_safetensors_params_metadata`, `WeightsMapper.apply_list`, `Fp8Config(weight_block_size=...)`. Fused modules
  `in_proj_qkvz`, `qkv_proj`, `gate_up_proj`: all their members are converted.
- The 1 536 `weight_scale_inv` already present (FP8 experts of the MTP drafter) belong to `RoutedExperts`; the shim
  only intercepts `LinearBase`, so the drafter keeps its path.

## vLLM v0.32.0: the indexer projection

v0.32.0 merges the QSA indexer projection (`indexer.index_qk_proj`) with q, k and v in one GEMM (vllm-project/vllm#59533)
and lists it among the members of `qkv_proj`. The converted checkpoint holds q, k and v in block FP8 and the indexer
projection in BF16: the shim, which sends a fused module to FP8 only when all its members are FP8 in the checkpoint,
would have left the merged module in BF16 over FP8 weights. The `qsa.py` overlay keeps the indexer projection apart
under `VLLM_FP8_HYBRID=1` (the path vLLM already takes with LoRA) and removes it from the members of `qkv_proj` for
that configuration. Read back at start-up on 2026-10-18: 192 modules sent to FP8, as on v0.31.0.

## Witness

`tests/convert-dense-fp8-synthetic.sh`: on a synthetic checkpoint, the streaming converter gives the same tensors,
bit for bit, as `fp8_convert.py` applied in place; `verifier` is green on a sound output and red when one byte of an
untouched tensor is flipped.
