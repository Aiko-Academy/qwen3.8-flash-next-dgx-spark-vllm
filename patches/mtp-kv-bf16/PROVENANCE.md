# Provenance: KV cache of the MTP layer in BF16

v0.31.0 accepts `--kv-cache-dtype-skip-layers`, but the QSA attention of Qwen3.8-Flash-Next reads
`cache_config.cache_dtype` without looking at the option: all its layers, the MTP layer included, keep an fp8 KV.
With this file, the layer listed in the option (index 48, the MTP layer) takes `auto` (BF16) and the others keep
fp8. The target keeps its fp8 KV: only the drafter changes, hence only the acceptance rate. Cost: a smaller KV pool
(one full-attention layer out of thirteen doubled).

## Source

| item | value |
|---|---|
| upstream PR | vllm-project/vllm#60023 by @stecasta, "[Bugfix][Qwen4Exp] Honor --kv-cache-dtype-skip-layers in QSA attention", merged on 2026-10-05, commit `4ff028d77e063e0055849ffec3855c6634179158`, Apache-2.0; not in v0.31.0 |
| base | `vllm/models/qwen4_exp/nvidia/qsa.py` of the image `vllm/vllm-openai:v0.31.0`, sha256 `b05df87a20a31e7f3d3374631967a39540407efb55ad4ed31580a3bd10ce4ddc` |

## File

| file | origin | role |
|---|---|---|
| `vllm-overlay/models/qwen4_exp/nvidia/qsa.py` | image file plus the three pieces of #60023 ported by hand by Aiko Academy (the diff does not apply as is: neighbouring imports differ) | mounted read-only; a modified file of vLLM (Apache-2.0) |

Port: import of `CacheConfig` and `extract_layer_index`, function `qsa_kv_cache_dtype` taken verbatim, and
`self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)` in `Qwen4ExpQSAAttention.__init__`. The type then
flows down without another hook: `kv_cache_torch_dtype`, the implementation and `get_kv_cache_spec` read the
attribute. Inert without the option. sha256: `../../MANIFEST.sha256`.
