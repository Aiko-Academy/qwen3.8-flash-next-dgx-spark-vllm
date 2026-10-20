# Provenance: checkpoint pages touched before each copy to the GPU

On the GB10, an integrated GPU, the shared weight loaders of vLLM v0.32.0 copy each checkpoint tensor to the GPU with
`param.copy_(loaded_weight)` straight from the safetensors mapping. The page faults then happen inside the driver
copy, several times slower than on the CPU (vllm-project/vllm#58726). This overlay is vllm-project/vllm#58868 by
@Willian-Zhang applied to v0.32.0: `copy_weight_()` in `model_executor/utils.py` touches one byte per page of a
contiguous CPU source of the same dtype, on an integrated GPU only, then copies; the 26 copy sites of the shared
loaders call it.

## Sources

| item | value |
|---|---|
| pull request | vllm-project/vllm#58868 (open), head `f66f2108` |
| base files | six files of the v0.32.0 image (sha256 in `serve/image-base.sha`), Apache-2.0, vLLM contributors |
| change | the diff of the pull request; in `routed_experts.py` one context line of `main` is absent from v0.32.0 (an import of `quantization.online.moe_base`), and the eight copy sites are the same |
| gate | `current_platform.is_integrated_gpu()`, that is `torch.cuda.get_device_properties(i).is_integrated`, which v0.32.0 already reads for the GB10 memory |
| not covered | model-specific loaders and copies from GPU to GPU keep a plain `copy_()` |
| licence | Apache-2.0 |

## Checks

| check | result |
|---|---|
| imports in the v0.32.0 image, CPU, no network (2026-10-20, 19:18) | the six modules and the Qwen4Exp model import; 26 copy sites; byte-exact copy |
| A-B-A on our server (2026-10-20, 19:20 to 19:58): restart order to first completion | 254 s with the overlay, 738 s for the default restarted right after |
| main weight load, `Loading weights took` in the container logs | 51.5 s with the overlay, 466.7 s without; drafter 33.8 s against 70.6 s; 73.78 GiB loaded in both |
| greedy probe | the same texts and the same MTP acceptance in the three arms; KV pool within the gap of the two default arms |

The sha256 of the six overlay files are in `MANIFEST.sha256`.
