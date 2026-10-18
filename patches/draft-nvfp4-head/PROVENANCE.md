# Provenance: NVFP4 rows of the reduced draft head

The served `mtp.py`. It is the reduced-vocabulary hook of `../draft-vocab/` (first 457 lines identical) plus one
step: under `VLLM_MTP_DRAFT_NVFP4=1`, the 65 536 rows of the draft head are quantized to NVFP4 and multiplied on
Marlin W4A16, about 230 MiB fewer bytes per draft pass. The target `lm_head` stays BF16, so the target's outputs do
not change.

## File

| file | base | addition | origin of the idea and of the code |
|---|---|---|---|
| `vllm-overlay/models/qwen4_exp/nvidia/mtp.py` | the hooked `mtp.py` of `../draft-vocab/` (image `mtp.py`, sha256 `35dc7e4b...`, plus blazux's hook) | NVFP4 rows on Marlin (`VLLM_MTP_DRAFT_NVFP4=1`); an FP8-head path (`VLLM_MTP_DRAFT_HEAD_CKPT`) that this recipe does not use | NVFP4 quantization taken from vllm-project/vllm#59973 (@stecasta, Apache-2.0), `_quantize_nvfp4` of `quantization/online/lm_head.py`, as carried by salient-data/qwen38-next-dgx-recipe@faf5cb30 (patch 0003) |

Written by Aiko Academy on 2026-10-12, Apache-2.0, as a modified file of vLLM. The log line it prints at start-up
is in French (`tete cible bfloat16 ; brouillon NVFP4 (Marlin)`). sha256: `../../MANIFEST.sha256`.

## Witness

`tests/draft-head-witnesses.sh`: on CPU, the hook is active, the set's logits equal the BF16 reference and the
others are `-inf`; with `--gpu`, the NVFP4 Marlin product is compared with BF16 (cosine) and timed at M = 1.
