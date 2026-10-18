# Provenance: repaired prefix cache and deterministic QSA top-k

Two defects of vLLM v0.31.0 on this model, both fixed by third-party code mounted on the image:

| defect | cause |
|---|---|
| prefix cache enabled but never hit | `EngineCore` overwrites `cache_config.block_size` with the smallest block of the KV groups (8 tokens with MTP 2); the scheduler and the worker take it for the Mamba block. States almost never fall on a boundary, and a hit would restore a null state. |
| greedy decoding not reproducible on long prompts | `persistent_topk` of the QSA sparse attention is not deterministic on GB10 and can drop candidates (vllm-project/vllm#51782) |

## Sources

| item | value |
|---|---|
| blazux | https://github.com/blazux/qwen3.8-Flash-DGX, commit `7d0798e8c3aaacee8c72afb302aa4247fe4277a7` (2026-10-09), Apache-2.0 (`LICENSE-blazux`, abbreviated; full text in `../../LICENSES/blazux-Apache-2.0.txt`) |
| Jürgen Schmied (jschmied) | https://github.com/jschmied/qwen38-flash-next-gb10, commit `e0ef69d4f5575dad00d34e05479eaf4c6547bace` (2026-09-07), Apache-2.0 (`LICENSE-jschmied`); the kernel is proposed upstream as vllm-project/vllm#55122 (open) |
| retrieved | 2026-10-13, files only; the five kernel files give the hashes that blazux's Dockerfile pins |

## Files

| file | origin | role |
|---|---|---|
| `patch_mamba_block_size.py` | blazux, `src/`, verbatim | two lines: the worker divides by `mamba_block_size`, the scheduler takes its own `block_size` (least common multiple of the groups) |
| `patch_qsadet.py` | blazux, `src/`, verbatim | in `ops/qsa_indexer.py::_topk`, under `VLLM_QSA_DET_TOPK=1`, loads `_C_det.so` and calls its `persistent_topk`; inert without the variable |
| `kernel-det/*` | jschmied, `patches/kernel-det/`, verbatim | deterministic kernel and its build (`torch.utils.cpp_extension.load`, sm_121a). `persistent_topk.cuh` carries its own attribution comments to sglang and flashinfer |
| `vllm-overlay/v1/worker/gpu/model_states/mamba_hybrid.py` | image `vllm/vllm-openai:v0.31.0` (`1ddd7dc9...`) + `patch_mamba_block_size.py` | mounted read-only; a modified file of vLLM (Apache-2.0) |
| `vllm-overlay/v1/core/sched/scheduler.py` | image (`f820bee4...`) + `patch_mamba_block_size.py` | mounted read-only; a modified file of vLLM |
| `vllm-overlay/models/qwen4_exp/nvidia/ops/qsa_indexer.py` | image (`4d09eb2f...`) + `patch_qsadet.py` | mounted read-only; a modified file of vLLM |
| `_C_det.so` | built on the Spark by `serve/prepare.sh kernel`, not versioned | the build is not reproducible to the byte; the witness is the op `torch.ops._C_det.persistent_topk` resolved at the end of the build |

Image hashes before the hooks: `../../serve/image-base.sha`. sha256 of every file here: `../../MANIFEST.sha256`.

## Review (2026-10-13)

- No network, no `subprocess`, no `eval` or `exec` in the two hooks; each fails by assertion if its hook point is
  missing or appears twice. The only URLs of the kernel are attribution comments.
- Diff against the image: one line in `scheduler.py`, two in `mamba_hybrid.py`, one block gated by an environment
  variable in `qsa_indexer.py`.
- The build runs in a container without network, CPU only, capped at 6 GB.

## What blazux measured (inspiration, not our measurement)

Time to first token from 14 s to 1.4 s on a repeated 20 000-token prefix, cold and cached outputs identical to the
bit; deterministic top-k 4/4 stable. Our own witness is `bench/prefix_cache.py` (results in `results/`).
