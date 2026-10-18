# Qwen3.8-Flash-Next on one DGX Spark, with vLLM v0.31.0

A serving recipe for Qwen3.8-Flash-Next (NVIDIA NVFP4 checkpoint) on a single NVIDIA DGX Spark: 524 288 tokens of
context, MTP speculative decoding, 16 concurrent sequences, tool calls. It is the configuration we serve every day,
extracted into one script, with every patch it needs, the benches that measured it, and credit to the people whose
work it assembles (`CREDITS.md`).

Maintained by Aiko Academy. Measured on 2026-10-18.

## Figures

One DGX Spark, the server started from this repository (`serve/serve.sh`), read back by `serve/verify.sh` (38 OK,
`results/verify-public-set-1018.json`). Clients on another machine of the local network, one pass at a time, every
pass checked clean (`docs/method.md`). The right-hand column is the same server with the draft vocabulary we use
ourselves, built on private texts (not published), measured the same day by the same commands.

| measure | this recipe | same, our private draft set | protocol | files |
|---|---|---|---|---|
| decoding, French prose | 40.9 tok/s | 43.4 | `bench/probe.py`: one stream, greedy, 600 tokens, reasoning off, second pass | `probe-public-set-1018`, `probe-private-set-1018-return` |
| decoding, English prose | 44.7 tok/s | 43.7 | same | same |
| decoding, code (TypeScript) | 51.6 tok/s | 51.9 | same | same |
| decoding, first pass (new text), French / English / code | 39.7 / 43.3 / 51.0 | 42.1 / 42.7 / 51.2 | same, first pass | same |
| MTP acceptance by position, French prose | 0.603, 0.387 | 0.684, 0.418 | same | same |
| MTP acceptance by position, code | 0.86, 0.776 | 0.86, 0.776 | same | same |
| aggregate throughput, 1 / 4 / 8 / 16 clients | 39.8 / 89.0 to 92.7 / 131.0 to 134.4 / 194.1 to 197.5 tok/s | 40.5 to 41.0 / 91.8 to 94.9 / 134.6 to 134.8 / 200.5 to 201.1 | `bench/load_bench.py`: 400 tokens per client, temperature 0, two passes | `load-public-set-1018*`, `load-private-set-1018-return*` |
| time to first token at 16 clients | 0.59 to 0.61 s | 0.59 to 0.60 s | same | same |
| prefill, 8 023 / 23 406 tokens (second pass) | 2 771 / 2 797 tok/s | 2 763 / 2 784 | same, cold prompts | same |
| prefix cache, 18 517-token prompt | cold 9.7 s, cached 3.3 s, 32 960 tokens hit, outputs identical | cold 10.1 s, cached 3.8 s | `bench/prefix_cache.py` | `prefix-cache-public-set-1018`, `prefix-cache-private-set-1018` |
| llama-benchy, generation tg32 / tg128 / tg512 | 36.2 / 37.5 / 39.7 tok/s | 36.4 / 38.1 / 38.4 | `bench/community/llama-benchy.sh` (0.4.0, run on the Spark, depth 0, 3 runs) | `llama-benchy-public-set-1018`, `llama-benchy-private-set-1018-return` |
| llama-benchy, same at depth 8 192 | 36.6 / 35.8 / 36.4 tok/s | 35.0 / 35.5 / 40.2 | same | same |
| llama-benchy, prefill pp2048 | 2 603 tok/s | 2 618 | same | same |
| tool call (qwen3_xml) | well formed | well formed | `bench/probe.py` | probe files |

The draft vocabulary only decides which tokens the drafter may propose; the target model checks every one. With
the public set, French prose decodes about 6 % slower than with our set, English 2 % faster, code the same. Under
load, the public set is 1.5 to 2.7 % below ours (mean of two passes); llama-benchy shows no difference beyond its own
spread.

Measured once, on the same server with our private draft set (these measures do not depend on draft speed):

| measure | result | protocol | file |
|---|---|---|---|
| long context with a needle | 449 365 tokens read, 3 reasoning items out of 3, needle found | `bench/reasoning_bench.py --depths 510000` | `reasoning-deep-510k-off-1018` |
| short reasoning items | 15 / 15 | `bench/reasoning_bench.py --short-only`, reasoning off | `reasoning-short-off-1018` |
| code gate, mechanical judge | 4 / 4 (Solidity compiles, TypeScript and bash parse) | `bench/quality/code_gate.py` then `judge_code.py` | `code-gate-off-1018` |
| literal `<tool_call>` written in prose | 35 complete answers out of 36, 1 short stop, no content lost, no fabricated call | `bench/toolcall.py`, temperature 0.7 | `toolcall-private-set-1018` |
| NVMe reads of decoding | disk wake-up about 10 ms after 120 ms of silence; 16 streams of new text: 140.0 tok/s, 5.4 ms per read | `bench/nvme_reads.py` | `nvme-reads-private-set-1018` |

How much of the model's output each draft set covers (share of generated tokens inside the set,
`tools/draft_vocab_coverage.py`, `results/draft-vocab-coverage-1018.json`):

| text | public set (this repository) | blazux's published set | our private set |
|---|---|---|---|
| probe, French prose | 98.8 % | 79.0 % | 99.2 % |
| probe, English prose | 98.0 % | 99.7 % | 99.8 % |
| probe, code | 99.2 % | 99.2 % | 99.7 % |
| code gate answers (Solidity, ethers, React, bash) | 94.0 to 95.7 % | 97.2 to 98.5 % | 100 % |

## What the recipe does

| layer | what | where |
|---|---|---|
| engine | vLLM v0.31.0, official image pinned by digest | `serve/serve.sh` |
| checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` at a pinned revision; dense BF16 layers converted to block FP8 | `serve/prepare.sh`, `patches/fp8-hybrid/` |
| memory | per-layer embedding table (51 GB) read in place from the checkpoint, rows prefetched with `MADV_WILLNEED` | `patches/ple-checkpoint-mapped/` (vllm#58439) |
| context | 524 288 tokens, YaRN factor 2 over 262 144; KV cache fp8, MTP layer in BF16 | `serve/serve.sh`, `patches/mtp-kv-bf16/` (vllm#60023) |
| speculative decoding | MTP 2; the drafter scores 65 536 rows of the head, in NVFP4 on Marlin | `patches/draft-vocab/`, `patches/draft-nvfp4-head/` |
| cache and determinism | prefix cache repaired; deterministic sparse-attention top-k | `patches/cache-det/` |
| tools | Qwen tool parser repaired: a literal `<tool_call>` in prose no longer cuts the answer | `patches/tool-parser/` |
| CPU | vLLM pinned on the ten Cortex-X925 cores; 16 sequences; recurrent state in BF16 | `serve/serve.sh` |

Reasoning is off by default; a request turns it on with `chat_template_kwargs` (`bench/reasoning_modes.py`).

## Prerequisites

| item | value |
|---|---|
| machine | one NVIDIA DGX Spark (GB10, 128 GB unified memory); the model takes the whole GPU (memory share 0.80) |
| system | DGX OS with Docker and the NVIDIA container runtime; measured on kernel 7.0.0-1019-nvidia, driver 580.178.04 |
| disk | about 124 GiB for the checkpoint, 71 GiB more for the converted copy (the rest is hard-linked), 22 GiB for the image |
| network | to download the image, the checkpoint and the public corpus once; none to serve |

## Start in three commands

On the Spark:

```bash
git clone https://github.com/Aiko-Academy/qwen3.8-flash-next-dgx-spark-vllm && cd qwen3.8-flash-next-dgx-spark-vllm
serve/prepare.sh all          # image, checkpoint, conversion, kernel, draft vocabulary (hours, mostly download)
serve/serve.sh && serve/verify.sh
```

`serve.sh` binds 127.0.0.1:8000 and serves the model as `qwen3.8-flash-next`. To serve the network, put a key in
a file (`VLLM_API_KEY=<key>`, `chmod 600`) and set `BIND` and `API_KEY_FILE` (`serve/config.example.env`). The first
completion comes 12 to 18 minutes after the start. `serve/serve.sh --check` runs every check without touching a
running container; `--print` shows the docker run.

## Limits

- vLLM v0.31.0 only. The overlays replace files of that image and are checked against its digest
  (`serve/image-base.sha`); a newer image needs them redone, or dropped where upstream has merged them.
- One machine, one model. Throughput figures are ours, on one Spark, at our protocol; other people's figures are
  quoted in `CREDITS.md` and `docs/negative-results.md` with theirs.
- Changing the draft vocabulary changed greedy prose in our A-B-A (code did not): `docs/method.md`.
- The tool parser still misses a `<tool_call>` example inside a code block indented by four spaces in a list item
  (`patches/tool-parser/PROVENANCE.md`).
- The public draft set covers code outside its corpus (Solidity, ethers, React) less well than ours: 94 to 96 %.
- The figures come from a server shared with other users, measured at rest; a pass with a third-party request is
  discarded and replayed.

## Repository

| path | content |
|---|---|
| `serve/` | `serve.sh`, `prepare.sh`, `verify.sh`, pinned checkpoint and image hashes, configuration example |
| `patches/` | the overlays, each with its `PROVENANCE.md` |
| `bench/` | probe, load, prefix cache, tool call, NVMe, reasoning, code gate, llama-benchy runner, frozen prompts |
| `tests/` | `offline.sh` (no Docker), and witnesses to run on the Spark |
| `results/` | the measurements above, with each pass verdict |
| `docs/` | `method.md`, `negative-results.md` |
| `tools/` | credits check, draft-set coverage, corpus freeze |

## Credits and licence

Our code is under Apache-2.0 (`LICENSE`, `NOTICE`). Third-party files keep their licence; `CREDITS.md` lists each
one with its author, repository, commit or pull request, and licence, and `tools/check_credits.py` checks the list.
The checkpoint is NVIDIA's and is downloaded, never redistributed. What we tried and dropped:
`docs/negative-results.md`.
