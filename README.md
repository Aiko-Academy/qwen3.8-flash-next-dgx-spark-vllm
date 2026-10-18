# Qwen3.8-Flash-Next on one DGX Spark, with vLLM v0.32.0

A serving recipe for Qwen3.8-Flash-Next (NVIDIA NVFP4 checkpoint) on a single NVIDIA DGX Spark: 524 288 tokens of
context, MTP speculative decoding, 16 concurrent sequences, tool calls. It is the configuration we serve every day,
extracted into one script, with every patch it needs, the benches that measured it, and credit to the people whose
work it assembles (`CREDITS.md`).

Maintained by Aiko Academy. Measured on 2026-10-18.

## Figures

One DGX Spark, the server started from this repository (`serve/serve.sh`) on vLLM v0.32.0, read back by
`serve/verify.sh` (38 OK, `results/verify-v0320-public-set-1018.json`). Clients on another machine, over the network,
one pass at a time, every pass checked clean by `bench/idle_pass.py` (`docs/method.md`); llama-benchy runs on the
Spark itself, and its window was checked with the server's access log instead. Three columns, measured the same day by
the same commands: this recipe; the same recipe restarted with the draft vocabulary we use ourselves, built on private
texts (not published); the previous version of this recipe, on vLLM v0.31.0 with the public draft set (tag
`v0.31.0-1`).

| measure | this recipe | v0.32.0, our private draft set | v0.31.0, tag `v0.31.0-1` | protocol | files |
|---|---|---|---|---|---|
| decoding, French prose | 41.1 tok/s | 41.2 | 40.9 | `bench/probe.py`: one stream, greedy, 600 tokens, reasoning off, second pass | `probe-v0320-public-set-1018`, `probe-v0320-private-set-1018`, `probe-public-set-1018` |
| decoding, English prose | 45.3 tok/s | 42.6 | 44.7 | same | same |
| decoding, code (TypeScript) | no value: invented tool calls, see below | no value, same | 51.6 | same | same |
| decoding, first pass (new text), French / English | 40.1 / 43.9 | 39.8 / 41.6 | 39.7 / 43.3 | same, first pass | same |
| MTP acceptance by position, French prose | 0.637, 0.343 | 0.636, 0.354 | 0.603, 0.387 | same | same |
| MTP acceptance by position, English prose | 0.749, 0.461 | 0.708, 0.372 | 0.751, 0.448 | same | same |
| aggregate throughput, 1 / 4 / 8 / 16 clients | 39.6 to 41.3 / 90.4 to 91.8 / 133.0 to 133.6 / 194.6 to 195.1 tok/s | 40.3 to 41.0 / 91.9 to 93.0 / 126.8 to 134.7 / 193.8 to 200.9 | 39.8 / 89.0 to 92.7 / 131.0 to 134.4 / 194.1 to 197.5 | `bench/load_bench.py`: 400 tokens per client, temperature 0, two passes | `load-v0320-public-set-1018*`, `load-v0320-private-set-1018-b` and `-c`, `load-public-set-1018*` |
| time to first token at 16 clients | 0.59 to 0.72 s | 0.60 to 0.62 s | 0.59 to 0.61 s | same | same |
| prefill, 8 023 / 23 406 tokens | 2 682 to 2 703 / 2 714 to 2 719 tok/s | 2 705 to 2 759 / 2 776 to 2 777 | 2 643 to 2 771 / 2 740 to 2 797 | same, median of three requests, two passes | same |
| prefix cache, 18 517-token prompt | cold 13.02 s, cached 3.85 and 3.58 s, 32 960 tokens hit, outputs identical | not measured | cold 9.67 s, cached 3.32 and 3.35 s, 32 960 tokens hit, outputs identical | `bench/prefix_cache.py` | `prefix-cache-v0320-public-set-1018`, `prefix-cache-public-set-1018` |
| llama-benchy, generation tg32 / tg128 / tg512 | 39.9 / 42.3 / 40.3 tok/s | not measured | 36.2 / 37.5 / 39.7 | `bench/community/llama-benchy.sh` (0.4.0, run on the Spark, depth 0, 3 runs) | `llama-benchy-v0320-public-set-1018`, `llama-benchy-public-set-1018` |
| llama-benchy, same at depth 8 192 | 33.9 / 33.6 / 36.8 tok/s | not measured | 36.6 / 35.8 / 36.4 | same | same |
| llama-benchy, prefill pp2048 | 2 024 ± 814 tok/s | not measured | 2 603 ± 20 | same | same |
| tool call (qwen3_xml) | well formed | well formed | well formed | `bench/probe.py` | probe files |

Code has no value on v0.32.0. To the probe's code prompt (a TypeScript module, no tool declared), the greedy answer
on v0.32.0 is one sentence of intent ("I'll start by exploring the project structure..."), then stops after 148
generated tokens with the public set, 304 with ours, most of which the client never sees. The sampled witness below
shows what they are: calls to `list_files` and `read_file`, tools the request never declared, which the parser drops.
A speed measured on that text is not a speed of code. On v0.31.0 the same greedy prompt answered with the module at
each of the six greedy passes of the day (600 tokens, the probe's limit). Sampled at temperature 0.7, the code prompt
gives invented calls in 6 of 12 draws with the public set and 11 of 12 with ours; the French prose prompt, in none of
12 (table below). Not measured yet: the same test on v0.31.0, and with a system message, reasoning on, or tools
declared.

The draft vocabulary only decides which tokens the drafter may propose; the target model checks every one. On
v0.32.0 the two sets are level in French prose (41.1 and 41.2 tok/s), and the public set is faster in English (45.3
against 42.6). On v0.31.0 our set was ahead in French (43.4 against 40.9, `probe-private-set-1018-return`); why the gap
closed on v0.32.0 is not established. Under load, the gap between the two sets is of the order of the spread of two
passes (our set's two passes at 8 clients: 126.8 and 134.7).

Against v0.31.0, with the same public set: prose within 1.3 %, the load bench within the spread of two passes.
llama-benchy generation is higher at depth 0 and lower at depth 8 192, by gaps of the order of its own spread (± 1.3 to
7.6 tok/s on these rows); its prefill pp2048 spread on v0.32.0 (± 814) is too wide to compare. The cold pass of the
prefix-cache witness is longer (13.02 s against 9.67 s); a cold pass depends on the state of the page cache
(`docs/method.md`), and it was measured once.

Measured once per version, the draft set named in each cell:

| measure | v0.32.0 | v0.31.0 | protocol | files |
|---|---|---|---|---|
| long context with a needle | 449 365 tokens read, 3 reasoning items out of 3, needle found (our set) | the same (our set) | `bench/reasoning_bench.py --depths 510000` | `reasoning-deep-510k-v0320-private-set-1018`, `reasoning-deep-510k-off-1018` |
| short reasoning items | 15 / 15 (public set) | 15 / 15 (our set) | `bench/reasoning_bench.py --short-only`, reasoning off | `reasoning-v0320-short-off-1018`, `reasoning-short-off-1018` |
| code gate, mechanical judge | 4 / 4 (public set) | 4 / 4 (our set) | `bench/quality/code_gate.py` then `judge_code.py`: Solidity compiles, TypeScript and bash parse | `code-gate-v0320-off-1018`, `code-gate-off-1018` |
| literal `<tool_call>` written in prose | 32 complete answers out of 36, 4 short stops, no content lost, no fabricated call (public set) | 35 out of 36, 1 short stop, no content lost, no fabricated call (our set) | `bench/toolcall.py`, temperature 0.7 | `toolcall-v0320-public-set-1018`, `toolcall-private-set-1018` |
| invented tool calls, code prompt, no tool declared | 6 of 12 (public set), 11 of 12 (our set) | not measured | `bench/toolcall.py --cases code_without_tool --series 3 --per-series 4`, temperature 0.7 | `toolcall-unprompted-v0320-1018`, `toolcall-unprompted-v0320-private-set-1018` |
| invented tool calls, French prose prompt, no tool declared | 0 of 12 (public set), 0 of 12 (our set) | not measured | same, `prose_fr_without_tool` | same |
| NVMe reads of decoding | not measured | disk wake-up about 10 ms after 120 ms of silence; 16 streams of new text: 140.0 tok/s, 5.4 ms per read (our set) | `bench/nvme_reads.py` | `nvme-reads-private-set-1018` |

How much of the model's output each draft set covers (share of generated tokens inside the set, measured on the
v0.31.0 outputs, `tools/draft_vocab_coverage.py`, `results/draft-vocab-coverage-1018.json`):

| text | public set (this repository) | blazux's published set | our private set |
|---|---|---|---|
| probe, French prose | 98.8 % | 79.0 % | 99.2 % |
| probe, English prose | 98.0 % | 99.7 % | 99.8 % |
| probe, code | 99.2 % | 99.2 % | 99.7 % |
| code gate answers (Solidity, ethers, React, bash) | 94.0 to 95.7 % | 97.2 to 98.5 % | 100 % |

## What the recipe does

| layer | what | where |
|---|---|---|
| engine | vLLM v0.32.0, official image pinned by digest | `serve/serve.sh` |
| checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` at a pinned revision; dense BF16 layers converted to block FP8; the QSA indexer projection kept apart from q, k and v | `serve/prepare.sh`, `patches/fp8-hybrid/` |
| memory | per-layer embedding table (51.2 GB) read in place from the checkpoint, rows prefetched with `MADV_WILLNEED` | `patches/ple-checkpoint-mapped/` (vllm#58439) |
| context | 524 288 tokens, YaRN factor 2 over 262 144; KV cache fp8, MTP layer in BF16 (`--kv-cache-dtype-skip-layers 48`, honoured by vLLM since v0.32.0, vllm#60023) | `serve/serve.sh` |
| speculative decoding | MTP 2; the drafter scores 65 536 rows of the head, in NVFP4 on Marlin | `patches/draft-vocab/`, `patches/draft-nvfp4-head/` |
| cache and determinism | prefix cache repaired (the scheduler half; the worker half is in vLLM since v0.32.0, vllm#55601); deterministic sparse-attention top-k | `patches/cache-det/` |
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
running container; `--print` shows the docker run. The version on vLLM v0.31.0 stays at tag `v0.31.0-1`.

## Limits

- vLLM v0.32.0 only. The overlays replace files of that image and are checked against its digest
  (`serve/image-base.sha`); a newer image needs them redone, or dropped where upstream has merged them.
- On v0.32.0, a request for code with no tool declared often comes back as calls to tools that do not exist (6 and
  11 of 12 draws at temperature 0.7, depending on the draft set). The parser drops them and the client receives a
  sentence of intent. Not measured yet: the same test on v0.31.0, and with a system message, reasoning on, or tools
  declared.
- One machine, one model. Throughput figures are ours, on one Spark, at our protocol; other people's figures are
  quoted in `CREDITS.md` and `docs/negative-results.md` with theirs.
- Changing the draft vocabulary changed greedy prose in our A-B-A on v0.31.0 (code did not): `docs/method.md`.
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
