# Negative results

What we tried on this machine and did not keep, with the measurement that decided. These were measured on our
Spark between 2026-10-12 and 2026-10-20 with the scripts that `bench/` derives from (same prompts and protocols
unless stated); the raw files are not published. The probe is greedy, 600 tokens, reasoning off, second pass.

## Summary

| experiment | date | what it gave | why it was not kept |
|---|---|---|---|
| a tuned config for the MTP layer's Triton FP8 MoE kernel | 2026-10-20 | the image's GB10 config of the nearest shape: 0 % of a decoding step at one client, 0.05 % at 16 clients | under the 1 % set before the measure; any config is bounded at 0.34 % of a one-client step |
| MTP 3 instead of MTP 2, on vLLM v0.32.0 | 2026-10-19 and 2026-10-20 | French prose +6.8 %, code +21.4 % | greedy outputs change, and a private code gate fell from 5 of 6 to 3 of 6 tasks, with a systematic error on one task |
| long-prefill threshold of 1 024 tokens (`--long-prefill-token-threshold`) | 2026-10-19 | a short stream 2.4 and 2.2 times faster during a long prefill | still under 10 tok/s, and the long prompt waits 56 to 66 % longer for its first token |
| Qwen's non-thinking sampling instead of the checkpoint's generation_config | 2026-10-19 | 16 clients 2.6 % faster | one client (French prose) level: 38.0 against 38.1 tok/s, under the 3 % set before the measure |
| vllm#62409 as it stands, for the drafter's M-RoPE cache | 2026-10-19 | drafter cache 524 288 rows | it also moves 8 of the 32 YaRN frequencies of the target; an adapted floor is served instead (`patches/mrope-draft-floor`) |
| NVMe APST delays raised to 2 s and 60 s | 2026-10-18 | disk wake-up 10.3 ms to 1.5 ms, 6.3 ms to 0.15 ms per read at 16 streams | throughput unchanged in A-B-A |
| NVIDIA FP4-Local checkpoint on a community vLLM image | 2026-10-17 | code +21 % | French prose -18 %, KV pool -21 %, tool-call witness 1 complete answer out of 36 |
| reasoning at effort xhigh on code | 2026-10-17 | right and fast on short questions | cut by the 16 384-token budget on 12 of 14 code tasks |
| MTP 3 instead of MTP 2 | 2026-10-12 | code +8 % | French prose -5 % |
| output head in FP8 | 2026-10-12 | prose +9 %, code +3 % | outputs change, two quality gates regressed |
| 262 144 tokens native instead of 524 288 with YaRN | 2026-10-12 | prose +4 %, code +12 % | KV pool -12 %, outputs change, a quality gate regressed, half the context |
| nightly image with the SM121 skinny-GEMM plans (vllm#59753) | 2026-10-12 | estimated +0.5 to +1 % | below any useful threshold; studied without download |

## MoE kernel of the MTP layer (2026-10-20, 09:39, server stopped)

The MTP layer is the only layer whose routed experts go through vLLM's Triton FP8 MoE kernel (E = 512, N = 640,
hidden 2 560, top-k 10, `fp8_w8a8`, block [128, 128]); the 48 target layers are NVFP4 and use another kernel. The
v0.32.0 image ships no tuned config for this shape on the GB10, so the kernel runs its default config. Read in the
v0.32.0 code (`v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py`): with MTP 2 a decoding step calls this
MoE twice, once on the step's tokens (3 per request) and once on 1 token per request, so M = 3 and 1 at one client,
48 and 16 at 16 clients.

`benchmark_config` from the image's `benchmarks/kernels/benchmark_moe.py` (a CUDA graph of 10 calls, 20 replays),
default config against the image's GB10 config for the nearest shape (E = 512, N = 512), microseconds per call:

| M | default | GB10 config, N = 512 |
|---|---|---|
| 1 | 231.74 | 234.97 |
| 3 | 652.04 | 660.85 |
| 16 | 2 961.43 | 2 954.38 |
| 48 | 6 693.13 | 6 610.82 |

Rule written before the measure: the gain on the two calls of a step, over the step (48.3 ms at one client, from the
probe; 171 ms at 16 clients, derived from the load bench: 2.43 tokens per step per stream at 14.2 tok/s), under 1 % at
both loads means no tuning. It came out at 0 % at one client (the neighbour config is slower at M = 1 and 3) and
0.05 % at 16 clients. Any config is bounded too: the two calls of a one-client step take 0.88 ms against a floor of
about 0.72 ms to read 40 experts of 4.9 MB at 273 GB/s, so 0.34 % of the step at most. A full `--tune` sweep (1 920
configs, hours of GPU) was not run. On the GB10, CUDA counts only strictly free memory (1.58 GB beside the server,
not the page cache), so this measure needs the server stopped. Not kept.

## MTP 3 on vLLM v0.32.0 (2026-10-19 and 2026-10-20)

On v0.32.0 a verification step at M = 4 falls into the SM121 skinny-GEMM plans, which reopened the 2026-10-12 result
below. A-B-A on the served recipe, same day, same commands, our private draft set; probe with a system message,
second pass, tok/s:

| measure | MTP 2 (A) | MTP 3 (B) | MTP 2 (A') |
|---|---|---|---|
| French prose | 43.27 | 46.22 | 43.31 |
| English prose | 44.67 | 42.21 | 44.73 |
| code | 55.45 | 67.34 | 55.53 |
| load, 1 / 4 / 8 / 16 clients, greedy | 40.0 / 95.3 / 132.1 / 192.3 | 38.1 / 87.0 / 135.0 / 203.7 | 39.2 / 89.6 / 134.3 / 196.9 |
| load, 1 / 8 / 16 clients, sampled as our clients (T 1.0, top_p 0.95, top_k 20) | 40.4 / 119.4 / 178.1 | 38.3 / 118.9 / 182.5 | 39.0 / 118.9 / 181.4 |

The speed criterion written before the measure held (French prose at least level, code at least +5 %, 8 and 16
clients not under the A arms). MTP 3 changes the greedy texts, so the quality gates of MTP 2 were replayed on it: all
held but one, a private code gate of three tasks taken from a payment node, 3 of 6 against at least 5 of 6. In both
passes MTP 3 wrote the same exchange-rate function, which triangulates as (USD to destination) / (USD to source)
on a rounded inverse instead of (source to USD) x (USD to destination) as the task states; the hidden case misses by
1.65e-9. The next night, six passes of the same gate on MTP 2 gave 15 of 18, the triangulation case 6 of 6: the bar
of 5 of 6 holds on the default. Not kept.

## Long-prefill threshold (2026-10-19, 22:19 to 23:19)

A short stream decoding while a long prompt prefills gets about 2 tok/s on the default recipe: each step carries
two 1 648-token blocks of the long prompt. `--long-prefill-token-threshold 1024` (batches of 4 096 unchanged) in A-B-A:

| measure | default (A) | threshold 1 024 (B) | default (A') |
|---|---|---|---|
| short stream during a 23 393-token prefill, tok/s | 2.07 | 4.96 | 2.01 |
| short stream during a 100 005-token prefill, tok/s | 1.85 | 4.03 | 1.86 |
| longest run under 10 tok/s, 23 393 / 100 005 tokens, s | 6 / 36 | 12 / 57 | 8 / 35 |
| first token of the long prompt, 23 393 / 100 005 tokens, s | 8.69 / 37.86 | 14.42 / 59.22 | 10.56 / 37.69 |
| prefill alone, 8 023 / 23 406 tokens, tok/s | 2 726 / 2 758 | 2 667 / 2 711 | 2 727 / 2 765 |

The criterion written before (the short stream at least twice as fast, prefill alone within 10 %) held, but the
short stream stays under 10 tok/s and the long prompt waits 56 to 66 % longer: a trade between the two requests,
not a fix. Read in the v0.32.0 scheduler afterwards: in Mamba `align` mode a prefill chunk that does not end the
prompt is cut back to a block boundary unless the configured threshold is under one block, so the threshold of
1 024 gives chunks of 1 024 then 624 tokens; `--long-prefill-token-threshold-adaptive` raises it to the fair share of
the batch, 2 048 for two requests, which the alignment brings to one block of 1 648, about 1.6 times the default
short-stream speed by the same arithmetic. That variant was not run. `--prefill-schedule-interval` acts only in the
data-parallel engine core, not on one engine. Not kept.

## Sampling: Qwen's non-thinking recommendation (2026-10-19, 11:28 to 11:44)

Our clients send no sampling parameters and get the checkpoint's `generation_config` (temperature 1.0, top_p 0.95,
top_k 20, the values of the reasoning mode). The model card recommends, for the non-thinking mode we serve, temperature
0.7, top_p 0.80, top_k 20, presence_penalty 1.5. `bench/load_bench.py` with `--no-prefill --clients 1,8,16`, no seed
(a per-request seed takes vLLM off its fused FlashInfer sampler), three repetitions of each setting alternated between
two greedy passes, our private draft set. Aggregate tok/s:

| pass | 1 client | 8 clients | 16 clients | files |
|---|---|---|---|---|
| greedy, before | 41.4 | 132.5 | 196.8 | `load-v0320-private-set-greedy-1019-a` |
| generation_config (T 1.0, top_p 0.95, top_k 20), three repetitions | 38.7 / 38.2 / 37.4 | 121.7 / 126.8 / 119.5 | 179.2 / 178.5 / 178.3 | `load-v0320-private-set-s7-r1-1019`, `-r2`, `-r3` |
| non-thinking (T 0.7, top_p 0.80, top_k 20, presence 1.5), three repetitions | 37.2 / 37.5 / 39.4 | 128.9 / 125.8 / 124.6 | 181.8 / 184.1 / 183.9 | `load-v0320-private-set-s6-r1-1019`, `-r2`, `-r3` |
| greedy, after | 41.9 | 134.3 | 196.1 | `load-v0320-private-set-greedy-1019-b` |

The criterion, written before: at least 3 % on one client, with no overlap between the repetitions. One client: 38.0
against 38.1 on average; not kept, so the quality gates were not run. What the table does show: sampled as our
clients are, throughput is 7 to 9 % under the greedy figures, and MTP acceptance at position 1 drops by about five
points (0.63 against 0.68 at one client). Greedy gains overstate what sampled clients get.

## vllm#62409 as it stands (2026-10-19, studied in the image, not served)

The pull request floors the M-RoPE cache on the engine's `max_model_len`, for every M-RoPE. In `MRotaryEmbedding`
that number is also the base of the YaRN correction range: in the v0.32.0 image, on CPU, the target built with our
YaRN parameters gets 1 048 576 cache rows instead of 524 288 and 8 of its 32 rotary frequencies change, so its outputs
would change. The served overlay applies the floor without YaRN only (`patches/mrope-draft-floor/PROVENANCE.md`).

## NVMe APST, A-B-A (2026-10-18, 02:18 to 03:14)

The disk of the Spark falls asleep 100 ms after its last read (kernel defaults `apst_primary_timeout_ms` 100,
`apst_secondary_timeout_ms` 2000) and takes about 10 ms to wake up. The per-layer embedding table is read from that
disk while decoding. Idea from a community setup (christopherowen/spark-ds41f and NVIDIA forum thread 384438).

| measure | A, defaults | B, 2 s and 60 s | A, defaults again |
|---|---|---|---|
| 4 KiB read after 120 to 1 000 ms of silence (median) | 10.2 to 10.5 ms | 1.4 to 1.6 ms | 9.8 to 10.0 ms |
| new text, 16 streams, throughput | 146.6 and 150.7 tok/s | 155.9, 154.8 and 156.4 | 160.2 and 155.7 |
| new text, 16 streams, latency per read | 6.26 and 6.31 ms | 0.14 and 0.15 ms | 6.06 and 6.60 ms |
| load bench, 1 / 4 / 8 / 16 clients | 37.0 / 83.5 / 116.6 / 170.0 | 37.3 / 85.0 / 129.2 / 194.3 | 39.1 / 87.3 / 129.0 / 187.6 |

The setting acts on the disk but not on throughput: back on the defaults, 16 streams and 8 clients give as much as
in B. The prefetch of table rows with `MADV_WILLNEED` hides the wake-up. The gap between the two A arms (116.6 then
129.0 at 8 clients) is larger than any gap between A and B. Witness: `bench/nvme_reads.py`.

## NVIDIA FP4-Local checkpoint (2026-10-17)

`nvidia/Qwen3.8-Flash-Next-FP4-Local` (dense and attention layers in MXFP8, experts in NVFP4) on the community
image `eugr/spark-vllm-b12x` pinned by digest, with the DGX Spark recipe of its model card: MTP 4, the embedding
table on disk through io_uring, 524 288 tokens with YaRN, 16 sequences, the same CPU set.

| measure | this recipe | FP4-Local | gap |
|---|---|---|---|
| KV pool | 1 281 072 tokens | 1 007 613 | -21 % |
| probe, French prose | 43.2 tok/s | 35.4 | -18 % |
| probe, code | 51.8 tok/s | 62.9 | +21 % |
| MTP acceptance, prose, by position | 0.684, 0.418 (MTP 2) | 0.668, 0.363, 0.187, 0.069 (MTP 4) | positions 3 and 4 almost always rejected in prose |
| tool-call witness, 36 requests | 31 complete | 1 complete, 27 contents lost | strong regression |
| prefix cache witness | green, 3.8 s cached | 3.1 s cached, cold output equal to cached once out of twice | |

The image's parser does not carry the repaired tool parser, and the checkpoint itself writes the literal
`<tool_call>` many times outside code in Hungarian with a tool declared. Not kept: French prose is our dominant use.

## Reasoning at effort xhigh on code (2026-10-17)

Reasoning modes compared on the same server, the same evening, the same commands. Temperature 0.1 for the code
gates, budget 16 384 tokens in reasoning modes.

| measure | off | low | medium | xhigh |
|---|---|---|---|---|
| code gate, 4 tasks (the three short tasks of `bench/quality/code_gate.py`, and a long task on another document), two passes | 7/8 | 8/8 | 8/8 | 0/4 in one pass: four `length` finishes without an answer |
| a second, private code gate (3 tasks) | 4/6 | 6/6 | 5/6 | 0/3: three `length` finishes |
| short reasoning items (15; three of them since replaced in `bench/reasoning_bench.py`) | 14/15 | 15/15 | 15/15 | 15/15 |

At effort xhigh the model reasons at length on code: cut by the budget on 12 of the 14 code tasks played that
evening (clean and discarded passes together); with a 32 000-token budget it answers after 8 to 12 minutes per task.
It does not loop. `serve.sh` turns reasoning off by default; low and medium are a request option
(`bench/reasoning_modes.py`).

## MTP 3 (2026-10-12)

One variable changed: three draft tokens instead of two, on the profile of that day (dense layers in FP8, reduced
draft vocabulary, draft head in BF16).

| measure | MTP 2 | MTP 3 | gap |
|---|---|---|---|
| probe, French prose, pass 1 / 2 | 32.2 / 33.2 | 30.7 / 31.3 | -5 % |
| probe, code, pass 1 / 2 | 37.8 / 39.0 | 40.9 / 41.9 | +8 % |
| step | 60 to 64 ms | 68 to 72 ms | +13 % |
| acceptance at position 3, prose / code | | 0.157 / 0.537 | |

The third draft pass costs about 8 ms per step; it pays in code, not in French prose.

## Output head in FP8 (2026-10-12)

The target `lm_head` (248 320 x 2 560) quantized to block FP8 at load time. A first attempt with an FP8 head written
in the checkpoint failed at load on v0.31.0: `ParallelLMHead` is built without a quantization config, and the
vocabulary loader refuses block scales.

| measure | head BF16 | head FP8 |
|---|---|---|
| probe, French prose, pass 1 / 2 | 31.8 / 32.5 | 34.8 / 35.6 |
| probe, code | 37.3 / 38.3 | 38.5 / 39.4 |
| greedy texts | | prose differs from character 656, code from 107 |
| mechanical judge of the code gates | 6/7 | 5/7 (a Solidity contract no longer compiles) |
| writing gate (six business texts) | no invented fact | invented facts in one e-mail |

The head decides every token: outputs change, and the two gates that matter regressed. Speed does not buy that.

## 262 144 tokens native (2026-10-12)

The same server without YaRN, at the checkpoint's native 262 144 tokens.

| measure | 524 288 with YaRN | 262 144 native |
|---|---|---|
| probe, French prose, pass 1 / 2 | 32.8 / 33.9 | 34.2 / 35.1 |
| probe, code | 40.4 / 42.1 | 45.4 / 46.9 |
| KV pool | 1 397 141 tokens | 1 224 212 (-12 %, not explained) |
| greedy texts | | differ from the start (YaRN changes every position) |
| writing gate | no invented fact | invented facts in one e-mail |

## Nightly image with the SM121 skinny-GEMM plans (2026-10-12, studied without download)

vllm-project/vllm#59753 adds GEMM plans for SM121 at M = 1, 2, 4, 8 and 16, BF16 only. Read at its commit and
against v0.31.0, it does not reach this recipe: the verification step runs at M = 3 (MTP 2), our dense layers are
FP8, and the announced gain came from the full output head read by the drafter, which the reduced draft vocabulary
already removes. Estimate on our step: +0.5 to +1 %. Not downloaded; to reopen with vLLM v0.32.0.
