# Negative results

What we tried on this machine and did not keep, with the measurement that decided. These were measured on our
Spark between 2026-10-12 and 2026-10-18 with the scripts that `bench/` derives from (same prompts and protocols
unless stated); the raw files are not published. The probe is greedy, 600 tokens, reasoning off, second pass.

## Summary

| experiment | date | what it gave | why it was not kept |
|---|---|---|---|
| NVMe APST delays raised to 2 s and 60 s | 2026-10-18 | disk wake-up 10.3 ms to 1.5 ms, 6.3 ms to 0.15 ms per read at 16 streams | throughput unchanged in A-B-A |
| NVIDIA FP4-Local checkpoint on a community vLLM image | 2026-10-17 | code +21 % | French prose -18 %, KV pool -21 %, tool-call witness 1 complete answer out of 36 |
| reasoning at effort xhigh on code | 2026-10-17 | right and fast on short questions | cut by the 16 384-token budget on 12 of 14 code tasks |
| MTP 3 instead of MTP 2 | 2026-10-12 | code +8 % | French prose -5 % |
| output head in FP8 | 2026-10-12 | prose +9 %, code +3 % | outputs change, two quality gates regressed |
| 262 144 tokens native instead of 524 288 with YaRN | 2026-10-12 | prose +4 %, code +12 % | KV pool -12 %, outputs change, a quality gate regressed, half the context |
| nightly image with the SM121 skinny-GEMM plans (vllm#59753) | 2026-10-12 | estimated +0.5 to +1 % | below any useful threshold; studied without download |

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
