# Negative results

What we tried on this machine and did not keep, with the measurement that decided. These were measured on our
Spark between 2026-10-12 and 2026-10-21 with the scripts that `bench/` derives from (same prompts and protocols
unless stated); the raw files are not published. The probe is greedy, 600 tokens, reasoning off, second pass.

## Summary

| experiment | date | what it gave | why it was not kept |
|---|---|---|---|
| narrow GEMM plans at M = 3, 6 and 12 on SM121 (the measured M = 4, 8 and 16 tiles reused) | 2026-10-21 | verification step 2.2 to 3.0 % shorter at one client | greedy texts change from the first words, and two code gates fell: 7 of 8 (8 of 8 required) and 3 of 6 (at least 5 of 6) |
| screened target head (`--screened-lm-head`, vllm#60566, open) | 2026-10-20 and 21 | greedy probe 6.0 to 6.2 % faster at one client, same texts and acceptance | 8 clients 1.9 % faster (median of three passes per arm), under the 2 % gain set before |
| internal prefill checkpoints (vllm#60050 and #61601, merged after v0.32.0) | 2026-10-20 | warm replayed turns 14 to 42 % sooner, KV pool +5.6 % (1 664-token blocks) | the summed first token of the replayed turns fell 3.6 %, not 10 %, and cached outputs differed from cold outputs |
| one draft token instead of two above 8 requests (`num_speculative_tokens_per_batch_size` `[[1, 8, 2], [9, 16, 1]]`) | 2026-10-20 | the same probe texts and speed (two draft tokens kept at one client) | 16 clients 8.7 % slower greedy and 5.1 % slower sampled, where a 5 % gain was set before |
| CUDA graphs for decoding only (`cudagraph_mode` `FULL_DECODE_ONLY`) | 2026-10-20 | graph capture 0.34 GiB instead of about 1 GiB, KV pool +3.6 % | under the 5 % pool gain set before; one client 5.1 % slower on the load bench, the French prose probe with a system message 3.2 % slower |
| MoE experts on Marlin (`--moe-backend marlin`) | 2026-10-20 | one client 10.6 % faster on the load bench, 4 to 16 clients 4.4 to 4.5 % faster, KV pool +4.4 % | the greedy probe with a system message fell 3.3 % in French prose (a 2 % gain was set before), the 23 406-token prefill fell 4.4 %, and greedy texts changed |
| MTP index sharing (`index_share_for_mtp_iteration`) | 2026-10-20 | the greedy probe without a system message 2.4 to 6.1 % faster | with a system message, the acceptance at position 2 fell 7.9 % on one prompt (2 % at most, set before) and its speed 3.9 %; greedy texts changed |
| FlashInfer autotuning at start-up (without `--no-enable-flashinfer-autotune`) | 2026-10-20 | 16 clients 3.9 % faster greedy, 4.4 % and 5.2 % faster sampled at one and eight clients | the prefill gained 2.3 % at 8 023 tokens but 0.1 % at 23 406, under the 2 % set before at both sizes; the greedy probe with a system message fell 1.6 to 4.1 % and its texts changed |
| a finer prefix-cache unit (`--prefix-match-unit 16`) with the Mamba shared-prefix checkpoint | 2026-10-20 | the first token of the second and third turns of two replayed agent sessions came 19 to 59 % sooner | the tokens served by the cache rose 1.2 to 1.6 %, not the 20 % set before the measure, and the greedy probe texts changed |
| a fixed KV cache size (`--kv-cache-memory-bytes`, 18.33 GiB) instead of memory profiling | 2026-10-20 | the same pool on two restarts, no out-of-memory error | the first completion came 12 s sooner (11.9 against 12.1 min), for a pool 3.0 % smaller than the profiled one |
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

## Narrow GEMM plans at M = 3 (2026-10-21, 01:04 to 01:43, quality gates 01:53 to 02:40)

vLLM v0.32.0 runs the BF16 layers of this model through measured CuTe skinny-GEMM plans on SM121 at M = 1, 2, 4, 8 and
16 tokens (`models/qwen4_exp/nvidia/low_latency_gemm.py`), and through cuBLAS at any other M. With two draft tokens the
target verifies 3 tokens per request, and the CUDA-graph sizes 3, 6, 12, 24 and 48 have no plan. On this configuration
about 2.1 GB of BF16 weights per verification step go through cuBLAS there: the target head (1.27 GB), the
hyper-connection up projections (0.64 GB), the router, the indexer projection and the GDN B/A projection. We added
plans at M = 3, 6 and 12 that reuse the measured M = 4, 8 and 16 tiles of each shape with `num_rows` alone changed (one
file, 38 entries; the kernel takes M up to 16, so 24 and 48 stay on cuBLAS), and 26 more kernels were compiled at
warm-up. Criterion written before: the greedy probe with a system message (French prose and code) at least 2 % faster,
outside the gap between the two default arms, and greedy texts identical; otherwise the quality gates before any keep.

| measure | default (A) | M = 3 plans (B) | default (A') |
|---|---|---|---|
| probe with a system message, French prose, tok/s | 43.16 | 44.51 | 43.21 |
| probe with a system message, code, tok/s | 55.32 | 56.62 | 55.35 |
| time per verification step, five prompts, ms | 48.42 to 49.53 | 46.96 to 48.04 | 48.37 to 49.50 |
| greedy probe texts identical to A | | 0 of 10 | 10 of 10 |

The step became 2.2 to 3.0 % shorter on all five prompts, under a gap of 0.1 % at most between the default arms. But the
new kernels round differently from cuBLAS, and every greedy text changed within its first words, so the speed of the
probe mixes the kernel and a different acceptance. The quality gates were then replayed on the variant, against their
reference on the default (2026-10-18): the code gate gave 7 of 8 (a Solidity contract no longer compiles; 8 of 8 required), the private code gate of
three tasks 3 of 6 (at least 5 of 6); the other gates held (no em dash in the institutional writing, tool-call witness
33 of 36, YaRN short 14 of 15, no unprompted call in prose). Not kept.

## Screened target head (2026-10-20, 22:09 to 22:52, and 2026-10-21, 00:00 to 00:38)

vllm-project/vllm#60566 (open, head 229abbf75d) adds `--screened-lm-head`: an FP8 copy of the target head ranks the
248 320 rows, and only the best candidates are recomputed in BF16 from the original head. We applied it to v0.32.0 (six
files, no conflict). Criterion written before each measure: the greedy probe with a system message (French prose and
code) and the greedy load bench at 8 clients each at least 2 % faster than the mean of the two default arms, outside the
gap between those arms, and greedy texts identical.

The first lot, one load pass per arm, gave +2.05 % at 8 clients under a 2.13 % gap between A and A': not a result. The
load bench opens each load prompt with a random identifier, so the arms did not send the same prompts. The confirmation
lot played the 8-client load three times in a row per arm, with Python's generator seeded per pass (the same seed in the
three arms, another one for each pass), and compared the medians.

| measure | default (A) | screened head (B) | default (A') |
|---|---|---|---|
| load bench, 8 clients, median of three passes, tok/s | 134.7 | 137.8 | 135.7 |
| the three passes, tok/s | 134.5, 134.7, 134.7 | 137.8, 137.2, 138.4 | 132.2, 136.6, 135.7 |
| probe with a system message, French prose, tok/s | 43.21 | 45.87 | 43.16 |
| probe with a system message, code, tok/s | 55.37 | 58.68 | 55.31 |
| KV pool, tokens | 1 265 468 | 1 259 227 | 1 259 227 |

At one client the probe gained 6.0 to 6.2 % on all five prompts, under a 0.1 to 0.5 % gap, with the same texts and the
same acceptance by position. At 8 clients the gain was out of the noise for the first time (1.9 % against a 0.7 % gap
between medians), but under the 2 % set before. Nothing regressed. Not kept; worth measuring again once the pull request
is merged, when it no longer needs an overlay.

## Internal prefill checkpoints (2026-10-20, 21:11 to 22:08)

vllm-project/vllm#60050 and #61601, both merged into `main` after v0.32.0, let the FlashInfer GDN prefill and the PLE
short convolution export their state at block boundaries inside a prefill, so a prompt that crosses its last cacheable
boundary no longer needs a split prefill. We applied both to v0.32.0 (eight files, no conflict). In `align` mode the
Mamba block then becomes a multiple of the GDN checkpoint stride (64 tokens): 1 664 tokens instead of 1 648. Criterion
written before the measure: the summed first-token time of the five replayed turns of two agent sessions at least 10 %
lower, cached outputs identical to cold outputs, no decoding loss.

| measure | default (A) | internal checkpoints (B) | default (A') |
|---|---|---|---|
| first token, five replayed turns summed, s | 27.849 | 26.766 | 27.655 |
| first session, turn 2: first token, s | 0.520 | 0.436 | 0.498 |
| second session, turn 3: first token, s | 0.769 | 0.447 | 0.767 |
| `bench/prefix_cache.py`: cold output identical to the cached outputs | yes | no | yes |
| prefill, 23 406 tokens, tok/s | 2 793.6 | 2 879.7 | 2 801.8 |
| KV pool, tokens | 1 262 348 | 1 344 168 | 1 282 633 |

The warm turns came 14 to 42 % sooner, but the cold first turns, which dominate the sum, only 1 to 2 %: the sum fell
3.6 %. And in B the cold output of `bench/prefix_cache.py` differed from the two cached outputs, and so did the last
turn of one replayed session, sent cold then served by the cache, while A and A' passed everywhere. The greedy probe
texts were identical in the three arms, and the decoding level. The cause of the cached against cold difference was
not isolated. Not kept.

## Fewer draft tokens under load (2026-10-20, 17:31 to 18:29)

vLLM v0.32.0 can pick the number of draft tokens from the batch size (`num_speculative_tokens_per_batch_size` in
`--speculative-config`). We tried the descending form: two draft tokens up to 8 running requests, as served, and one
from 9 to 16, where verifying fewer drafted tokens might pay. Criterion written before the measure: 16 clients at least
5 % faster, or one stream at least 3 %, with no loss elsewhere beyond the A-A' gap.

| measure | default (A) | one draft above 8 (B) | default (A') |
|---|---|---|---|
| load bench, 16 clients, tok/s | 201.0 | 181.8 | 197.1 |
| sampled load (T 1.0, top-p 0.95, top-k 20), 16 clients, tok/s | 179.9 | 170.3 | 179.0 |
| load bench, 8 clients, tok/s | 136.9 | 134.8 | 135.3 |
| probe, French prose with a system message, tok/s | 43.29 | 43.34 | 43.26 |
| probe, code with a system message, tok/s | 55.46 | 55.52 | 55.41 |

At one client the schedule keeps two draft tokens, and the probe gave the same texts and the same speed. At 16 clients
one draft token cost 8.7 % greedy (A-A' gap 2.0 %) and 5.1 % sampled (gap 0.5 %): on this GB10, two draft tokens still
pay at 16 clients. Not kept.

## CUDA graphs for decoding only (2026-10-20, 16:37 to 17:30)

The served configuration captures full CUDA graphs for decoding steps and piecewise graphs for mixed steps
(`FULL_AND_PIECEWISE`). `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` keeps only the full ones, and a
step that mixes a prefill with decodes then runs without a graph. Criterion written before the measure: the KV pool at
least 5 % larger, no decoding loss beyond the A-A' gap.

| measure | default (A) | decode-only graphs (B) | default (A') |
|---|---|---|---|
| KV pool, tokens | 1 274 831 | 1 316 961 | 1 267 029 |
| probe, French prose with a system message, tok/s | 43.24 | 41.89 | 43.30 |
| probe, code with a system message, tok/s | 55.42 | 55.21 | 55.47 |
| load bench, 1 client, tok/s | 40.8 | 38.9 | 41.2 |
| load bench, 16 clients, tok/s | 192.4 | 195.9 | 198.0 |
| prefill, 23 406 tokens, tok/s | 2 771.6 | 2 781.1 | 2 790.3 |

The graphs took 0.34 GiB in B, against 0.92 and 1.12 GiB on two default starts of the same day, and the pool grew
3.6 % (A-A' gap 0.6 %), under the 5 % set before the measure. Decoding lost 3.2 % on the French prose probe with a
system message (its greedy text changed; the other 8 generations were identical) and 5.1 % at one client on the load
bench (gap 1.0 %). The prefill was level, and the first completion came 12.1 minutes after the restart order, as for the
default. Not kept.

## MoE experts on Marlin (2026-10-20, 15:38 to 16:36)

`--moe-backend marlin` sends every routed-expert layer to Marlin: the NVFP4 experts of the 48 target layers, which the
default runs on FlashInfer's CUTLASS backend, and the FP8 experts of the MTP layer, which the default runs on Triton.
Marlin runs both weight-only (FP4 and FP8 weights, BF16 activations, as its log says). Criterion written before the
measure: one stream at least 2 % faster (probe with a system message, French prose and code), 16 clients and the
prefill not down beyond the A-A' gap.

| measure | default (A) | Marlin (B) | default (A') |
|---|---|---|---|
| probe, code with a system message, tok/s | 55.60 | 57.24 | 55.42 |
| probe, French prose with a system message, tok/s | 43.38 | 41.88 | 43.26 |
| probe, French prose without a system message, tok/s | 41.25 | 43.12 | 41.11 |
| load bench, 1 client, tok/s | 39.0 | 43.9 | 40.4 |
| load bench, 16 clients, tok/s | 194.1 | 203.7 | 196.1 |
| sampled load (T 1.0, top-p 0.95, top-k 20), 16 clients, tok/s | 181.3 | 187.3 | 182.4 |
| prefill, 8 023 tokens, tok/s | 2 762.5 | 2 605.1 | 2 644.4 |
| prefill, 23 406 tokens, tok/s | 2 789.5 | 2 656.9 | 2 766.8 |
| KV pool, tokens | 1 277 952 | 1 332 565 | 1 274 831 |

Marlin was faster under load (+4.4 % at 16 greedy clients, A-A' gap 1.0 %) and on the code probe (+3.1 %), and left a
4.4 % larger KV pool. But the probe in French prose with a system message fell 3.3 % (gap 0.3 %), the 23 406-token
prefill fell 4.4 % (gap 0.8 %), and the greedy probe texts of B differed from A in all 10 generations (A and A'
identical), which would have sent it through the quality gates. Not kept.

## MTP index sharing (2026-10-20, 14:03 to 14:56)

`"index_share_for_mtp_iteration": true` in `--speculative-config` makes the second draft step reuse the sparse (QSA)
top-k indices of the first one instead of running the indexer again; the target still verifies every drafted token.
Criterion written before the measure: the acceptance at position 2 down by less than 2 % relative on every probe prompt,
and the decoding after a 100 000-token prompt at least 3 % faster.

| measure | default (A) | index sharing (B) | default (A') |
|---|---|---|---|
| probe, English prose with a system message: acceptance at position 2 | 0.456 | 0.420 | 0.456 |
| probe, English prose with a system message, tok/s | 44.82 | 43.02 | 44.75 |
| probe, French prose with a system message, tok/s | 43.38 | 43.45 | 43.37 |
| probe, code with a system message, tok/s | 55.61 | 56.59 | 55.47 |
| probe, French prose without a system message, tok/s | 41.25 | 43.76 | 41.24 |

The probe is deterministic here (A and A' gave the same texts and the same acceptance), so the 7.9 % loss at position 2
on English prose with a system message would come back; the greedy texts of B differed from A on 4 of the 5 prompts.
The decoding after a 100 000-token prompt measured +2.7 %, but it is no evidence: our measure put a random marker at the
head of the prompt in each arm, so the three arms decoded three different prompts, and the two default arms differed by
6.7 %. The marker is now fixed for the three arms of an experiment. Not kept.

## FlashInfer autotuning (2026-10-20, 12:56 to 13:55)

The served command turns FlashInfer's autotuner off (`--no-enable-flashinfer-autotune`). Without that flag vLLM tunes
the tactics of the two GEMMs of FlashInfer's fused MoE (`trtllm::fused_moe`) during the kernel warm-up, on 20 token
buckets each; the first completion came 12.5 minutes after the restart order, as for the default's restart that
followed. Criterion written before the measure: prefill at least 2 % faster at both sizes of the load bench, no
decoding loss beyond the A-A' gap, no out-of-memory error and no invalid GEMM profile in the log, start-up under 20
minutes.

| measure | default (A) | autotuning (B) | default (A') |
|---|---|---|---|
| prefill, 8 023 tokens, tok/s | 2 685.9 | 2 764.6 | 2 716.6 |
| prefill, 23 406 tokens, tok/s | 2 748.4 | 2 765.2 | 2 774.2 |
| probe, French prose with a system message, tok/s | 43.32 | 41.58 | 43.36 |
| probe, code with a system message, tok/s | 55.50 | 54.64 | 55.55 |
| load bench, 16 clients, tok/s | 196.2 | 203.5 | 195.4 |
| sampled load (T 1.0, top-p 0.95, top-k 20), 8 clients, tok/s | 117.6 | 124.4 | 118.9 |
| sampled load, 16 clients, tok/s | 173.1 | 186.5 | 183.5 |

The log of B had no out-of-memory error and no invalid GEMM profile. The prefill gain held at 8 023 tokens only
(+2.3 %, A-A' gap 1.1 %); at 23 406 tokens it was +0.1 %, inside the gap. The greedy probe texts of B differed from A in
all 10 generations (A and A' identical), and the probe with a system message fell 1.6 to 4.1 %, outside a gap of 0.1 %.
Under load B was faster: +3.9 % at 16 greedy clients (gap 0.4 %) and +5.2 % at 8 sampled clients (gap 1.1 %); at 16
sampled clients the +4.6 % stayed inside a 5.8 % gap. Not kept.

## A finer prefix-cache unit (2026-10-20, 11:49 to 12:53)

On this hybrid model a prefix-cache hit lands on a multiple of the scheduler block, 1 648 tokens. `--prefix-match-unit`
hashes the prompt on a finer grid; it must divide the block of every cacheable KV cache group, and 1 648 = 16 x 103, so
16 is the largest unit under 103 (64, read in the code, would be refused at start-up).
`--enable-mamba-shared-prefix-checkpoint` was set with it. Criterion written before the measure: on the warm turns of
two agent sessions captured from our clients and replayed (two turns of about 22 000 tokens, three turns of about
48 000), the tokens served by the cache at least 20 % higher, outputs identical, the KV pool down by less than 3 %.
A-B-A' on the served configuration (`disable_eagle_block_drop`):

| measure | default (A) | unit 16 (B) | default (A') |
|---|---|---|---|
| first session, turn 2: first token, s | 0.499 | 0.403 | 0.498 |
| second session, turn 2: first token, s | 0.600 | 0.425 | 0.598 |
| second session, turn 3: first token, s | 0.776 | 0.315 | 0.767 |
| first session, turn 2: tokens served by the cache | 21 424 | 21 760 | 21 424 |
| second session, turns 2 and 3: tokens served by the cache | 95 584 | 96 848 | 95 584 |
| KV pool, tokens | 1 270 150 | 1 240 502 | 1 252 985 |
| probe, French prose with a system message, tok/s | 43.32 | 41.08 | 43.28 |
| probe, code with a system message, tok/s | 55.69 | 56.19 | 55.43 |

The first turn of each session starts cold in B and A'; in A it was served by the cache of an earlier, abandoned run of
the same replay on the same server, and it is left out of the table. The finer unit serves 1.2 to 1.6 % more tokens on
the warm turns (21 760 tokens is 1 360 units of 16, 21 424 is 13 blocks), far from the 20 % set before the measure, but
the part of the prompt left to prefill shrinks, and the first token of turns 2 and 3 came 19 to 59 % sooner, outside the
A-A' gap of 0.2 to 1.2 %. The outputs of the cache and of a cold run stayed identical within each arm, but the greedy
probe texts of B differed from A in all 10 generations (A and A' identical), with the acceptance at position 2 moving
by -16 % to +17 % depending on the prompt; the cause was not isolated. Read in the v0.32.0 code: with
`disable_eagle_block_drop` the KV cache manager is built without a draft group (`use_eagle` follows the block drop in
`v1/core/sched/scheduler.py`), so the shared-prefix checkpoint stays off and only the unit acted. Not kept.

## A fixed KV cache size (2026-10-20, 10:19 to 11:17)

`--kv-cache-memory-bytes` skips the memory profiling at start-up and fixes the KV pool. Criterion written before the
measure: the same pool on two restarts of the variant, with no out-of-memory error. A-B-B2-A' with 18.33 GiB, the
smallest value the profiler had found that day:

| measure | default (A) | 18.33 GiB (B) | 18.33 GiB, second restart (B2) | default (A') |
|---|---|---|---|---|
| KV pool, tokens | 1 262 348 | 1 228 019 | 1 228 019 | 1 270 150 |
| restart order to first completion, min | 12.1 | 11.9 | 11.9 | 12.1 |

The criterion held, and the greedy probe texts were identical in the three probed arms. But the profiling it skips
costs about 12 s of a 12-minute start, and the fixed value gives up 3.0 % of the pool the profiler finds on a typical
start; the profiled pool varied by 0.6 % between A and A'. Not kept.

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
