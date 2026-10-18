# Method

How the figures of this repository are measured, and what makes two figures comparable.

## The machine

| item | value |
|---|---|
| machine | one NVIDIA DGX Spark (GB10, 128 GB unified memory, 20 cores: ten Cortex-X925 at 3.9 GHz, ten Cortex-A725) |
| system | DGX OS, kernel 7.0.0-1019-nvidia, driver 580.178.04, Docker |
| server | the container of `serve/serve.sh`, read back by `serve/verify.sh` (25 settings) |
| client | another machine of the same local network for the benches; on the Spark itself for llama-benchy |

The server we measure is shared with other users. Every pass is played at rest and checked afterwards.

## A clean pass

`bench/idle_pass.py` wraps every bench:

1. it waits until no request has run or waited for 30 s;
2. it runs the bench and samples `/metrics` every 5 s;
3. it compares the requests finished by the engine (`vllm:request_success_total`) with the requests the bench sent.

A pass is clean when both counts are equal, when no more requests ran at once than the bench itself sends, and
when none waited. A contaminated pass is discarded and replayed, never averaged. A pass that sent nothing, or got an
HTTP error, is never green. The verdict sits next to the result, in `results/<name>.pass.json`.

We also read the server's access log before each pass, to know who else was there; that tool is ours and not part
of this repository.

## First pass and second pass

The per-layer embedding table (51 GB) is read from the NVMe as decoding needs its rows. The probe
(`bench/probe.py`) plays each prompt twice:

| pass | what it measures |
|---|---|
| 1 | decoding of new text: rows come from the disk (tens of MB read per answer) |
| 2 | the same text again: rows come from the page cache (0 MB read) |

The reference figures are pass 2, greedy, 600 tokens, reasoning off. Pass 1 is reported next to it.

## What is comparable

- A setting is judged A-B-A: reference, change, reference again, the same day, by the same commands. A gap between
  A and B smaller than the gap between the two A arms is not a result.
- The load bench is judged on two passes: the first bench of a series has come out low more than once.
- A prefill is only comparable with a prefill taken in the same state of the page cache. Long sampled loads fill the
  cache with unrelated table rows and lower the next prefills.
- A prompt is frozen in `bench/prompts/`. A prompt cut out of a document that keeps being edited moves with each edit.
- Greedy outputs are identical from one pass to the next on this recipe (deterministic top-k, `bench/prefix_cache.py`):
  a different text at temperature 0 is a finding, not noise.
- Figures from other people are quoted with their protocol and never put in the same column as ours.

## Result files

Each bench writes `results/<kind>-<label>.json` with:

| field | content |
|---|---|
| `measured_at`, `finished_at` | local time with offset |
| `repo_commit` | commit of this repository that ran the bench (`-dirty` if the tree had changes) |
| `served_model`, `protocol` | what was asked, with which settings |
| `requests` | requests sent and in error, read by `idle_pass.py` |

No file carries an address, a key or a host name.

## Draft vocabulary

The reduced draft vocabulary only changes which tokens the drafter may propose; the target verifies every token.
It changes acceptance and speed, never a greedy output. This repository builds its set from a frozen public corpus
(`patches/draft-vocab/`); our own server uses a set built on private texts. Both are measured on the same server,
with the same commands, and reported side by side.
