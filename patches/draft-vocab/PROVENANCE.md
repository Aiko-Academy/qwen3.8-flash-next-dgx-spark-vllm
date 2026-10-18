# Provenance: reduced draft vocabulary of the MTP drafter

The MTP drafter scores the 248 320 rows of the output head at every draft step, and almost none of them is ever
its argmax. With this hook, it only scores 65 536 rows chosen in advance (a 320 MiB private copy); the others are
`-inf`. The target verifies every drafted token, so the set changes the acceptance rate, and it can also change a
greedy text (measured on 2026-10-18, `docs/method.md`).

## Sources

| item | value |
|---|---|
| blazux | https://github.com/blazux/qwen3.8-Flash-DGX, commit `7d0798e8c3aaacee8c72afb302aa4247fe4277a7` (2026-10-09), Apache-2.0 (`LICENSE-blazux`, abbreviated; full text in `../../LICENSES/blazux-Apache-2.0.txt`) |
| retrieved | 2026-10-12, `raw.githubusercontent.com` at the commit, files only |
| public corpus | `public-corpus.tsv`: 213 Wikipedia featured articles at fixed revisions (123 French, 90 English), 738 files of vllm-project/vllm (tag v0.31.0), microsoft/TypeScript (tag v5.9.3) and tokio-rs/tokio (tag tokio-1.47.1) at fixed commits, each with the sha256 of its bytes; frozen on 2026-10-18 by `../../tools/freeze_public_corpus.py` |

## Files

| file | origin | role |
|---|---|---|
| `patch_mtp_draft_vocab.py` | blazux, `src/`, verbatim | appends the hook to `mtp.py`; played once, here, to make the mounted file |
| `vllm-overlay/models/qwen4_exp/nvidia/mtp.py` | `mtp.py` of `vllm/vllm-openai:v0.32.0` plus the hook, nothing else (replayed without fuzz) | under `VLLM_MTP_DRAFT_VOCAB=<ids.npy>`, replaces `Qwen4ExpMTP.compute_logits`. Kept for reference: `serve.sh` mounts `../draft-nvfp4-head/`, which starts with this file |
| `build_draft_vocab.py` | blazux, `tools/`, verbatim | the original builder, kept for reference |
| `draft_vocab_65536.npy` | blazux, `src/`, verbatim | blazux's published set (English and code); used here as the complement of the public set |
| `build_draft_vocab_public.py` | Aiko Academy, Apache-2.0 | rule of `build_draft_vocab.py` plus a complement: remaining slots filled from blazux's set by decreasing id |
| `fetch_public_corpus.py` | Aiko Academy, Apache-2.0 | downloads `public-corpus.tsv`, refuses any byte that differs, cleans wikitext |
| `public-corpus.tsv` | Aiko Academy | the frozen corpus manifest |
| `temoin_crochet.py` | Aiko Academy, Apache-2.0 | CPU witness of the hook in the image (French messages): hook active, logits of the set equal to the full logits, the others at `-inf` |
| `LICENSE-blazux` | blazux | licence, as published |

## The set we serve and the set we publish

Our own service uses a set built on our private texts; it stays private. This repository builds its set from the
public corpus above (`serve/prepare.sh draft-vocab`, output `build/draft_vocab_public_65536.npy`, sha256
`19a4415d6146a0fc91355674a11d9929d0e726da56656096a9a52c3d168368a8`, rebuilt to the byte on 2026-10-18 and checked by
`prepare.sh`). The difference
between the two (coverage of the benchmark outputs, acceptance, throughput) is measured and published in
`results/` and in the README.

The Wikipedia text is licensed CC BY-SA 4.0 and is not redistributed: only the manifest (revision ids and hashes)
is, and the `.npy` holds token ids only.

## Review (2026-10-12)

- No network, no `subprocess`, no `eval` or `exec` in the two third-party Python files. The hook only reads the
  `.npy` named by the variable (`np.load` without pickle: int32 only).
- Replaced function (unchanged from v0.31.0 to v0.32.0): `return self.logits_processor(self.lm_head, hidden_states)`, with
  `LogitsProcessor(config.vocab_size)` (scale 1, no soft cap), TP 1: same logits, restricted to the set.
- The drafter samples greedily (`draft_sample_method = "greedy"`, default, read back on the serving container):
  the set only changes the drafter's argmax. Greedy texts were identical when the set first went into service
  (2026-10-12); with the public set, French and English prose diverged, code did not (2026-10-18).
- The 320 MiB copy is created at the first call, after the KV pool is sized: outside the memory budget.
