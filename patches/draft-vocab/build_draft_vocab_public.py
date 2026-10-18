#!/usr/bin/env python3
"""Build the reduced draft vocabulary of the MTP drafter from the frozen public corpus.

Same rule as build_draft_vocab.py (blazux, kept unchanged beside this file), with one addition:
  1. special and added tokens (chat template, tool-call markers, <think>), always;
  2. bytes 0-255, always;
  3. corpus tokens, by frequency (French and English Wikipedia prose, Python, TypeScript, Rust, shell);
  4. added: remaining slots filled from a complement set (blazux's published set, English and generic code),
     by DECREASING id: its low ids are already a fill by order, its high ids are the ones its corpus chose
     (by increasing id, the result equalled a plain fill by order);
  5. then the lowest ids not yet chosen.
The target verifies every drafted token: the set changes the acceptance rate, and it can also change a
greedy text (docs/method.md).

Usage (container --network none, a vLLM image that carries transformers and numpy):
  build_draft_vocab_public.py <tokenizer_dir> <out.npy> --corpus <dir> [--complement ids.npy] [--n 65536]
Reads .txt, .md and .json files of the corpus directory.
"""
import argparse, collections, os, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer_dir"); ap.add_argument("out")
ap.add_argument("--corpus", required=True)
ap.add_argument("--complement")
ap.add_argument("--n", type=int, default=65536)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
V = len(tok)

counts = collections.Counter()
nchars = 0
for root, _, files in os.walk(a.corpus):
    for f in sorted(files):
        if f.rsplit(".", 1)[-1] not in ("txt", "md", "json"):
            continue
        t = open(os.path.join(root, f), "rb").read().decode("utf-8", errors="ignore")
        nchars += len(t)
        for i in range(0, len(t), 200_000):
            counts.update(tok.encode(t[i:i + 200_000], add_special_tokens=False))
print(f"corpus: {nchars / 1e6:.1f} M characters, {sum(counts.values()) / 1e6:.2f} M tokens, {len(counts)} distinct ids")

keep = set(tok.all_special_ids) | set(int(i) for i in tok.added_tokens_decoder.keys())
n_spec = len(keep)
keep |= set(range(min(256, V)))
for tid, _ in counts.most_common():
    if len(keep) >= a.n:
        break
    keep.add(int(tid))
n_corpus = len(keep)
if a.complement:
    for tid in sorted((int(i) for i in np.load(a.complement, allow_pickle=False)), reverse=True):
        if len(keep) >= a.n:
            break
        if 0 <= tid < V:
            keep.add(tid)
n_compl = len(keep)
i = 0
while len(keep) < a.n and i < V:
    keep.add(i); i += 1
ids = np.array(sorted(keep), dtype=np.int32)
np.save(a.out, ids)
print(f"{len(ids)} ids: {n_spec} special, corpus up to {n_corpus}, complement up to {n_compl}, "
      f"filled by order up to {len(ids)}")
sys.exit(0 if len(ids) == a.n else 1)
