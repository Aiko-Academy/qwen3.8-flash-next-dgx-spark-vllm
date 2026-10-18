#!/usr/bin/env python3
"""Warm the server up after a start, and measure the first long request.

The first request after a start prefilled about 40 % slower than the next one, and the Triton kernels of the
rejection sampler of speculative decoding only compile at the first sampled request (salient-data recipe). This
script sends the requests that trigger those compilations, then compares the prefill of a first long cold prompt
with the next one. Exit code 0 in every case: a warm-up never blocks a start.
  python3 -I bench/warmup.py
"""
import os, sys, time, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402

PARAGRAPH = (
    "Le reglement interbancaire repose sur des regles communes que les banques acceptent avant d'echanger des "
    "paiements. Chaque etablissement designe des correspondants, ouvre des comptes de reglement et se soumet a "
    "des procedures de traitement des files, de gestion des incidents et de cloture de journee. Les montants sont "
    "changes a un cours constate, les instructions sont tracees et conservees, et tout retard est signale aux "
    "autorites de tutelle. La finalite est double : empecher qu'une seule panne bloque l'ensemble du systeme et "
    "garantir qu'un paiement accepte devient definitif et irrevocable pour les deux parties concernees."
)


def prompt(head):
    return "\n".join([head] + [f"Paragraphe {i} : {PARAGRAPH}" for i in range(1, 81)] + ["Resume en une phrase."])


def request(name, content, max_tokens, temperature, thinking=None):
    body = {"model": c.served_model(), "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    try:
        t0 = time.monotonic()
        d = c.call_json("/v1/chat/completions", body)
        dur = time.monotonic() - t0
    except Exception as e:
        print(f"  {name}: FAILED {type(e).__name__}: {e}")
        return None
    if name.startswith("long"):
        n = d.get("usage", {}).get("prompt_tokens", 0)
        print(f"  {name}: {dur:.1f} s, {n} prompt tokens, {n / dur:.0f} tok/s prefill")
        return n / dur
    print(f"  {name}: {dur:.1f} s")
    return None


print("== warm-up ==")
request("greedy short", "Compte de 1 a 20.", 64, 0)
request("sampled short", "Compte de 1 a 20.", 64, 1.0)
r3 = request("long cold", prompt(f"chauffe {uuid.uuid4().hex}"), 1, 0, thinking=False)
r4 = request("long next", prompt(f"chauffe {uuid.uuid4().hex}"), 1, 0, thinking=False)
if r3 and r4:
    print(f"  first long request: {100 * (r3 / r4 - 1):+.0f} % prefill against the next one")
