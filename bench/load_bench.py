#!/usr/bin/env python3
"""Prefill and load bench (OpenAI API of vLLM, read-only).

1. Cold prefill: two prefixes of the frozen French text (7 980 and 23 363 tokens of text), each prefixed with a
   fresh identifier so that no prefix-cache entry serves; three passes each, median; prefill = prompt tokens /
   time to first token.
2. Load: 1, 4, 8 then 16 simultaneous clients (--clients), distinct prompts (French prose and code alternating),
   400 tokens each, temperature 0, reasoning off; aggregate rate = tokens produced / batch duration, and per-stream
   rate.

A prefill is only comparable with a prefill taken in the same state of the page cache (docs/method.md).
Refuses to start while a request is running.
  python3 -I bench/load_bench.py <label> [--clients 1,4,8,16]      writes results/load-<label>.json
"""
import argparse, concurrent.futures, json, os, random, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402


def stream(model, prompt, max_tokens):
    """One streamed request: (time to first token, total duration, prompt tokens, completion tokens)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(c.URL + "/v1/chat/completions", json.dumps(body).encode(), c.HEADERS)
    t0 = time.time(); ttft = None; usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if ttft is None and any((ch.get("delta") or {}).get("content") for ch in d.get("choices", [])):
                ttft = time.time() - t0
    return ttft, time.time() - t0, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--clients", default="1,4,8,16")
    ap.add_argument("--no-prefill", action="store_true")
    a = ap.parse_args()
    p = c.prompts("load.json")
    text = c.prompts("long-fr-verne.txt")
    clients = [int(x) for x in a.clients.split(",") if x.strip()]
    if sum(c.running()):
        sys.exit("a request is already running: bench refused (run it at rest)")
    model = c.served_model()
    res = c.header("load", a.label, prefill_chars=p["prefill_chars"], prefill_passes=3, clients=clients,
                   max_tokens=400, temperature=0, reasoning="off")
    sent = 0
    res["prefill"] = []
    for chars in ([] if a.no_prefill else p["prefill_chars"]):
        runs = []
        for _ in range(3):  # a fresh identifier each pass: no cached prefix
            prompt = f"Identifiant {random.random():.12f}.\n\n{text[:chars]}\n\n{p['prefill_question']}"
            ttft, total, n_in, n_out = stream(model, prompt, 32); sent += 1
            runs.append((n_in / ttft, ttft, n_in))
        med = sorted(runs)[1]
        row = {"prompt_tokens": med[2], "first_token_s": round(med[1], 2), "prefill_tok_s": round(med[0], 1),
               "passes_tok_s": [round(r[0], 1) for r in runs]}
        res["prefill"].append(row)
        print(f"prefill: {med[2]} tokens, median {med[0]:.0f} tok/s (first token {med[1]:.2f} s), passes {row['passes_tok_s']}")
    res["load"] = []
    subjects = p["load_subjects"]
    for n in clients:
        prompts = [f"[{random.random():.9f}] {subjects[i % len(subjects)]}" for i in range(n)]
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            done = list(ex.map(lambda q: stream(model, q, 400), prompts))
        sent += n
        dur = time.time() - t0
        produced = sum(r[3] for r in done)
        per = [r[3] / (r[1] - r[0]) for r in done if r[0] and r[1] > r[0]]
        row = {"clients": n, "tokens": produced, "duration_s": round(dur, 2), "aggregate_tok_s": round(produced / dur, 1),
               "per_stream_tok_s": round(sum(per) / len(per), 1) if per else None,
               "mean_first_token_s": round(sum(r[0] for r in done) / n, 2)}
        res["load"].append(row)
        print(f"{n} client(s): {row['aggregate_tok_s']} tok/s aggregate, {row['per_stream_tok_s']} per stream, "
              f"first token {row['mean_first_token_s']} s")
    res["requests"] = {"sent": sent, "errors": 0}
    c.write_result(res, f"load-{a.label}.json")


if __name__ == "__main__":
    main()
