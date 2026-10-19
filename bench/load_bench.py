#!/usr/bin/env python3
"""Prefill and load bench (OpenAI API of vLLM, read-only).

1. Cold prefill: two prefixes of the frozen French text (7 980 and 23 363 tokens of text), each prefixed with a
   fresh identifier so that no prefix-cache entry serves; three passes each, median; prefill = prompt tokens /
   time to first token.
2. Load: 1, 4, 8 then 16 simultaneous clients (--clients), distinct prompts (French prose and code alternating),
   400 tokens each, temperature 0, reasoning off; aggregate rate = tokens produced / batch duration, and per-stream
   rate; MTP acceptance by position over each batch, from /metrics.

Sampling (greedy by default): --temperature, --top-p, --top-k, --presence-penalty set it for every request; --seed N
gives request i the seed N + i, so a sampled pass can be replayed. Our clients send none of these and get the
checkpoint's generation_config (temperature 1.0, top_p 0.95, top_k 20). --self-check tests the request bodies offline.

A prefill is only comparable with a prefill taken in the same state of the page cache (docs/method.md).
Refuses to start while a request is running.
  python3 -I bench/load_bench.py <label> [--clients 1,4,8,16] [--no-prefill] [--temperature T --top-p P --top-k K
                                  --presence-penalty X --seed N]      writes results/load-<label>.json
  python3 -I bench/load_bench.py --self-check
"""
import argparse, concurrent.futures, json, os, random, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402

SAMPLING = ("temperature", "top_p", "top_k", "presence_penalty")
SPEC = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens")


def body_for(model, prompt, max_tokens, sampling=None, seed=None):
    """Request body: greedy unless sampling (a dict over SAMPLING) says otherwise."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update({k: v for k, v in (sampling or {}).items() if v is not None})
    if seed is not None:
        body["seed"] = seed
    return body


def acceptance(m0, m1):
    """Accepted draft tokens per draft, by position, between two /metrics readings."""
    drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    if not drafts:
        return None
    return [round((m1.get(k, 0) - m0.get(k, 0)) / drafts, 3) for k in sorted(m1)
            if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total[")]


def self_check():
    greedy = body_for("m", "p", 400)
    assert greedy["temperature"] == 0 and not any(k in greedy for k in SAMPLING[1:]) and "seed" not in greedy
    s = body_for("m", "p", 400, {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}, seed=7)
    assert (s["temperature"], s["top_p"], s["top_k"], s["presence_penalty"], s["seed"]) == (0.7, 0.8, 20, 1.5, 7)
    t = body_for("m", "p", 400, {"temperature": 1.0, "top_p": None, "top_k": None, "presence_penalty": None})
    assert t["temperature"] == 1.0 and not any(k in t for k in SAMPLING[1:])
    m0 = {"vllm:spec_decode_num_drafts_total": 10, "vllm:spec_decode_num_accepted_tokens_per_pos_total[0]": 5,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total[1]": 2}
    m1 = {"vllm:spec_decode_num_drafts_total": 110, "vllm:spec_decode_num_accepted_tokens_per_pos_total[0]": 75,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total[1]": 42}
    assert acceptance(m0, m1) == [0.7, 0.4] and acceptance(m0, m0) is None
    p = c.prompts("load.json")
    assert p["prefill_chars"] and p["load_subjects"] and c.prompts("long-fr-verne.txt")
    print("load_bench self-check: request bodies and acceptance as expected")


def stream(model, prompt, max_tokens, sampling=None, seed=None):
    """One streamed request: (time to first token, total duration, prompt tokens, completion tokens)."""
    body = body_for(model, prompt, max_tokens, sampling, seed)
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
    ap.add_argument("label", nargs="?")
    ap.add_argument("--clients", default="1,4,8,16")
    ap.add_argument("--no-prefill", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--presence-penalty", type=float)
    ap.add_argument("--seed", type=int, help="request i gets seed N + i")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if not a.label:
        ap.error("a label is required")
    sampling = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "presence_penalty": a.presence_penalty}
    seq = [0]

    def seed():
        if a.seed is None:
            return None
        seq[0] += 1
        return a.seed + seq[0] - 1
    p = c.prompts("load.json")
    text = c.prompts("long-fr-verne.txt")
    clients = [int(x) for x in a.clients.split(",") if x.strip()]
    if sum(c.running()):
        sys.exit("a request is already running: bench refused (run it at rest)")
    model = c.served_model()
    res = c.header("load", a.label, prefill_chars=p["prefill_chars"], prefill_passes=3, clients=clients,
                   max_tokens=400, reasoning="off", seed=a.seed, **sampling)
    sent = 0
    res["prefill"] = []
    for chars in ([] if a.no_prefill else p["prefill_chars"]):
        runs = []
        for _ in range(3):  # a fresh identifier each pass: no cached prefix
            prompt = f"Identifiant {random.random():.12f}.\n\n{text[:chars]}\n\n{p['prefill_question']}"
            ttft, total, n_in, n_out = stream(model, prompt, 32, sampling, seed()); sent += 1
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
        seeds = [seed() for _ in prompts]
        m0 = c.metric_sums(SPEC)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            done = list(ex.map(lambda q: stream(model, q[0], 400, sampling, q[1]), zip(prompts, seeds)))
        sent += n
        dur = time.time() - t0
        acc = acceptance(m0, c.metric_sums(SPEC))
        produced = sum(r[3] for r in done)
        per = [r[3] / (r[1] - r[0]) for r in done if r[0] and r[1] > r[0]]
        row = {"clients": n, "tokens": produced, "duration_s": round(dur, 2), "aggregate_tok_s": round(produced / dur, 1),
               "per_stream_tok_s": round(sum(per) / len(per), 1) if per else None,
               "mean_first_token_s": round(sum(r[0] for r in done) / n, 2), "acceptance_by_position": acc}
        res["load"].append(row)
        print(f"{n} client(s): {row['aggregate_tok_s']} tok/s aggregate, {row['per_stream_tok_s']} per stream, "
              f"first token {row['mean_first_token_s']} s, acceptance {acc}")
    res["requests"] = {"sent": sent, "errors": 0}
    c.write_result(res, f"load-{a.label}.json")


if __name__ == "__main__":
    main()
