#!/usr/bin/env python3
"""Witness of the prefix cache and of greedy determinism on a long prompt (read-only).

Sends three times the same long prompt (the first 18 470 tokens of the frozen French text, prefixed with a fresh
identifier to start cold), temperature 0, reasoning off. Length matters: the Mamba block is 3 184 tokens under an
fp8 KV and, with MTP, the scheduler steps back one block; under two blocks nothing is cacheable and the witness
would be red on a sound cache. GREEN if vllm:prefix_cache_hits_total rises, if both repeated requests are faster
than the first, and if the three outputs are identical.
Run it at rest: a third-party request changes the batches and can change a greedy output.
  python3 -I bench/prefix_cache.py <label>         writes results/prefix-cache-<label>.json
"""
import os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402

CHARS = 63575  # prefix of long-fr-verne.txt ending on a paragraph break: 18 470 tokens of text


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
    model = c.served_model()
    busy = c.running()[0]
    if busy:
        print(f"WARNING: {busy:.0f} request(s) running, the greedy output may diverge for that reason")
    text = c.prompts("long-fr-verne.txt")[:CHARS]
    prompt = (f"Identifiant {random.random():.12f}.\n\n{text}\n\n"
              "Resume en trois phrases precises ce document, en citant deux details.")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 120,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    res = c.header("prefix-cache", label, prompt_chars=CHARS, repeats=3, max_tokens=120, temperature=0)
    before = c.metric_sums(("vllm:prefix_cache_hits",)).get("vllm:prefix_cache_hits_total", 0)
    passes = []
    for i in range(3):
        t = time.time()
        d = c.call_json("/v1/chat/completions", body)
        passes.append((time.time() - t, d["usage"]["prompt_tokens"], d["choices"][0]["message"]["content"]))
        print(f"  pass {i}: {passes[-1][0]:.1f} s, {passes[-1][1]} prompt tokens")
    hits = c.metric_sums(("vllm:prefix_cache_hits",)).get("vllm:prefix_cache_hits_total", 0) - before
    cases = [
        (f"prefix cache served ({hits:.0f} tokens hit)", hits > 0),
        ("repeats faster than the cold pass (under 0.7 times)", max(passes[1][0], passes[2][0]) < 0.7 * passes[0][0]),
        ("the two cached outputs identical (passes 1 and 2)", passes[1][2] == passes[2][2]),
        ("cold output identical to the cached outputs (pass 0)", passes[0][2] == passes[1][2]),
    ]
    for name, ok in cases:
        print(("  OK    " if ok else "  FAIL  ") + name)
    res.update(passes=[{"seconds": round(p[0], 2), "prompt_tokens": p[1], "output": p[2]} for p in passes],
               hit_tokens=hits, cases=[{"name": n, "ok": ok} for n, ok in cases],
               verdict="green" if all(ok for _, ok in cases) else "red", requests={"sent": 3, "errors": 0})
    c.write_result(res, f"prefix-cache-{label}.json")
    return 0 if res["verdict"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
