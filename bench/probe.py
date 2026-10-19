#!/usr/bin/env python3
"""Greedy single-stream probe: one generation at a time, at rest, with what the server and the machine say.

For each prompt of prompts/probe.json (French prose, English prose, code), N greedy generations of 600 tokens,
reasoning off. Before each one, 20 s of rest (no request running or waiting). Around each one, /metrics (drafts,
accepted tokens per position, steps, finished requests: a third-party request finished during the measure is
flagged) and, when the Spark is reachable (common.py), its major faults, NVMe reads and EngineCore CPU. Then one tool
call (qwen3_xml) that must come back well formed.

Pass 1 of a prompt decodes new text (rows of the PLE table may come from disk); pass 2 decodes the same text again
(rows in the page cache). The reference figures are pass 2.

--system TEXT puts a system message before each generation prompt (none by default; the tool call is unchanged). On
vLLM v0.32.0 the bare code prompt comes back as invented tool calls (README), so a code speed needs it.

  python3 -I bench/probe.py <label> [--passes 2] [--system TEXT]        writes results/probe-<label>.json
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
from toolcall import build_messages  # noqa: E402

COUNTERS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens", "vllm:request_success",
            "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count")


def generate(model, prompt, system=None):
    body = {"model": model, "messages": build_messages(prompt, system), "max_tokens": 600, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(c.URL + "/v1/chat/completions", json.dumps(body).encode(), c.HEADERS)
    first = last = None; n = None; text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                n = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    text.append(d)
                    first = first or time.time()
                    last = time.time()
    return n, ((n - 1) / (last - first) if n and last > first else None), "".join(text)


def tool_call(model, spec):
    body = {"model": model, "temperature": 0, "max_tokens": 200, "tools": [{"type": "function", "function": spec["function"]}],
            "messages": [{"role": "user", "content": spec["user"]}], "chat_template_kwargs": {"enable_thinking": False}}
    msg = c.call_json("/v1/chat/completions", body, timeout=120)["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    ok = (len(tc) == 1 and tc[0]["function"]["name"] == spec["function"]["name"]
          and json.loads(tc[0]["function"]["arguments"]) == spec["expected_argument"])
    return {"well_formed": ok, "tool_calls": tc, "content": msg.get("content")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--only", default="prose_fr,prose_en,code", help="prompts to play, comma-separated")
    ap.add_argument("--system", default=None, help="system message put before each generation prompt (none by default)")
    a = ap.parse_args()
    p = c.prompts("probe.json")
    model = c.served_model()
    res = c.header("probe", a.label, max_tokens=600, temperature=0, reasoning="off", passes=a.passes, rest_s=20,
                   prompts={k: p[k] for k in a.only.split(",")}, system=a.system)
    res["generations"] = []
    sent = errors = 0
    for name in a.only.split(","):
        for i in range(a.passes):
            if not c.wait_idle():
                print(f"{name} {i + 1}: no rest within 10 min, skipped"); continue
            m0, s0 = c.metric_sums(COUNTERS), c.machine()
            try:
                tokens, rate, text = generate(model, p[name], a.system); sent += 1
            except Exception as e:
                errors += 1; print(f"{name} {i + 1}: error {e}"); continue
            m1, s1 = c.metric_sums(COUNTERS), c.machine()
            d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
            drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
            acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            steps = d.get("vllm:inter_token_latency_seconds_count", 0)
            g = {"prompt": name, "pass": i + 1, "tokens": tokens, "tok_s": round(rate, 2) if rate else None,
                 "tokens_per_step": round(1 + acc / drafts, 2) if drafts else None,
                 "acceptance_by_position": [round(d.get(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{q}]", 0) / drafts, 3)
                                            for q in range(4) if drafts and f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{q}]" in d],
                 "mean_itl_ms": round(1000 * d.get("vllm:inter_token_latency_seconds_sum", 0) / steps, 1) if steps else None,
                 "finished_requests": d.get("vllm:request_success_total", 0), "text": text}
            if s0 and s1:
                g.update(major_faults=s1["majflt"] - s0["majflt"],
                         nvme_read_mb=round((s1["sectors_read"] - s0["sectors_read"]) * 512 / 1e6, 1),
                         engine_cpu_s=round((s1["engine_cpu_ticks"] - s0["engine_cpu_ticks"]) / 100, 1),
                         swap_used_mb=round((s0["swap_free_kib"] - s1["swap_free_kib"]) / 1024, 1))
            third = " ; THIRD-PARTY REQUEST" if g["finished_requests"] > 1 else ""
            print(f"{name} {i + 1}: {g['tok_s']} tok/s, {g['tokens_per_step']} tok/step, acceptance "
                  f"{g['acceptance_by_position']}, itl {g['mean_itl_ms']} ms, nvme {g.get('nvme_read_mb')} MB{third}", flush=True)
            res["generations"].append(g)
    c.wait_idle(5)
    try:
        res["tool"] = tool_call(model, p["tool"]); sent += 1
    except Exception as e:
        errors += 1; res["tool"] = {"well_formed": False, "error": str(e)}
    print(f"tool call well formed: {res['tool']['well_formed']}")
    res["requests"] = {"sent": sent, "errors": errors}
    c.write_result(res, f"probe-{a.label}.json")


if __name__ == "__main__":
    main()
