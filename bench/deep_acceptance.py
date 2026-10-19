#!/usr/bin/env python3
"""Long context and MTP acceptance by position past 262 144 tokens (read-only for the server).

For each target in REAL prompt tokens (default 200 000, 280 000, 450 000): the needle and the neutral filler of
bench/reasoning_bench.py, sized through /tokenize (no engine request) to within 0.2 %, then two greedy requests with
reasoning off: reasoning item r01 (full prefill) and the needle question (same prefix: the prefix cache can serve it).
Around each request, the speculative-decoding counters of /metrics give the acceptance by position of that request.
When SPARK_SSH is set, the kernel journal of the Spark is read after each target for Xid lines; the number of lines
read is reported, so that an unreadable journal is never taken for a clean one.

Why real tokens: reasoning_bench.py sizes its filler at 3.6 characters per token, so its "--depths 510000" read
449 365 tokens, and a depth of 280 000 would stay under 262 144. On vLLM v0.32.0 the MTP drafter's rotary cache stops at
262 144 rows unless patches/mrope-draft-floor is mounted (vllm-project/vllm#62383).

  python3 -I bench/deep_acceptance.py <label> [--targets 200000,280000,450000]   writes results/deep-acceptance-<label>.json
"""
import argparse, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
import reasoning_bench as rb  # noqa: E402

SPEC = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens", "vllm:request_success")


def kernel_xid(since):
    """(lines read, Xid lines) of the Spark's kernel journal since `since`; None when SPARK_SSH is unset or fails."""
    host, key = os.environ.get("SPARK_SSH"), os.environ.get("SPARK_SSH_KEY")
    if not host:
        return None
    cmd = ["ssh", "-n", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"] + (["-i", key] if key else []) + \
          [host, f"journalctl -k --no-pager --since '@{int(since)}' -o short-iso; echo END-OF-JOURNAL"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or "END-OF-JOURNAL" not in r.stdout:
        return None
    lines = [l for l in r.stdout.splitlines() if l and l != "END-OF-JOURNAL"]
    return {"lines_read": len(lines), "xid": [l[:300] for l in lines if re.search(r"\bxid\b", l, re.I)]}


def tokens(model, content):
    return len(c.call_json("/tokenize", {"model": model, "messages": [{"role": "user", "content": content}],
                                         "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["tokens"])


def prefix_for(model, target, question):
    """Needle and filler sized so that the rendered prompt has `target` tokens, to within 0.2 %."""
    approx = target
    for _ in range(8):
        pre = rb.NEEDLE + "\n\n" + rb.filler(approx)
        n = tokens(model, pre + "\n\n" + question)
        if abs(n - target) <= target * 0.002:
            break
        approx = int(approx * target / n)
    return pre, n


def acceptance(m0, m1):
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    pos = [round(d.get(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{q}]", 0) / drafts, 3)
           for q in range(4) if drafts and f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{q}]" in d]
    return {"drafts": round(drafts), "by_position": pos, "finished_requests": round(d.get("vllm:request_success_total", 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--targets", default="200000,280000,450000")
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    model = c.served_model()
    item = [it for it in rb.ITEMS if it[0] == "r01"][0]
    res = c.header("deep-acceptance", a.label, targets_real_tokens=a.targets, item="r01", needle=True, temperature=0,
                   reasoning="off", max_tokens=a.max_tokens)
    res["targets"] = []
    sent = errors = 0
    for target in [int(x) for x in a.targets.split(",")]:
        t0 = time.time()
        pre, n = prefix_for(model, target, item[1])
        row = {"target": target, "item_prompt_tokens": n}
        for name, question, expected in (("item", item[1], item[2]), ("needle", rb.NEEDLE_Q, rb.NEEDLE_ANS)):
            m0 = c.metric_sums(SPEC)
            try:
                out, mes = rb.ask(model, pre + "\n\n" + question, False, a.max_tokens, 1800, "off"); sent += 1
                row[name] = {"correct": rb.grade(out, expected), **mes, "acceptance": acceptance(m0, c.metric_sums(SPEC))}
            except Exception as e:  # noqa: BLE001
                sent += 1; errors += 1
                row[name] = {"error": str(e)[:300]}
            print(target, name, {k: v for k, v in row[name].items() if k != "text"}, flush=True)
        row["kernel_journal"] = kernel_xid(t0 - 5)
        res["targets"].append(row)
    res["requests"] = {"sent": sent, "errors": errors}
    c.write_result(res, f"deep-acceptance-{a.label}.json")


if __name__ == "__main__":
    main()
