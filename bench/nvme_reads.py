#!/usr/bin/env python3
"""Witness of the NVMe reads of decoding (the PLE table is read from disk), read-only for the server.

1. Silences (--silences, on the Spark): an O_DIRECT read of 4 KiB in the PLE table after a silence of 0 to 4 s.
   With the kernel's default APST, the disk falls asleep 100 ms after its last read.
2. Load: N simultaneous streams on new text (temperature 1.0, distinct subjects, 300 tokens), counters of the NVMe
   before and after: reads, cumulated read time, mean latency per read. A greedy bench would read the same rows
   again from the page cache; sampling keeps rows new.
A latency per read does not prove a gain: replay the throughput in both settings (docs/negative-results.md, APST).
Needs the Spark's /proc (common.py: local server or SPARK_SSH).
  python3 -I bench/nvme_reads.py <label> [--silences] [--streams 16,1] [--ple <path on the Spark>]
"""
import argparse, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402

PLE = "/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia-fp8hybrid/model-fp8-mtp-ple.safetensors"
SILENCE_PROBE = r'''
import os, mmap, time, random, statistics, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECT)
blocks = os.fstat(fd).st_size // 4096
buf = mmap.mmap(-1, 4096)
def read():
    t = time.perf_counter(); os.preadv(fd, [buf], random.randrange(blocks) * 4096)
    return (time.perf_counter() - t) * 1000
for gap in [0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0, 2.5, 4.0]:
    read()
    v = []
    for _ in range(12 if gap < 1 else 6):
        time.sleep(gap); v.append(read())
    print(f"{gap} {statistics.median(v):.3f} {min(v):.3f} {max(v):.3f}")
'''


def silences(ple):
    out = c.on_spark(f"python3 -I - {ple}", stdin_text=SILENCE_PROBE, timeout=300)
    if out is None:
        sys.exit("silences: the Spark is not reachable (common.py: local server or SPARK_SSH)")
    rows = []
    for line in out.split("\n"):
        if line.strip():
            g, med, lo, hi = (float(x) for x in line.split())
            rows.append({"silence_ms": round(g * 1000), "median_ms": med, "min_ms": lo, "max_ms": hi})
            print(f"  silence {g * 1000:6.0f} ms: median {med:7.3f} ms, min {lo:7.3f}, max {hi:7.3f}")
    return rows


def disk():
    out = c.on_spark(f"awk '$3==\"{c.NVME}\"{{print $4, $7}}' /proc/diskstats")
    if not out:
        sys.exit("disk counters not readable (common.py: local server or SPARK_SSH)")
    a, b = out.split()[:2]
    return int(a), int(b)


def stream(model, subject):
    body = {"model": model, "max_tokens": 300, "temperature": 1.0,
            "messages": [{"role": "user", "content": f"[{random.random():.9f}] Ecris un recit en prose francaise sur "
                          f"{subject}, avec des details concrets et inattendus."}]}
    return c.call_json("/v1/chat/completions", body, timeout=300)["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--silences", action="store_true")
    ap.add_argument("--streams", default="16,1")
    ap.add_argument("--ple", default=PLE)
    a = ap.parse_args()
    subjects = c.prompts("load.json")["story_subjects"]
    model = c.served_model()
    res = c.header("nvme-reads", a.label, streams=a.streams, max_tokens=300, temperature=1.0, device=c.NVME)
    if a.silences:
        res["silences"] = silences(a.ple)
    res["load"] = []
    sent = 0
    for n in [int(x) for x in a.streams.split(",") if x.strip()]:
        chosen = random.sample(subjects, n) if n <= len(subjects) else [random.choice(subjects) for _ in range(n)]
        r0, ms0 = disk(); t0 = time.time()
        with ThreadPoolExecutor(n) as ex:
            tokens = sum(ex.map(lambda s: stream(model, s), chosen))
        dur = time.time() - t0; r1, ms1 = disk(); sent += n
        row = {"streams": n, "tokens": tokens, "duration_s": round(dur, 1), "aggregate_tok_s": round(tokens / dur, 1),
               "reads": r1 - r0, "read_time_ms": ms1 - ms0, "ms_per_read": round((ms1 - ms0) / max(r1 - r0, 1), 2)}
        res["load"].append(row)
        print(f"{n} streams: {row['aggregate_tok_s']} tok/s, {row['reads']} reads, {row['ms_per_read']} ms per read")
    res["requests"] = {"sent": sent, "errors": 0}
    c.write_result(res, f"nvme-reads-{a.label}.json")


if __name__ == "__main__":
    main()
