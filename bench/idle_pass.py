#!/usr/bin/env python3
"""Play ONE measurement pass on a shared server and say whether it is clean (read-only for the server).

1. Wait for rest: no request running or waiting for --rest seconds (at most --max-wait).
2. Run the command and, while it runs, sample /metrics every 5 s (running and waiting requests).
3. Compare the requests FINISHED by the engine during the pass (sum of vllm:request_success_total) with the
   requests the pass sent, read in its result file ("requests": {"sent": n, "errors": m}). Clean if equal, at most
   --max-running requests running at once, none waiting. Another client of the same server shows up in that count.

  python3 -I bench/idle_pass.py --result results/probe-x.json -- python3 -I bench/probe.py x
Returns the command's code if the pass is clean, 5 if contaminated (discard and replay), 4 without rest, 6 if it
measured nothing or carries a request error: never a green on a pass that measured nothing.
Verdict written to <result>.pass.json.
"""
import argparse, json, os, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402


def counters():
    s = c.metric_sums(("vllm:request_success", "vllm:num_requests_running", "vllm:num_requests_waiting"))
    return {"finished": s.get("vllm:request_success_total", 0.0), "running": s.get("vllm:num_requests_running", 0.0),
            "waiting": s.get("vllm:num_requests_waiting", 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="JSON file the command writes")
    ap.add_argument("--rest", type=int, default=30)
    ap.add_argument("--max-wait", type=int, default=1800)
    ap.add_argument("--max-running", type=int, default=1, help="requests the pass itself runs at once")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    verdict = {"command": " ".join(os.path.basename(x) if os.sep in x else x for x in cmd), "result": os.path.basename(a.result)}
    out = a.result + ".pass.json"

    t0, calm = time.time(), 0
    while calm < a.rest:
        m = counters()
        calm = calm + 5 if (m["running"] == 0 and m["waiting"] == 0) else 0
        if time.time() - t0 > a.max_wait:
            verdict.update(verdict="no rest", waited_s=round(time.time() - t0))
            json.dump(verdict, open(out, "w"), indent=1)
            print(f"idle_pass: no rest within {a.max_wait} s", file=sys.stderr)
            return 4
        if calm < a.rest:
            time.sleep(5)
    verdict["waited_for_rest_s"] = round(time.time() - t0)

    before = counters()
    stop = threading.Event()
    peaks = {"running": 0.0, "waiting": 0.0}

    def sample():
        while not stop.wait(5):
            try:
                m = counters()
                peaks["running"] = max(peaks["running"], m["running"])
                peaks["waiting"] = max(peaks["waiting"], m["waiting"])
            except Exception:
                pass

    start = c.stamp()
    th = threading.Thread(target=sample, daemon=True); th.start()
    rc = subprocess.call(cmd, stdin=subprocess.DEVNULL)
    stop.set()
    after = counters()
    try:
        req = json.load(open(a.result)).get("requests") or {}
        sent, errors = req.get("sent"), req.get("errors", 0)
    except Exception as e:
        sent, errors = None, None
        verdict["count_error"] = str(e)
    delta = round(after["finished"] - before["finished"])
    clean = (sent is not None and delta == sent and peaks["running"] <= a.max_running and peaks["waiting"] == 0)
    empty = not sent or bool(errors)
    verdict.update(start=start, end=c.stamp(), rc=rc, sent=sent, errors=errors, finished_by_engine=delta,
                   peak_running=peaks["running"], peak_waiting=peaks["waiting"],
                   verdict="empty or in error" if empty else ("clean" if clean else "contaminated"))
    json.dump(verdict, open(out, "w"), indent=1)
    print(f"idle_pass: {verdict['verdict']} (sent {sent}, finished by the engine {delta}, peak running "
          f"{peaks['running']:.0f}, waiting {peaks['waiting']:.0f}) code {rc}", file=sys.stderr)
    if empty:
        return 6
    return rc if clean else 5


if __name__ == "__main__":
    sys.exit(main())
