"""Shared helpers of the benches: server address, API key, /metrics, machine counters, result files.

Configuration, by environment variable only:
  VLLM_URL           server root, default http://127.0.0.1:8000
  VLLM_API_KEY       the key itself, or
  VLLM_API_KEY_FILE  a file holding a line VLLM_API_KEY=<key> (the docker --env-file of serve.sh fits)
  SPARK_SSH          optional user@host of the Spark, to read its /proc counters from another machine; when
                     VLLM_URL points at localhost the counters are read locally; otherwise they are left null
  SPARK_SSH_KEY      optional private key for SPARK_SSH
  NVME_DEVICE        disk of the checkpoint as named in /proc/diskstats, default nvme0n1
The key is read by name and never printed or written. Result files carry no address, no key and no host name.
"""
import json, os, re, subprocess, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
NVME = os.environ.get("NVME_DEVICE", "nvme0n1")


def api_key():
    if os.environ.get("VLLM_API_KEY"):
        return os.environ["VLLM_API_KEY"]
    path = os.environ.get("VLLM_API_KEY_FILE")
    if path:
        for line in open(os.path.expanduser(path)):
            m = re.match(r"\s*(?:export\s+)?VLLM_API_KEY=(.*)", line)
            if m:
                return m.group(1).strip().strip("'\"")
        sys.exit(f"no VLLM_API_KEY line in {path}")
    return ""


KEY = api_key()
HEADERS = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}


def call(path, body=None, timeout=600):
    """GET (body None) or POST JSON; returns the decoded text."""
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None, HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def call_json(path, body=None, timeout=600):
    return json.loads(call(path, body, timeout))


def served_model():
    return call_json("/v1/models")["data"][0]["id"]


def metric_sums(names):
    """Sum of each vllm: metric over all its label sets; per-position counters keyed name[pos]."""
    out = {}
    for line in call("/metrics", timeout=10).splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m or not m.group(1).startswith(tuple(names)):
            continue
        pos = re.search(r'position="(\d+)"', m.group(2) or "")
        key = m.group(1) + (f"[{pos.group(1)}]" if pos else "")
        out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def running():
    s = metric_sums(("vllm:num_requests_running", "vllm:num_requests_waiting"))
    return s.get("vllm:num_requests_running", 0.0), s.get("vllm:num_requests_waiting", 0.0)


def wait_idle(seconds=20, limit=600):
    """True once no request has run or waited for `seconds`; False after `limit` seconds."""
    t0, calm = time.time(), None
    while time.time() - t0 < limit:
        if running() == (0.0, 0.0):
            calm = calm or time.time()
            if time.time() - calm >= seconds:
                return True
        else:
            calm = None
        time.sleep(1)
    return False


def on_spark(cmd, stdin_text=None, timeout=60):
    """Run a shell command on the Spark (locally if the server is local, else through SPARK_SSH); None if neither
    or if it fails. stdin_text, when given, is fed to the command."""
    host = urllib.parse.urlparse(URL).hostname
    if host in ("127.0.0.1", "localhost", "::1"):
        argv = ["sh", "-c", cmd]
    elif os.environ.get("SPARK_SSH"):
        argv = ["ssh"] + (["-n"] if stdin_text is None else []) + ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if os.environ.get("SPARK_SSH_KEY"):
            argv += ["-i", os.path.expanduser(os.environ["SPARK_SSH_KEY"])]
        argv += [os.environ["SPARK_SSH"], cmd]
    else:
        return None
    if stdin_text is None:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    else:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=stdin_text)
    return r.stdout if r.returncode == 0 else None


def machine():
    """Major faults, sectors read on the NVMe, CPU ticks of EngineCore, free swap (KiB); None if unreadable."""
    out = on_spark("awk '/^pgmajfault/{print $2}' /proc/vmstat; "
                   f"awk '$3==\"{NVME}\"{{print $6}}' /proc/diskstats; "
                   "p=$(pgrep -f 'VLLM::EngineCore' | head -1); [ -n \"$p\" ] && awk '{print $14+$15}' /proc/$p/stat || echo 0; "
                   "awk '/^SwapFree/{print $2}' /proc/meminfo")
    if not out or len(out.split()) < 4:
        return None
    v = [int(x) for x in out.split()[:4]]
    return {"majflt": v[0], "sectors_read": v[1], "engine_cpu_ticks": v[2], "swap_free_kib": v[3]}


def repo_commit():
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return (sha or "unknown") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def header(kind, label, **protocol):
    """Common head of every result file."""
    return {"kind": kind, "label": label, "measured_at": stamp(), "repo_commit": repo_commit(),
            "served_model": served_model(), "protocol": protocol}


def write_result(result, name):
    path = name if os.path.isabs(name) or os.sep in name else os.path.join(REPO, "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result["finished_at"] = stamp()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"written: {os.path.relpath(path, REPO) if path.startswith(REPO) else path}")
    return path


def prompts(name):
    with open(os.path.join(HERE, "prompts", name), encoding="utf-8") as f:
        return json.load(f) if name.endswith(".json") else f.read()
