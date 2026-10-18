#!/usr/bin/env python3
"""Mechanical judge of the code gate: a witness that runs is worth more than a review. For each result file, takes
the code block of each task and:
  - solidity_guard: compiles with solc (SOLC, a solc 0.8.x binary);
  - abi_runner_ethers, react_hook_ts: TypeScript syntax through esbuild (ESBUILD, an esbuild binary);
  - code_long: bash -n.
An empty answer is never right (solc, esbuild and bash -n accept an empty file). A missing tool marks the task "not
judged", never OK.
  SOLC=/path/to/solc-0.8.28 ESBUILD=/path/to/esbuild python3 -I bench/quality/judge_code.py results/code-gate-*.json
Adds a "judged" section to each file and prints the count.
"""
import json, os, re, shutil, subprocess, sys, tempfile

SOLC = os.environ.get("SOLC") or shutil.which("solc")
ESBUILD = os.environ.get("ESBUILD") or shutil.which("esbuild")


def block(text, langs):
    blocks = re.findall(r"```([a-zA-Z]*)\n(.*?)```", text, re.S)
    for lang, body in blocks:
        if lang.lower() in langs:
            return body
    return blocks[0][1] if blocks else text


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180)
    return r.returncode, r.stdout + r.stderr


def judge(name, content, d):
    if not (content or "").strip():
        return False, "empty content"
    if name == "solidity_guard":
        if not SOLC:
            return None, "not judged: no solc (SOLC)"
        p = os.path.join(d, "C.sol"); open(p, "w").write(block(content, {"solidity", "sol"}))
        rc, out = sh([SOLC, "--bin", p])
        return rc == 0, "compiles" if rc == 0 else "does not compile: " + " | ".join(l for l in out.splitlines() if "Error" in l)[:200]
    if name in ("abi_runner_ethers", "react_hook_ts"):
        if not ESBUILD:
            return None, "not judged: no esbuild (ESBUILD)"
        p = os.path.join(d, "m.tsx" if name == "react_hook_ts" else "m.ts")
        open(p, "w").write(block(content, {"ts", "typescript", "tsx", "js", "javascript"}))
        rc, out = sh([ESBUILD, p, "--format=esm", "--log-level=error", f"--outfile={d}/m.mjs"])
        return rc == 0, "valid syntax" if rc == 0 else "syntax: " + out.strip()[:200]
    if name == "code_long":
        p = os.path.join(d, "s.sh"); open(p, "w").write(block(content, {"bash", "sh", "shell"}))
        rc, out = sh(["bash", "-n", p])
        return rc == 0, "bash -n valid" if rc == 0 else "bash -n: " + out[:150]
    return None, "not judged"


def main():
    total = 0
    for f in sys.argv[1:]:
        data = json.load(open(f))
        judged = []
        print(f"== {os.path.basename(f)}")
        for r in data["results"]:
            with tempfile.TemporaryDirectory() as d:
                ok, msg = judge(r["name"], r.get("content", ""), d)
            judged.append({"name": r["name"], "ok": ok, "detail": msg})
            total += bool(ok)
            print(f"  {'OK   ' if ok else ('-    ' if ok is None else 'FAIL ')} {r['name']}: {msg}")
        data["judged"] = {"ok": sum(1 for j in judged if j["ok"]), "total": len(judged), "tasks": judged}
        json.dump(data, open(f, "w"), ensure_ascii=False, indent=1)
    print(f"total OK: {total}")


if __name__ == "__main__":
    main()
