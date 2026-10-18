#!/usr/bin/env python3
"""Coverage of model outputs by draft vocabulary sets: the share of generated tokens that each set contains.

The drafter can only propose tokens of its set, so a token outside the set is a guaranteed rejection at that
position. Coverage is the first-order predictor of acceptance; acceptance itself is measured by bench/probe.py.
Texts are the outputs saved by the benches (probe generations, code gate answers); tokens come from the server's
/tokenize (common.py configuration). Standard library only (the .npy files are int32 arrays, read directly).

  python3 -I tools/draft_vocab_coverage.py <label> --set public=build/draft_vocab_public_65536.npy \
      --set blazux=patches/draft-vocab/draft_vocab_65536.npy results/probe-x.json [results/code-gate-x.json ...]
Writes results/draft-vocab-coverage-<label>.json.
"""
import argparse, ast, hashlib, json, os, struct, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import common as c  # noqa: E402


def load_npy_int32(path):
    raw = open(path, "rb").read()
    if raw[:6] != b"\x93NUMPY":
        sys.exit(f"not a .npy file: {path}")
    major = raw[6]
    hlen = struct.unpack("<H" if major == 1 else "<I", raw[8:10] if major == 1 else raw[8:12])[0]
    start = (10 if major == 1 else 12) + hlen
    header = ast.literal_eval(raw[(10 if major == 1 else 12):start].decode("latin1"))
    if header["descr"] not in ("<i4", "|i4") or header.get("fortran_order"):
        sys.exit(f"{path}: expected a C-ordered int32 array, got {header}")
    n = header["shape"][0]
    return set(struct.unpack(f"<{n}i", raw[start:start + 4 * n])), hashlib.sha256(raw).hexdigest()


def texts(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    if d.get("kind") == "probe":
        seen = set()
        for g in d.get("generations", []):
            if g["prompt"] not in seen:  # greedy: pass 2 repeats pass 1
                seen.add(g["prompt"]); out.append((f"probe {g['prompt']}", g["text"]))
    elif d.get("kind") == "code-gate":
        out += [(f"code gate {r['name']}", r["content"]) for r in d.get("results", []) if r.get("content")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--set", action="append", required=True, help="name=path.npy")
    ap.add_argument("results", nargs="+")
    a = ap.parse_args()
    sets = {}
    for s in a.set:
        name, path = s.split("=", 1)
        sets[name] = load_npy_int32(path)
    model = c.served_model()
    res = c.header("draft-vocab-coverage", a.label, sets={k: {"ids": len(v[0]), "sha256": v[1]} for k, v in sets.items()},
                   sources=[os.path.basename(p) for p in a.results])
    rows = []
    for p in a.results:
        for name, text in texts(p):
            ids = c.call_json("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
            row = {"text": name, "tokens": len(ids)}
            for k, (ids_set, _) in sets.items():
                row[k] = round(100 * sum(1 for t in ids if t in ids_set) / max(1, len(ids)), 1)
            rows.append(row)
            print(f"{name:28} {len(ids):5} tokens  " + "  ".join(f"{k} {row[k]:5.1f} %" for k in sets))
    res["coverage_percent"] = rows
    res["requests"] = {"sent": 0, "errors": 0}
    c.write_result(res, f"draft-vocab-coverage-{a.label}.json")


if __name__ == "__main__":
    main()
