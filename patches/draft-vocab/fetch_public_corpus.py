#!/usr/bin/env python3
"""Download the frozen public corpus listed in public-corpus.tsv and refuse any byte that differs.

Each source is immutable: a Wikipedia revision (oldid) or a file of a public repository at a fixed commit. The raw
bytes are checked against the sha256 of the manifest, then written as plain text: Wikipedia wikitext goes through
a small deterministic cleaner (templates, references, tables, link targets and markup removed), code is kept as
is. Only token counts of this text reach the draft vocabulary; the text itself is not redistributed.

  python3 -I patches/draft-vocab/fetch_public_corpus.py patches/draft-vocab/public-corpus.tsv build/public-corpus
Standard library only. Already downloaded files are checked again and kept.
"""
import hashlib, os, re, sys, time, urllib.parse, urllib.request

UA = "qwen38-flash-next-dgx-spark-vllm/0.1 (https://github.com/Aiko-Academy)"


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def clean_wikitext(t):
    """Readable prose out of wikitext. Deterministic; good enough for token counts, not for reading."""
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>/]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    for _ in range(6):  # nested templates and tables, innermost first
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
        t = re.sub(r"\{\|[^{}]*?\|\}", "", t, flags=re.S)
    t = re.sub(r"\[\[(?:Fichier|File|Image|Catégorie|Category):[^\]]*\]\]", "", t)
    t = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"


def main():
    manifest, out = sys.argv[1], sys.argv[2]
    rows = [l.rstrip("\n").split("\t") for l in open(manifest, encoding="utf-8")][1:]
    os.makedirs(out, exist_ok=True)
    chars = {}
    bad = 0
    for i, (kind, source, rev, path, want, _n) in enumerate(rows):
        name = os.path.join(out, f"{i:05d}." + ("txt" if kind == "wiki" else "code.txt"))
        raw_path = name + ".raw"
        if os.path.exists(raw_path) and hashlib.sha256(open(raw_path, "rb").read()).hexdigest() == want:
            raw = open(raw_path, "rb").read()
        else:
            if kind == "wiki":
                raw = get(f"https://{source}.wikipedia.org/w/index.php?oldid={rev}&action=raw")
            else:
                raw = get(f"https://raw.githubusercontent.com/{source}/{rev}/{urllib.parse.quote(path)}")
            if hashlib.sha256(raw).hexdigest() != want:
                print(f"DIFFERS {kind} {source} {rev} {path}", file=sys.stderr)
                bad += 1
                continue
            open(raw_path, "wb").write(raw)
        text = raw.decode("utf-8", errors="replace")
        text = clean_wikitext(text) if kind == "wiki" else text
        open(name, "w", encoding="utf-8").write(text)
        key = source if kind == "git" else f"wikipedia-{source}"
        chars[key] = chars.get(key, 0) + len(text)
    for k, v in sorted(chars.items()):
        print(f"{k}: {v / 1e6:.2f} M characters")
    print(f"{len(rows) - bad} / {len(rows)} sources match the manifest")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
