#!/usr/bin/env python3
"""Freeze the public corpus of the draft vocabulary: pick the sources once and record each one in a manifest.

The manifest lists immutable sources only, each with the sha256 of its bytes:
  - Wikipedia articles at a fixed revision id (index.php?oldid=<id>&action=raw), French and English
    featured articles, picked deterministically (titles sorted, every k-th, until a character budget);
  - source files of public repositories at a fixed commit (raw.githubusercontent.com/<repo>/<commit>/<path>).
fetch_public_corpus.py downloads the manifest and refuses any byte that differs. This script is run once, by the
maintainers, and its output is committed; rerunning it later gives a different manifest (new revisions).

  python3 -I tools/freeze_public_corpus.py patches/draft-vocab/public-corpus.tsv
Needs network access. Standard library only.
"""
import hashlib, json, sys, time, urllib.parse, urllib.request

UA = "qwen38-flash-next-dgx-spark-vllm/0.1 (https://github.com/Aiko-Academy)"

WIKIS = [  # (language, featured-article category, character budget of raw wikitext)
    ("fr", "Catégorie:Article_de_qualité", 12_000_000),
    ("en", "Category:Featured_articles", 6_000_000),
]
REPOS = [  # (repository, commit, path prefixes, extensions, character budget)
    ("vllm-project/vllm", "db9527a46873454610df6dbedf79a36d6bf1a7f6",  # tag v0.31.0
     ("vllm/entrypoints/", "vllm/v1/core/", "vllm/config/", "tools/", "docker/"), (".py", ".sh"), 4_000_000),
    ("microsoft/TypeScript", "c63de15a992d37f0d6cec03ac7631872838602cb",  # tag v5.9.3
     ("src/services/", "src/server/"), (".ts",), 4_000_000),
    ("tokio-rs/tokio", "be8ee45b3fc2d107174e586141b1cb12c93e2ddf",  # tag tokio-1.47.1
     ("tokio/src/",), (".rs",), 2_000_000),
]


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def api(lang, **params):
    q = urllib.parse.urlencode(dict(params, format="json", formatversion="2"))
    return json.loads(get(f"https://{lang}.wikipedia.org/w/api.php?{q}"))


def featured(lang, category):
    titles, cont = [], {}
    while True:
        d = api(lang, action="query", list="categorymembers", cmtitle=category, cmnamespace="0", cmlimit="500", **cont)
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return sorted(titles)
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def revisions(lang, titles):
    out = {}
    for i in range(0, len(titles), 50):
        d = api(lang, action="query", prop="revisions", rvprop="ids|size", titles="|".join(titles[i:i + 50]))
        for p in d["query"]["pages"]:
            if p.get("revisions"):
                out[p["title"]] = (p["revisions"][0]["revid"], p["revisions"][0]["size"])
    return out


def main():
    rows = []
    for lang, category, budget in WIKIS:
        titles = featured(lang, category)
        revs = revisions(lang, titles)
        step = max(1, len(titles) // 400)
        total = 0
        for t in titles[::step]:
            if total >= budget or t not in revs:
                continue
            revid = revs[t][0]
            raw = get(f"https://{lang}.wikipedia.org/w/index.php?oldid={revid}&action=raw")
            text = raw.decode("utf-8")
            rows.append(("wiki", lang, str(revid), t, hashlib.sha256(raw).hexdigest(), str(len(text))))
            total += len(text)
        print(f"{lang}: {len(titles)} featured articles, {sum(1 for r in rows if r[1] == lang)} kept, {total} characters",
              file=sys.stderr)
    for repo, commit, prefixes, exts, budget in REPOS:
        tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
        paths = sorted(e["path"] for e in tree["tree"]
                       if e["type"] == "blob" and e["path"].startswith(prefixes) and e["path"].endswith(exts))
        total, kept = 0, 0
        for p in paths:
            if total >= budget:
                break
            raw = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(p)}")
            text = raw.decode("utf-8", errors="replace")
            rows.append(("git", repo, commit, p, hashlib.sha256(raw).hexdigest(), str(len(text))))
            total += len(text); kept += 1
        print(f"{repo}@{commit[:12]}: {kept} files, {total} characters", file=sys.stderr)
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write("kind\tsource\trevision\tpath\tsha256\tchars\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"written: {sys.argv[1]} ({len(rows)} sources)", file=sys.stderr)


if __name__ == "__main__":
    main()
