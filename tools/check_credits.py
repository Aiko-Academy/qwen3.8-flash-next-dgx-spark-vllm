#!/usr/bin/env python3
"""Check that every third-party file of the repository is linked to a row of CREDITS.md.

Every file of the repository (git ls-files, or a walk without .git, build and results when there is no index) must
be either a row of the "Third-party files" table of CREDITS.md, with an author, a repository, a commit or PR and a
licence, or matched by a pattern of the "Our own files" list. A file in neither, or in both, a row naming a file that
does not exist, or a row with an empty field, is an error. Prints the count of third-party files and of linked rows;
exit 0 only if they are equal and nothing is wrong.
  python3 -I tools/check_credits.py
"""
import fnmatch, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_files():
    r = subprocess.run(["git", "-C", REPO, "ls-files", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return sorted(set(r.stdout.split("\n")) - {""})
    out = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__")]
        out += [os.path.relpath(os.path.join(root, f), REPO) for f in files if f != ".DS_Store"]
    return sorted(out)


def main():
    text = open(os.path.join(REPO, "CREDITS.md"), encoding="utf-8").read()
    table = text.split("## Third-party files", 1)[1].split("\n## ", 1)[0]
    rows, errors = {}, []
    for line in table.splitlines():
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(cells) != 6 or not cells[0].startswith("`"):
            continue
        name = cells[0].strip("`")
        if name in rows:
            errors.append(f"row twice: {name}")
        if not all(cells[1:5]):
            errors.append(f"row with an empty field: {name}")
        rows[name] = cells
    own_section = text.split("## Our own files", 1)[1]
    patterns = re.findall(r"`([^`]+)`", own_section)
    files = repo_files()
    third = [f for f in files if f in rows]
    for f in files:
        own = any(fnmatch.fnmatch(f, p) for p in patterns)
        if f in rows and own:
            errors.append(f"both third-party and own: {f}")
        if f not in rows and not own:
            errors.append(f"not classified: {f}")
    for name in rows:
        if not os.path.exists(os.path.join(REPO, name)):
            errors.append(f"row names a missing file: {name}")
    for e in errors:
        print("ERROR " + e)
    print(f"third-party files: {len(third)}; rows linking them to an author, a repository, a commit or PR and a "
          f"licence: {sum(1 for n in third if all(rows[n][1:5]))}; files checked: {len(files)}")
    ok = not errors and len(third) == len(rows)
    print("CREDITS complete" if ok else "CREDITS incomplete")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
