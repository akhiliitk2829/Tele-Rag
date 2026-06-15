#!/usr/bin/env python3
# Trim a large chunks.jsonl before embedding: drop boilerplate, restrict to
# releases/specs, and remove exact duplicates.
#
#   python filter_chunks.py --in chunks.jsonl --stats
#   python filter_chunks.py --in chunks.jsonl --out chunks_dedup.jsonl \
#       --releases 16 17 18 --keep 38.300 38.331 38.321 38.323 38.401 38.413 \
#       --clean --dedup

import argparse
import json
import re
from collections import defaultdict

# legal/front-matter text that repeats near-identically across every spec
BOILERPLATE = re.compile(
    r"Organizational Partners|Postal address|Route des Lucioles|Copyright Notification|"
    r"All rights reserved|accept no liability|shall not be implemented|Internet\s+http|"
    r"Keywords\b", re.I)
DROP_SECTIONS = {"(none)", "foreword", "contents", "table of contents"}


def is_junk(rec):
    last = (rec.get("section", "") or "").split(">")[-1].strip().lower()
    last = re.sub(r"^\d+(\.\d+)*\s*", "", last)
    if last in DROP_SECTIONS:
        return True
    return bool(BOILERPLATE.search(rec.get("text", "")))


def load(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def stats(path):
    per_spec, per_rel, total = defaultdict(int), defaultdict(int), 0
    for r in load(path):
        per_spec[r["spec"]] += 1
        per_rel[r["release"]] += 1
        total += 1
    print(f"total chunks: {total}\n\nby release:")
    for rel, n in sorted(per_rel.items()):
        print(f"  R{rel}: {n}")
    print("\nper spec (highest first):")
    for spec, n in sorted(per_spec.items(), key=lambda x: -x[1]):
        print(f"  {spec:>8}: {n:>7}")


def filter_file(path, out, releases, keep, drop, clean, dedup):
    rel_set = set(releases) if releases else None
    keep_set = set(keep) if keep else None
    drop_set = set(drop) if drop else set()
    seen = set()
    kept = total = junk = dups = 0
    with open(out, "w", encoding="utf-8") as w:
        for r in load(path):
            total += 1
            if rel_set is not None and r["release"] not in rel_set:
                continue
            if keep_set is not None and r["spec"] not in keep_set:
                continue
            if r["spec"] in drop_set:
                continue
            if clean and is_junk(r):
                junk += 1
                continue
            if dedup:
                key = (r["spec"], r["release"], r["text"])
                if key in seen:
                    dups += 1
                    continue
                seen.add(key)
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    extra = ""
    if clean:
        extra += f" ({junk} boilerplate dropped)"
    if dedup:
        extra += f" ({dups} dups dropped)"
    print(f"kept {kept} / {total}{extra} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--releases", nargs="+", type=int)
    ap.add_argument("--keep", nargs="+")
    ap.add_argument("--drop", nargs="+")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    a = ap.parse_args()
    if a.stats:
        stats(a.in_path)
    else:
        if not a.out_path:
            ap.error("--out required when not using --stats")
        filter_file(a.in_path, a.out_path, a.releases, a.keep, a.drop, a.clean, a.dedup)
