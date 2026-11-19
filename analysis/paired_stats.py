#!/usr/bin/env python3
"""Paired comparison of two runs on the same questions: accuracy of each, wins and losses of B against A, and the exact
two-sided sign test on the questions where exactly one of the two is right.

    python paired_stats.py A B [--ids FILE] [--by-type]

A and B are a run's results.jsonl (agent.py and answer_qwen.py write one per answer run, e.g. OUT/agent_only/results.jsonl)
or a JSON file {question id: true|false}. Only the questions in both are compared, or only those in --ids (one id per
line, or a JSON list or dict). The camera rows are the four MMSI-Bench types that involve the camera.
Example, LatentParse against photo only on the 986 MMSI-Bench questions of the appendix tables:
    python paired_stats.py mmsi_photo_only_paired986.json RUN/agent_only/results.jsonl --by-type
"""
import argparse
import json
from math import comb
from pathlib import Path

CAMERA = {"Motion (Cam.)", "Positional Relationship (Cam.–Cam.)", "Positional Relationship (Cam.–Obj.)",
          "Positional Relationship (Cam.–Reg.)"}


def load(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return {str(r["id"]): bool(r["correct"]) for r in rows}, {str(r["id"]): r.get("type") for r in rows}
    d = json.loads(p.read_text())
    return {str(k): bool(v) for k, v in d.items()}, {}


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def row(name, ids, a, b):
    w = sum(b[i] and not a[i] for i in ids)
    l = sum(a[i] and not b[i] for i in ids)
    n = len(ids)
    print(f"{name:<45} {n:>5} {100 * sum(a[i] for i in ids) / n:6.1f} {100 * sum(b[i] for i in ids) / n:6.1f} "
          f"{w:>4}W/{l}L  p={sign_test(w, l):.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ids")
    ap.add_argument("--by-type", action="store_true")
    args = ap.parse_args()
    a, ta = load(args.a)
    b, tb = load(args.b)
    types = {**ta, **tb}
    ids = sorted(set(a) & set(b), key=lambda x: (len(x), x))
    if args.ids:
        txt = Path(args.ids).read_text()
        keep = set(map(str, json.loads(txt))) if txt.lstrip().startswith(("[", "{")) else set(txt.split())
        ids = [i for i in ids if i in keep]
    print(f"{'':<45} {'n':>5} {'A':>6} {'B':>6}")
    row("All", ids, a, b)
    if types:
        row("Camera, all four types", [i for i in ids if types.get(i) in CAMERA], a, b)
        for t in ("Motion (Cam.)", "MSR"):
            sub = [i for i in ids if types.get(i) == t]
            if sub:
                row(t, sub, a, b)
        if args.by_type:
            for t in sorted({types.get(i) for i in ids} - {None}):
                row(t, [i for i in ids if types.get(i) == t], a, b)


if __name__ == "__main__":
    main()
