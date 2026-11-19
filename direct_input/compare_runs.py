#!/usr/bin/env python3
"""Compare latent runs against the image-only baseline, per sample and pooled.

Usage: python compare_runs.py SAMPLE_TAG=RESULTS_DIR [SAMPLE_TAG=RESULTS_DIR ...]

Each RESULTS_DIR holds <run>/results.jsonl for baseline and the latent runs (the
VGGT and CLIP jobs write to separate dirs, so pass the same tag twice). For each
run: accuracy, answers fixed / broken relative to the baseline on the same
questions, and a two-sided exact sign test on those changed answers.
"""
import json
import math
import sys
from pathlib import Path

RUNS = [f"{k}_{f}{w}" for k in ("clip", "vggt") for f in ("image", "pad", "padglobal", "imagez", "tilez", "padz", "tokz") for w in ("", "_wrong")]


def sign_p(k, n):
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def load_runs(args):
    """{run name: {(sample tag, bench, id): row}} from TAG=DIR arguments."""
    data = {}
    for arg in args:
        tag, path = arg.split("=", 1)
        for f in Path(path).glob("*/results.jsonl"):
            for line in f.open():
                r = json.loads(line)
                data.setdefault(f.parent.name, {})[(tag, r["bench"], r["id"])] = r
    return data


def main():
    data = load_runs(sys.argv[1:])
    common = set.intersection(*(set(rows) for rows in data.values()))  # e.g. a pilot subset
    data = {run: {k: r for k, r in rows.items() if k in common} for run, rows in data.items()}
    base = data["baseline"]
    runs = [r for r in RUNS if r in data] + sorted(r for r in data if r not in RUNS and r != "baseline")
    tags = sorted({k[0] for k in base})
    print("unparsed:", {run: sum(r["prediction"] is None for r in rows.values()) for run, rows in data.items()})
    print("| sample | bench | n | chance | baseline | " + " | ".join(runs) + " |")
    for scope in tags + (["pooled"] if len(tags) > 1 else []):
        for bench in ("3dsr", "mmsi", "mindcube", "all"):
            keys = [k for k in base if scope in (k[0], "pooled") and bench in (k[1], "all")]
            if not keys:  # a benchmark this sample does not have (e.g. mindcube, in a sample dir of its own)
                continue
            chance = 100 * sum(1 / len(base[k]["options"]) for k in keys) / len(keys)
            cells = [f"{100 * sum(base[k]['correct'] for k in keys) / len(keys):.1f}"]
            for run in runs:
                r = data[run]
                fixed = sum(r[k]["correct"] and not base[k]["correct"] for k in keys)
                broke = sum(base[k]["correct"] and not r[k]["correct"] for k in keys)
                acc = 100 * sum(r[k]["correct"] for k in keys) / len(keys)
                cells.append(f"{acc:.1f} (+{fixed}/-{broke}, p={sign_p(fixed, fixed + broke):.2f})")
            print(f"| {scope} | {bench} | {len(keys)} | {chance:.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
