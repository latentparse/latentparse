#!/usr/bin/env python3
"""Draw a sample: random questions per benchmark with a fixed seed.

Usage: python3 make_subset.py SRC_DIR OUT_DIR [--n 100 | --max-images 100] [--seed 42]

--n takes N questions per benchmark. --max-images instead walks the questions in
seeded random order and keeps each one whose images still fit in the budget, so
multi-image MMSI questions end up with about that many images in total.

SRC_DIR holds samples_3dsr_200.json, samples_mmsi_200.json and img/{3dsr,mmsi}/.
OUT_DIR gets the same layout with only the sampled questions and their images,
plus sample_ids.json and README.md. Paths inside the JSON stay relative (img/...).
MMSI questions whose four options were merged into one entry are split back apart.
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

BENCHES = {"3dsr": ("3dsrbench", "img"), "mmsi": ("mmsi", "imgs")}


def split_merged_options(opts):
    """[['A', 'x; B: y; C: z; D: w']] -> [['A','x'],['B','y'],['C','z'],['D','w']]."""
    if len(opts) != 1:
        return opts, False
    text = f"A: {opts[0][1]}"
    parts = re.split(r"[\s;.]*\b([A-D]):\s+", text)
    pairs = [[parts[i], parts[i + 1].strip().rstrip(";")] for i in range(1, len(parts) - 1, 2)]
    if [p[0] for p in pairs] != ["A", "B", "C", "D"]:
        return opts, False
    return pairs, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tag = f"{args.max_images}img" if args.max_images else str(args.n)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; pick a new directory")
    ids, notes = {}, []
    for bench, (key, img_field) in BENCHES.items():
        items = json.loads((args.src / f"samples_{bench}_200.json").read_text())[key]
        order = list(range(len(items)))
        random.Random(args.seed).shuffle(order)
        if args.max_images:
            picked, used = [], 0
            for i in order:
                k = len(items[i][img_field]) if isinstance(items[i][img_field], list) else 1
                if used + k <= args.max_images:
                    picked.append(i)
                    used += k
        else:
            picked = order[: args.n]
        sample = [items[i] for i in sorted(picked)]  # keep the source file's order

        (args.out / "img" / bench).mkdir(parents=True)
        n_imgs = 0
        for it in sample:
            paths = it[img_field] if isinstance(it[img_field], list) else [it[img_field]]
            for p in paths:
                shutil.copy2(args.src / p, args.out / p)
                n_imgs += 1
            if bench == "mmsi":
                it["opts"], fixed = split_merged_options(it["opts"])
                if fixed:
                    notes.append(f"mmsi {it['id']}: merged options split into A-D")
                elif len(it["opts"]) != 4:
                    notes.append(f"mmsi {it['id']}: options still malformed, check by hand")

        (args.out / f"samples_{bench}_{tag}.json").write_text(json.dumps({key: sample}, ensure_ascii=False, indent=1))
        ids[bench] = [it["id"] for it in sample]
        print(f"{bench}: {len(sample)} questions, {n_imgs} images")

    meta = {"seed": args.seed, "n_per_benchmark": args.n if not args.max_images else None, "max_images": args.max_images, "source": str(args.src.resolve()), "ids": ids, "notes": notes}
    (args.out / "sample_ids.json").write_text(json.dumps(meta, indent=1))
    what = (f"Random questions per benchmark, up to {args.max_images} images each"
            if args.max_images else f"{args.n} random questions per benchmark")
    (args.out / "README.md").write_text(
        f"# Phase 1 sample\n\n"
        f"{what}, seed {args.seed}, drawn from "
        f"`samples_3dsr_200.json` and `samples_mmsi_200.json` in `{args.src.resolve()}`.\n\n"
        f"- `samples_3dsr_{tag}.json`: 3DSRBench, one image per question (`img`)\n"
        f"- `samples_mmsi_{tag}.json`: MMSI-Bench, 2-9 images per question (`imgs`, `<id>_<view>.jpg`)\n"
        f"- `img/`: only the images these questions use\n"
        f"- `sample_ids.json`: seed and question IDs\n\n"
        f"Fixes applied:\n\n" + ("\n".join(f"- {n}" for n in notes) or "- none") + "\n"
    )
    for n in notes:
        print(n)


if __name__ == "__main__":
    main()
