#!/usr/bin/env python3
"""Build a sample, or with --full the whole MMSI-Bench and 3DSRBench, straight from the Hugging Face files.

Usage: python build_from_hf.py --dsr-parquet P --dsr-zip Z --mmsi-parquet M --out DIR \
           [--exclude PREV_SAMPLE_DIR ...] [--max-images 100] [--seed 42]

Output has the same layout and JSON fields as make_subset.py, so answer_qwen.py
reads it unchanged.
- 3DSRBench: original rows only (no "-flip" twins), one question per COCO image
  (picked at random when an image has several), images not used in any --exclude
  sample. Then up to --max-images questions in seeded random order.
- MMSI-Bench: options are split out of the question text; questions whose options
  cannot be split into A-D are dropped (reported). Ids in --exclude are skipped,
  then questions are added in seeded random order while their images fit the budget.
"""
import argparse
import collections
import io
import json
import random
import re
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

OPT_SPLIT = re.compile(r"[\s,;.]*\b([A-D]):\s+")


def split_options(text):
    """'q text\\nOptions: A: x, B: y, ...' -> ('q text', [['A','x'], ...]) or None."""
    if "\nOptions:" not in text:
        return None
    q, opts = text.rsplit("\nOptions:", 1)
    parts = OPT_SPLIT.split(" " + opts.strip())
    pairs = [[parts[i], parts[i + 1].strip().rstrip(",;")] for i in range(1, len(parts) - 1, 2)]
    if [p[0] for p in pairs] != list("ABCD")[: len(pairs)] or len(pairs) < 2:
        return None
    return q.strip(), pairs


def excluded(dirs):
    dsr_imgs, mmsi_ids = set(), set()
    for d in dirs:
        for f in Path(d).glob("samples_3dsr_*.json"):
            dsr_imgs |= {Path(x["img"]).name for x in json.loads(f.read_text())["3dsrbench"]}
        for f in Path(d).glob("samples_mmsi_*.json"):
            mmsi_ids |= {str(x["id"]) for x in json.loads(f.read_text())["mmsi"]}
    return dsr_imgs, mmsi_ids


def build_3dsr(parquet, zip_path, out, skip_imgs, budget, rng, every_question=False):
    rows = [r for r in pq.read_table(parquet).to_pylist() if not r["index"].endswith("-flip")]
    by_image = collections.defaultdict(list)
    for r in rows:
        by_image[Path(r["image_url"]).name].append(r)
    zf = zipfile.ZipFile(zip_path)
    in_zip = {Path(n).name: n for n in zf.namelist() if n.endswith(".jpg")}
    images = sorted(n for n in by_image if n not in skip_imgs and n in in_zip)
    rng.shuffle(images)
    picked = []
    for name in images[:budget]:
        # every_question: the full benchmark, all original questions of each image; else one at random
        chosen = by_image[name] if every_question else [rng.choice(by_image[name])]
        (out / "img" / "3dsr" / name).write_bytes(zf.read(in_zip[name]))
        for r in chosen:
            opts = [[k, r[k]] for k in "ABCD" if r[k] not in (None, "None")]
            picked.append({"cat": r["category"], "id": r["index"], "q": r["question"], "opts": opts,
                           "ans": r["answer"], "img": f"img/3dsr/{name}", "n": len(by_image[name])})
    missing = sum(1 for n in by_image if n not in in_zip)
    return picked, {"original_rows": len(rows), "images": len(by_image), "images_missing_from_zip": missing}


def build_mmsi(parquet, out, skip_ids, budget, rng):
    pf = pq.ParquetFile(parquet)
    meta = pf.read(columns=["id", "question", "answer", "question_type", "difficulty"]).to_pylist()
    dropped, usable = [], []
    for i, r in enumerate(meta):
        if str(r["id"]) in skip_ids:
            continue
        parsed = split_options(r["question"])
        if parsed is None:
            dropped.append(str(r["id"]))
        else:
            usable.append((i, r, parsed))
    rng.shuffle(usable)
    images_col = pf.read(columns=["images"]).column("images")
    picked, used = [], 0
    for i, r, (q, opts) in usable:
        imgs = images_col[i].as_py()
        if used + len(imgs) > budget:
            continue
        paths = []
        for k, im in enumerate(imgs):
            data = im["bytes"] if isinstance(im, dict) else im
            p = f"img/mmsi/{r['id']}_{k}.jpg"
            (out / p).write_bytes(data)
            paths.append(p)
        used += len(imgs)
        picked.append({"type": r["question_type"], "id": r["id"], "q": q, "opts": opts, "ans": r["answer"],
                       "diff": r["difficulty"], "imgs": paths})
    picked.sort(key=lambda x: int(x["id"]))
    return picked, {"questions": len(meta), "dropped_unparsable_options": dropped, "images_used": used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsr-parquet", required=True)
    ap.add_argument("--dsr-zip", required=True)
    ap.add_argument("--mmsi-parquet", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--max-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true",
                    help="whole benchmarks: every original 3DSRBench question and every parsable MMSI question")
    args = ap.parse_args()
    if args.full:
        args.max_images = 10 ** 9

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; pick a new directory")
    for b in ("3dsr", "mmsi"):
        (args.out / "img" / b).mkdir(parents=True)
    skip_imgs, skip_ids = excluded(args.exclude)
    tag = "full" if args.full else f"{args.max_images}img"
    dsr, dsr_info = build_3dsr(args.dsr_parquet, args.dsr_zip, args.out, skip_imgs, args.max_images, random.Random(args.seed),
                               every_question=args.full)
    mmsi, mmsi_info = build_mmsi(args.mmsi_parquet, args.out, skip_ids, args.max_images, random.Random(args.seed))
    (args.out / f"samples_3dsr_{tag}.json").write_text(json.dumps({"3dsrbench": dsr}, ensure_ascii=False, indent=1))
    (args.out / f"samples_mmsi_{tag}.json").write_text(json.dumps({"mmsi": mmsi}, ensure_ascii=False, indent=1))
    meta = {"seed": args.seed, "max_images": args.max_images, "excluded_samples": args.exclude,
            "3dsr": dsr_info, "mmsi": mmsi_info,
            "ids": {"3dsr": [x["id"] for x in dsr], "mmsi": [x["id"] for x in mmsi]}}
    (args.out / "sample_ids.json").write_text(json.dumps(meta, indent=1))
    (args.out / "README.md").write_text(
        f"# {'Full benchmarks' if args.full else 'Phase 1 sample'}\n\nBuilt from the Hugging Face releases (ccvl/3DSRBench, RunsenXu/MMSI-Bench) with "
        f"`build_from_hf.py`, seed {args.seed}, up to {args.max_images} images per benchmark, no overlap with: "
        f"{', '.join(args.exclude) or 'nothing'}.\n\n"
        f"- `samples_3dsr_{tag}.json`: {len(dsr)} 3DSRBench questions, original (unflipped), one per image\n"
        f"- `samples_mmsi_{tag}.json`: {len(mmsi)} MMSI-Bench questions, {mmsi_info['images_used']} images\n"
        f"- `sample_ids.json`: seed, ids and build notes\n")
    print(f"3dsr: {len(dsr)} questions / images; mmsi: {len(mmsi)} questions, {mmsi_info['images_used']} images")
    print("3dsr info:", dsr_info)
    print("mmsi dropped (options not splittable):", mmsi_info["dropped_unparsable_options"][:20], len(mmsi_info["dropped_unparsable_options"]))


if __name__ == "__main__":
    main()
