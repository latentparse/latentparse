#!/usr/bin/env python3
"""Build the SPAR-Bench-Tiny sample (550 multiple-choice questions) from the Hugging Face release.

    python build_spar_tiny.py SRC OUT_DIR

SRC is a local copy of the dataset jasonzhango/SPAR-Bench-Tiny, revision REVISION below (data/test-*.parquet, 1,000 rows,
all ScanNet). The paper uses its 550 rows with format_type "select" (multiple choice): 350 multi-view (2 or 3 views) and
200 single-view, 50 per task; the 450 "fill" rows ask for a number and are left out. OUT_DIR must not exist. Output, in
the layout of build_mindcube_tiny.py, so direct_input/common.load_sample reads it as benchmark "spar":
- samples_spar_full.json: {"spar": [{"type", "id", "q", "opts", "ans", "imgs", ...}]} in the release order. id is the
  release id, type its task. Each release question ends with its options, one per line ("A. ..."), and then one line
  that says to answer with a letter ("Your answer can only include one of options A, B, C or D."): q is the text before
  the first option line, opts [["A", text], ...], and the closing line is dropped, since question_text adds the
  pipeline's own. The raw fields are kept: img_type, task, source, instruction (the dropped line).
- img/spar/<id>_<k>.<jpg|png>: the views, the bytes stored in the parquet, unchanged.
- sample_ids.json, build_info.json (source, counts, chance), README.md.
Option letters must run A, B, ... on consecutive lines, at most one line may follow them, and the answer must be one of
them, or the build stops with the offending ids.
"""
import argparse
import collections
import glob
import hashlib
import io
import json
import re
from pathlib import Path

REPO = "jasonzhango/SPAR-Bench-Tiny"
REVISION = "ae9bbc5297fd277123c42b0628d1f40bf72f89f1"  # HF api, lastModified 2025-04-01
OPTION = re.compile(r"^([A-Z])\.\s*(.*\S)?\s*$")


def split_options(text):
    """'stem\\nA. x\\nB. y\\nYour answer ...' -> (stem, [['A', 'x'], ['B', 'y']], closing line) or None."""
    lines = text.strip().split("\n")
    first = next((i for i, line in enumerate(lines) if OPTION.match(line.strip())), None)
    if first is None:
        return None
    opts, i = [], first
    while i < len(lines) and (m := OPTION.match(lines[i].strip())) and m.group(1) == chr(ord("A") + len(opts)):
        opts.append([m.group(1), (m.group(2) or "").strip()])
        i += 1
    rest = [line.strip() for line in lines[i:] if line.strip()]
    stem = "\n".join(lines[:first]).strip()
    if len(opts) < 2 or len(rest) > 1 or not stem or any(not t for _, t in opts):
        return None
    return stem, opts, rest[0] if rest else ""


def image_ext(b):
    return "jpg" if b[:3] == b"\xff\xd8\xff" else "png" if b[:8] == b"\x89PNG\r\n\x1a\n" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help=f"local copy of {REPO} at revision {REVISION}")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; pick a new directory")
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(args.src / "data" / "*.parquet")))
    if not files:
        raise SystemExit(f"no data/*.parquet under {args.src}")
    rows = [r for f in files for r in pq.read_table(f).to_pylist()]
    items, bad, images = [], collections.defaultdict(list), []
    for r in rows:
        if r["format_type"] != "select":
            continue
        qid = str(r["id"])
        parsed = split_options(r["question"])
        if parsed is None:
            bad["options not parsable"].append(qid)
            continue
        stem, opts, closing = parsed
        if r["answer"] not in [k for k, _ in opts]:
            bad["answer not among the options"].append(f"{qid} ({r['answer']!r})")
            continue
        imgs = []
        for k, im in enumerate(r["image"]):
            b = im["bytes"] if isinstance(im, dict) else im
            ext = image_ext(b or b"")
            if ext is None:
                bad["view is not a JPEG or PNG"].append(f"{qid}/{k}")
                break
            dst = f"img/spar/{qid}_{k}.{ext}"
            images.append((dst, b))
            imgs.append(dst)
        else:
            items.append({"type": r["task"], "id": qid, "q": stem, "opts": opts, "ans": r["answer"], "imgs": imgs,
                          "img_type": r["img_type"], "task": r["task"], "source": r["source"], "instruction": closing})
    dupes = [i for i, c in collections.Counter(it["id"] for it in items).items() if c > 1]
    if dupes:
        bad["duplicate id"] += dupes
    if bad:
        for why, which in bad.items():
            print(f"FAIL {why}: {len(which)}, e.g. {which[:5]}")
        raise SystemExit(f"{sum(map(len, bad.values()))} problems; nothing written")

    from PIL import Image

    out, sizes = args.out, collections.Counter()
    (out / "img" / "spar").mkdir(parents=True)
    for dst, b in images:
        with open(out / dst, "xb") as f:  # x: never overwrite
            f.write(b)
        with Image.open(io.BytesIO(b)) as im:
            im.load()
            sizes[f"{im.size[0]}x{im.size[1]} {im.mode}"] += 1
    samples = out / "samples_spar_full.json"
    samples.write_text(json.dumps({"spar": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    C = collections.Counter
    hist = {"types": dict(sorted(C(it["type"] for it in items).items())),
            "img_types": dict(sorted(C(it["img_type"] for it in items).items())),
            "views_per_question": dict(sorted(C(len(it["imgs"]) for it in items).items())),
            "options_per_question": dict(sorted(C(len(it["opts"]) for it in items).items())),
            "answer_letters": dict(sorted(C(it["ans"] for it in items).items())),
            "closing_lines": dict(C(it["instruction"] for it in items).most_common(5)),
            "image_sizes": dict(sizes.most_common())}
    chance = sum(1 / len(it["opts"]) for it in items) / len(items)
    info = {"source": {"repo": REPO, "repo_type": "dataset", "revision": REVISION, "paper": "arXiv 2503.22976",
                       "files": [Path(f).name for f in files]},
            "rows": len(rows), "questions": len(items), "images": len(images), "chance": round(chance, 4),
            "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(), **hist}
    (out / "build_info.json").write_text(json.dumps(info, indent=1))
    (out / "sample_ids.json").write_text(json.dumps({"source": f"{REPO}@{REVISION}",
                                                     "ids": {"spar": [it["id"] for it in items]}}, indent=1))
    (out / "README.md").write_text(
        f"# SPAR-Bench-Tiny (multiple choice)\n\nBuilt with `data/build_spar_tiny.py` from the Hugging Face dataset {REPO}, "
        f"revision {REVISION}.\n\n- `samples_spar_full.json`: the {len(items)} multiple-choice questions of its {len(rows)} "
        f"rows ({', '.join(f'{v} {k}' for k, v in hist['img_types'].items())}); options split out of the question text, "
        f"the closing answer-format line dropped\n- `img/spar/`: the {len(images)} views, bytes as stored\n"
        f"- `build_info.json`: counts; `sample_ids.json`: the ids\n")
    print(f"spar: {len(items)} questions of {len(rows)} rows, {len(images)} images, chance {100 * chance:.1f}%")
    for k in ("types", "img_types", "views_per_question", "options_per_question", "answer_letters", "closing_lines",
              "image_sizes"):
        print(f"  {k}: {hist[k]}")


if __name__ == "__main__":
    main()
