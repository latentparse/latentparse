#!/usr/bin/env python3
"""Sanity-check the benchmark samples.

Usage: python3 check_samples.py /path/to/benchmark_samples

For each samples_*_200.json: item count, duplicate ids, missing / unused images,
answer balance, question-type coverage, and overlap with the older subsets.
"""
import collections
import json
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")
ID_KEYS = ("id", "question_id", "qid", "index", "sample_id")
TYPE_KEYS = ("category", "question_type", "type", "qtype", "subtype", "sub_category", "task")
ANSWER_KEYS = ("answer", "gt_answer", "ground_truth", "label", "gt")


def load_items(path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if len(lists) == 1 else list(data.values())
    return data


def first_key(item, keys):
    return next((k for k in keys if isinstance(item, dict) and k in item), None)


def image_refs(item):
    refs = []

    def walk(x):
        if isinstance(x, str) and x.lower().endswith(IMG_EXT):
            refs.append(x)
        elif isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)

    walk(item)
    return refs


def resolve(root, bench, ref):
    for p in (root / ref, root / "img" / bench / Path(ref).name):
        if p.exists():
            return p
    return None


def check(root, bench):
    path = root / f"samples_{bench}_200.json"
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print("  MISSING FILE")
        return
    items = load_items(path)
    print(f"  items: {len(items)}")
    if not items:
        return
    print(f"  keys of first item: {sorted(items[0].keys()) if isinstance(items[0], dict) else type(items[0])}")

    id_key = first_key(items[0], ID_KEYS)
    ids = [str(it.get(id_key)) for it in items] if id_key else []
    if id_key:
        dupes = [i for i, c in collections.Counter(ids).items() if c > 1]
        print(f"  id field '{id_key}': {len(set(ids))} unique, {len(dupes)} duplicated {dupes[:5]}")
    else:
        print("  no id field found")

    used, missing, per_item = set(), [], []
    for it in items:
        refs = image_refs(it)
        per_item.append(len(refs))
        for r in refs:
            p = resolve(root, bench, r)
            if p:
                used.add(p.resolve())
            else:
                missing.append(r)
    print(f"  images per question: min {min(per_item)}, max {max(per_item)}, total refs {sum(per_item)}")
    print(f"  missing images: {len(missing)} {missing[:5]}")
    img_dir = root / "img" / bench
    if img_dir.exists():
        on_disk = {p.resolve() for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        print(f"  images on disk: {len(on_disk)}, unused by any question: {len(on_disk - used)}")

    ans_key = first_key(items[0], ANSWER_KEYS)
    if ans_key:
        print(f"  answers ('{ans_key}'): {dict(collections.Counter(str(it.get(ans_key)) for it in items).most_common(10))}")
    type_key = first_key(items[0], TYPE_KEYS)
    if type_key:
        types = collections.Counter(str(it.get(type_key)) for it in items)
        print(f"  question types ('{type_key}'): {len(types)} kinds")
        for t, c in types.most_common():
            print(f"    {c:4d}  {t}")
    else:
        print("  no question-type field found")

    old = root / f"samples_{bench}_all.json"
    if id_key and old.exists():
        old_ids = [str(it.get(id_key)) for it in load_items(old) if isinstance(it, dict)]
        print(f"  older subset: {len(old_ids)} items, first {len(old_ids)} of the 200 match it: {ids[:len(old_ids)] == old_ids}")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for bench in ("3dsr", "mmsi"):
        check(root, bench)
