#!/usr/bin/env python3
"""Build the MV-RoboBench sample.

    python build_mv_robobench.py SRC OUT_DIR [--all-tasks] [--pilot N]

SRC is a local copy of the Hugging Face dataset AaronFengZY24/MV_Robobench (microsoft/MV-RoboBench, ICLR 2026),
revision REVISION below: <source>/<task>/<QA>/QA.json plus the views, source agiworld (left gripper, head and right
gripper cameras) or bridgev2 (four fixed cameras). Every question is five-way multiple choice (A-E). By default only the
spatial tasks are kept (861 questions: cross-view matching, distance judgement, 3D spatial consistency and, AgiWorld
only, viewpoint identification); --all-tasks adds the robotic ones (action planning, step execution, trajectory
selection, affordance recognition). OUT_DIR must not exist. Output, in the layout of build_mindcube_tiny.py, so
direct_input/common.load_sample reads it as benchmark "mvrb":
- samples_mvrb_full.json: {"mvrb": [{"type", "id", "q", "opts", "ans", "imgs", ...}]}. id is <source>-<task>-<QA>,
  type <source>/<task> with the release's spatial_/robotic_ prefix dropped. The release names each view in its
  "illustration" field and the questions refer to them by those names ("the right-gripper camera view", "view2",
  "Option A picture"), so q starts with one line per photo in the order the photos are shown ("Photo 1: left gripper
  view."), then the release's question; opts [["A", text], ...] in letter order, ans the letter. The raw fields are
  kept: source, task, kind (spatial or robotic), illustrations.
- img/mvrb/<source>/<task>/<QA>/<file>: each view, copied byte for byte.
- sample_ids.json, build_info.json (source, counts, chance), README.md, and with --pilot N pilot<N>_ids.txt: N
  questions spread evenly over the types (round-robin over shuffled types, seed 0).
A view is used only if its path is ./<name> with [A-Za-z0-9._-] characters, is a regular file and not a symlink.
The options must be A, B, ... with no gaps and the answer one of them, or the build stops with the offending ids.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

REPO = "AaronFengZY24/MV_Robobench"
REVISION = "869ea1113cf2f08498e81f57ced72597238455c0"  # HF api, lastModified 2026-02-27
SOURCES = ("agiworld", "bridgev2")
SAFE = re.compile(r"[A-Za-z0-9._-]+")


def pilot_ids(items, n, seed=0):
    """n ids spread over the types: one from each type in turn (types shuffled), each type's questions shuffled."""
    rng = random.Random(seed)
    by_type = collections.defaultdict(list)
    for it in items:
        by_type[it["type"]].append(it["id"])
    queues = [rng.sample(v, len(v)) for _, v in sorted(by_type.items())]
    rng.shuffle(queues)
    chosen = []
    while len(chosen) < n and any(queues):
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop())
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help=f"local copy of {REPO} at revision {REVISION}")
    ap.add_argument("out", type=Path)
    ap.add_argument("--all-tasks", action="store_true", help="keep the robotic tasks too")
    ap.add_argument("--pilot", type=int, help="also write pilot<N>_ids.txt, N questions spread over the types")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; pick a new directory")

    items, bad, copies = [], collections.defaultdict(list), []
    for qa_json in sorted(args.src.glob("*/*/*/QA.json")):
        qdir = qa_json.parent
        source, task, qa = qdir.parts[-3:]
        qid = f"{source}-{task}-{qa}"
        if source not in SOURCES or not all(SAFE.fullmatch(p) for p in (source, task, qa)):
            bad["unexpected folder"].append(str(qdir))
            continue
        kind = task.split("_", 1)[0]
        if kind not in ("spatial", "robotic"):
            bad["task is neither spatial_ nor robotic_"].append(qid)
            continue
        if kind == "robotic" and not args.all_tasks:
            continue
        r = json.loads(qa_json.read_text(encoding="utf-8"))
        opts = r.get("options")
        if not (isinstance(opts, dict) and sorted(opts) == [chr(ord("A") + i) for i in range(len(opts))] and len(opts) >= 2
                and all(isinstance(v, str) and v.strip() for v in opts.values())):
            bad["options are not A, B, ... with text"].append(qid)
            continue
        if r.get("correct") not in opts:
            bad["answer not among the options"].append(qid)
            continue
        if not (isinstance(r.get("question"), str) and r["question"].strip()):
            bad["no question"].append(qid)
            continue
        imgs, labels = [], []
        for k, v in enumerate(r.get("images") or [], 1):
            name = str(v.get("path", "")).removeprefix("./")
            f = qdir / name
            if not SAFE.fullmatch(name) or f.is_symlink() or not f.is_file():
                bad["view path not ./<name> or not a regular file"].append(f"{qid}: {v.get('path')!r}")
                break
            dst = "/".join(("img", "mvrb", source, task, qa, name))
            copies.append((f, dst))
            imgs.append(dst)
            labels.append(f"Photo {k}: {' '.join(str(v.get('illustration') or name).split())}.")
        else:
            if not imgs:
                bad["no views"].append(qid)
                continue
            items.append({"type": f"{source}/{task.split('_', 1)[1]}", "id": qid,
                          "q": "\n".join(labels) + "\n" + r["question"].strip(),
                          "opts": [[k, opts[k].strip()] for k in sorted(opts)], "ans": r["correct"], "imgs": imgs,
                          "source": source, "task": task, "kind": kind,
                          "illustrations": [v.get("illustration") for v in r["images"]]})
    if bad:
        for why, which in bad.items():
            print(f"FAIL {why}: {len(which)}, e.g. {which[:5]}")
        raise SystemExit(f"{sum(map(len, bad.values()))} problems; nothing written")
    if not items:
        raise SystemExit(f"no questions under {args.src}")

    from PIL import Image

    out, total, sizes = args.out, 0, collections.Counter()
    for f, dst in copies:
        target = out / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "rb") as src, open(target, "xb") as dest:  # x: never overwrite
            shutil.copyfileobj(src, dest)
        total += target.stat().st_size
        with Image.open(target) as im:
            im.verify()
        with Image.open(target) as im:
            im.load()
            sizes[f"{im.size[0]}x{im.size[1]} {im.mode}"] += 1

    samples = out / "samples_mvrb_full.json"
    samples.write_text(json.dumps({"mvrb": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    C = collections.Counter
    hist = {"types": dict(sorted(C(it["type"] for it in items).items())),
            "views_per_question": dict(sorted(C(len(it["imgs"]) for it in items).items())),
            "options_per_question": dict(sorted(C(len(it["opts"]) for it in items).items())),
            "answer_letters": dict(sorted(C(it["ans"] for it in items).items())),
            "image_sizes": dict(sizes.most_common())}
    chance = sum(1 / len(it["opts"]) for it in items) / len(items)
    info = {"source": {"repo": REPO, "repo_type": "dataset", "revision": REVISION, "paper": "arXiv 2510.19400"},
            "tasks": "all" if args.all_tasks else "spatial", "questions": len(items), "images": len(copies),
            "image_bytes": total, "chance": round(chance, 4),
            "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(), **hist}
    (out / "build_info.json").write_text(json.dumps(info, indent=1))
    (out / "sample_ids.json").write_text(json.dumps({"source": f"{REPO}@{REVISION}",
                                                     "ids": {"mvrb": [it["id"] for it in items]}}, indent=1))
    if args.pilot:
        (out / f"pilot{args.pilot}_ids.txt").write_text("\n".join(pilot_ids(items, args.pilot)) + "\n")
    (out / "README.md").write_text(
        f"# MV-RoboBench ({info['tasks']} tasks)\n\nBuilt with `data/build_mv_robobench.py` from the Hugging Face dataset "
        f"{REPO}, revision {REVISION}.\n\n- `samples_mvrb_full.json`: {len(items)} five-way questions, "
        f"{', '.join(f'{v} with {k} views' for k, v in hist['views_per_question'].items())}; each question starts with "
        f"one line naming each photo\n- `img/mvrb/<source>/<task>/<QA>/`: the {len(copies)} views, byte for byte\n"
        f"- `build_info.json`: counts; `sample_ids.json`: the ids\n")
    print(f"mvrb: {len(items)} questions, {len(copies)} images ({total / 2 ** 20:.1f} MB), chance {100 * chance:.1f}%")
    for k in ("types", "views_per_question", "options_per_question", "answer_letters", "image_sizes"):
        print(f"  {k}: {hist[k]}")


if __name__ == "__main__":
    main()
