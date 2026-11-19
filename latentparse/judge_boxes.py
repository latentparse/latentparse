#!/usr/bin/env python3
"""Do the boxes land on their objects? Evaluation only; nothing here reaches an answering model.

Usage: python judge_boxes.py --sample DIR --model QWEN_DIR RUN_DIR [RUN_DIR ...]

For every transcript in each RUN_DIR (an agent.py --boxes run, Agent-Directed or Fixed-Sequence), each box is drawn alone,
unlabelled, in red on its photo, and Qwen (thinking off, reply pre-filled "Answer:") is asked whether the named object
is inside it ("all or most of it"). Two sets of boxes are judged:
  kept  the boxes the analyst gave in "regions" (what the answering model sees; Agent-Directed and Fixed-Sequence mode)
  e1    every box the fixed tools produced (log["located"], fixed mode only; with --all-tools, the rule box per
        object): the same named objects under every condition, so Correct, Wrong and Noise compare like for like
  e3_high, e3_medium  --latents runs (both modes): the consensus box of every object locate() puts at that tier
        (log["located3"])
  kept_random, e1_random, e3_*_random  for each box above, a box of the same size at a random place in the same photo
        (seeded by question, object and set): the chance level for boxes of that size
Writes RUN_DIR/box_judge.jsonl (one row per box) and RUN_DIR/box_judge.json (per set: boxes, hit rate and mean box
area, overall and per benchmark; larger boxes pass more easily, so compare areas too). Under --wrong-latent or
--noise-latent the boxes carry no information about the photo, which gives a latent-free control level.
"""
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "direct_input"))
from agent import draw_boxes, valid_regions  # noqa: E402
from answer_qwen import ANSWER_PREFIX  # noqa: E402
from common import load_sample  # noqa: E402

JUDGE = "Look at the red box drawn on the photo. Is {obj} inside the red box (all or most of it)? Answer yes or no."


def judge(model, processor, photo, box, obj):
    img = draw_boxes(photo, [{"label": "", "box": box}])
    content = [{"type": "image", "image": img}, {"type": "text", "text": JUDGE.format(obj=obj)}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False) + ANSWER_PREFIX
    inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    reply = processor.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    return reply, True if reply.startswith("yes") else False if reply.startswith("no") else None


def size_matched(box, key):
    """A box of the same width and height at a seeded random place in the photo."""
    rng = random.Random(key)
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    return [round(x0, 4), round(y0, 4), round(x0 + w, 4), round(y0 + h, 4)]


def boxes_of(lg, n_views):
    """(set, label, view, box) for the analyst's kept boxes and, in fixed mode, every E1 box, each followed by a
    size-matched random box."""
    sets = [("kept", r) for r in valid_regions((lg.get("final_findings") or {}).get("regions"), n_views)]
    for f in lg.get("located") or []:
        sets += [("e1", r) for r in valid_regions([{"label": f.get("object", ""), "view": f.get("view", 0),
                                                    "box": f.get("box")}], n_views)]
    for f in lg.get("located3") or []:  # --latents: consensus boxes, by locate tier
        sets += [(f"e3_{f.get('tier', 'medium')}", r) for r in valid_regions(
            [{"label": f.get("object", ""), "view": f.get("view", 0), "box": f.get("box")}], n_views)]
    for s, r in sets:
        if r["label"]:
            yield s, r["label"], r["view"], r["box"]
            yield s + "_random", r["label"], r["view"], size_matched(r["box"], f"{lg['bench']}/{lg['id']}/{s}/{r['label']}")


def summary(rows):
    out = {}
    for s in ("kept", "e1", "e3_high", "e3_medium", "kept_random", "e1_random", "e3_high_random", "e3_medium_random"):
        by = collections.defaultdict(list)
        for r in rows:
            if r["set"] == s:
                by["all"].append(r)
                by[r["bench"]].append(r)
        for k, rs in sorted(by.items()):
            judged = [r["hit"] for r in rs if r["hit"] is not None]
            out.setdefault(s, {})[k] = {
                "questions": len({(r["bench"], r["id"]) for r in rs}), "boxes": len(rs), "unparsed": len(rs) - len(judged),
                "hit_rate": round(sum(judged) / len(judged), 4) if judged else None,
                "mean_area": round(sum((r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) for r in rs) / len(rs), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    images = {(it["bench"], it["id"]): it["images"] for it in load_sample(args.sample)}

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=os.environ.get("DEVICE_MAP", "cuda"),
                                                        attn_implementation="sdpa").eval()
    for run in map(Path, args.runs):
        rows, done = [], set()
        out = run / "box_judge.jsonl"
        if out.exists():
            parsed = []
            for line in out.read_text().splitlines():
                try:
                    parsed.append(json.loads(line))
                except ValueError:
                    pass
            done = {tuple(r["done"]) for r in parsed if "done" in r}  # questions whose boxes were all written
            rows = [r for r in parsed if "done" not in r and (r["bench"], r["id"]) in done]
            out.write_text("".join(json.dumps(r) + "\n" for r in parsed if "done" in r or (r["bench"], r["id"]) in done))
        with open(out, "a") as f:
            for p in sorted((run / "transcripts").glob("*.json")):
                lg = json.loads(p.read_text())
                key = (lg["bench"], str(lg["id"]))
                if key in done or key not in images or "error" in lg:  # errored questions get redone first
                    continue
                imgs = images[key]
                for s, label, view, box in boxes_of(lg, len(imgs)):
                    reply, hit = judge(model, processor, Image.open(imgs[view]).convert("RGB"), box, label)
                    row = {"set": s, "bench": lg["bench"], "id": str(lg["id"]), "label": label, "view": view,
                           "box": box, "reply": reply, "hit": hit}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                f.write(json.dumps({"done": list(key)}) + "\n")
                f.flush()
        stats = summary(rows)
        (run / "box_judge.json").write_text(json.dumps(stats, indent=1))
        print(run.name, json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
