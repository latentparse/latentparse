#!/usr/bin/env python3
"""Set the blend thresholds (sandbox.py) on a calibration sample that no reported run uses, then freeze them.

Usage: python calib_blend.py --sample DATA_FULL --latents-root LATENTS_FULL --exclude S1_DIR S2_DIR --model QWEN_DIR
                             --out DIR [--n 100 50] [--seed 7]        (--fit-only: refit from DIR/rows.jsonl)

c1 = --n[0] 3DSR and --n[1] MMSI questions of DATA_FULL that are in none of the --exclude samples (seeded). No answers
are used anywhere. Per question: objects named as in the runs (object_names.list_objects, cleaned); then, with the
latents as they are (Correct) and with Noise:
  - locate() of every object at z in --z, with the numbers its tier comes from; under Correct its consensus box and a
    size-matched random box are judged as in judge_boxes.py (Qwen sees the photo: evaluation only, nothing here reaches
    an analyst or an answering model);
  - motion() of every photo pair (at most 7), per latent n and inl;
  - track() of every object that is not tier none into the neighbouring photo.
Fit (sandbox.tier_of with other thresholds), per z:
  high:   the most objects at tier high with judged hit rate >= --hit-high (0.65) and at most 1% of Noise objects high;
  medium: then the most objects at tier medium with hit rate >= --hit-med (0.50) and at most 5% of Noise objects high
          or medium;
  MIN_INLIERS: the 99th percentile of inl under Noise + 0.02, at least 0.05, over the readouts (all latents, all
    pairs) with n >= 15, the lowest the n gate of motion() goes: with a few chance matches inl is large (2 of 5
    points always fit) but such a latent is already unreliable on n;
  PT_AGREE_HIGH: the 99th percentile of pt_agree under Noise + 0.05, within [0.1, 0.3] (its Correct median is printed);
  report the share of track "yes" under Noise (must be <= 2%), and per latent the share of Correct photo pairs on which
  it would be reliable (if VGGT is under about 0.7 there, the bar is costing the MMSI matches that gave VGGT its gain).
Writes DIR/c1_ids.json, DIR/rows.jsonl (resumable), DIR/calib.json and prints the constants to paste into sandbox.py.
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "direct_input"))
from common import load_sample  # noqa: E402

import sandbox as SB  # noqa: E402

GRID = {"KEEP_MIN": [0.3, 0.5, 0.7], "IOU_GG": [0.2, 0.3, 0.4], "PEAK_HIGH": [3.0, 3.5, 4.0],
        "STANDOUT_HIGH": [0.5, 1.0, 1.5], "PEAK_MED": [2.0, 2.5, 3.0], "STANDOUT_MED": [0.25, 0.5, 1.0]}


def pick_c1(items, exclude, n, seed):
    """c1: questions of the full benchmarks in none of the excluded samples, n[0] 3DSR and n[1] MMSI, seeded."""
    excluded = [it for d in exclude for it in load_sample(d)]
    taken = {(it["bench"], it["id"]) for it in excluded}
    # data_full holds every question of each 3DSR photo, so leave out other questions about the excluded photos too
    # (same photo, same latents)
    taken_imgs = {Path(p).name for it in excluded if it["bench"] == "3dsr" for p in it["images"]}
    rng = random.Random(seed)
    out = []
    for bench, k in zip(("3dsr", "mmsi"), n):
        pool = sorted((it for it in items if it["bench"] == bench and (bench, it["id"]) not in taken
                       and not (bench == "3dsr" and Path(it["images"][0]).name in taken_imgs)), key=lambda it: it["id"])
        out += rng.sample(pool, min(k, len(pool)))
    return out


def measure(h, objects, zs, nv):
    """locate numbers per object and z, motion per pair, track per object (one sandbox load)."""
    rows = []
    for z in zs:
        found = h.locate_all(objects, z=z)
        for o, r in found.items():
            rows.append({"kind": "locate", "object": o, "z": z, "view": r["view"], "box": r.get("box"), "area": r.get("area"),
                         "keep": r.get("keep", {}), "iou_gg": r.get("iou_vggt_dinov3"), "peak": r["peak"],
                         "standout": r["standout"], "same_as": r.get("same_as"), "tier": r["tier"]})
            if z == SB.SUPPORT_Z and nv > 1 and r["tier"] != "none":
                vb = r["view"] + 1 if r["view"] + 1 < nv else r["view"] - 1
                rows.append({"kind": "track", "object": o, "found": h.track(r, vb)["found"]})
    pairs = sorted(((a, b) for a in range(nv) for b in range(a + 1, nv)), key=lambda p: (p[1] - p[0], p))[:7]
    for a, b in pairs:
        m = h.motion(a, b)
        rows.append({"kind": "motion", "pair": [a, b], "x": m["x"], "y": m["y"], "pt_agree": m.get("pt_agree"),
                     **{k: {"n": m[k]["n"], "inl": m[k].get("inl")} for k in ("vggt", "dinov3", "clip") if k in m}})
    return rows


def fit(rows, zs, hit_high, hit_med):
    """Thresholds per z (see the module docstring) -> {z: {...}}, and the motion / track noise readouts."""
    loc = [r for r in rows if r["kind"] == "locate"]
    out = {}
    for z in zs:
        cor = [r for r in loc if r["z"] == z and r["cond"] == "correct" and r.get("hit") is not None]
        noi = [r for r in loc if r["z"] == z and r["cond"] == "noise"]
        best = None
        for vals in itertools.product(*GRID.values()):
            th = dict(zip(GRID, vals))

            def tiers(rs):
                return ["low" if r["same_as"] else SB.tier_of(r["keep"], r["iou_gg"], r["area"], r["peak"], r["standout"], th)
                        for r in rs]

            tc, tn = tiers(cor), tiers(noi)
            hi = [r["hit"] for r, t in zip(cor, tc) if t == "high"]
            me = [r["hit"] for r, t in zip(cor, tc) if t == "medium"]
            noise_hi = tn.count("high") / max(len(tn), 1)
            noise_any = (tn.count("high") + tn.count("medium")) / max(len(tn), 1)
            if not hi or np.mean(hi) < hit_high or noise_hi > 0.01 or (me and np.mean(me) < hit_med) or noise_any > 0.05:
                continue
            score = (len(hi), len(me))
            if best is None or score > best[0]:
                best = (score, th, {"high": len(hi), "high_hit": round(float(np.mean(hi)), 3), "medium": len(me),
                                    "medium_hit": round(float(np.mean(me)), 3) if me else None, "objects": len(cor),
                                    "noise_high": round(noise_hi, 3), "noise_high_or_medium": round(noise_any, 3)})
        rnd = [r["hit_random"] for r in cor if r.get("hit_random") is not None]
        out[z] = {"thresholds": best[1] if best else None, "result": best[2] if best else "no feasible thresholds",
                  "random_box_hit": round(float(np.mean(rnd)), 3) if rnd else None}
    inl = [r[k]["inl"] for r in rows if r["kind"] == "motion" and r["cond"] == "noise"
           for k in ("vggt", "dinov3", "clip") if k in r and r[k]["inl"] is not None and r[k]["n"] >= 15]
    trk = [r["found"] for r in rows if r["kind"] == "track" and r["cond"] == "noise"]
    pt = {c: [r["pt_agree"] for r in rows if r["kind"] == "motion" and r["cond"] == c and r.get("pt_agree") is not None]
          for c in ("correct", "noise")}
    mot = {"min_inliers": round(max(0.05, float(np.percentile(inl, 99)) + 0.02), 3) if inl else None,
           "noise_inl_p99": round(float(np.percentile(inl, 99)), 3) if inl else None,
           "pt_agree_high": round(min(0.3, max(0.1, float(np.percentile(pt["noise"], 99)) + 0.05)), 2) if pt["noise"] else None,
           "pt_agree_correct_median": round(float(np.median(pt["correct"])), 2) if pt["correct"] else None,
           "noise_track_yes": round(sum(t.startswith("yes") for t in trk) / max(len(trk), 1), 3)}
    # How often each latent would be reliable on real pairs at that MIN_INLIERS (VGGT's MMSI gain needs it to be).
    for k in ("vggt", "dinov3", "clip"):
        rs = [r[k] for r in rows if r["kind"] == "motion" and r["cond"] == "correct" and k in r and r[k]["inl"] is not None]
        if rs and mot["min_inliers"] is not None:
            mot[f"{k}_reliable_correct"] = round(float(np.mean([x["n"] >= SB.MIN_MATCHES and x["inl"] >= mot["min_inliers"]
                                                                for x in rs])), 2)
    return out, mot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--latents-root", required=True)
    ap.add_argument("--exclude", nargs="+", required=True, help="sample dirs whose questions c1 must not contain (s1, s2)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", nargs=2, type=int, default=[100, 50])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--z", nargs="+", type=float, default=[1.5, 2.0, 2.5])
    ap.add_argument("--hit-high", type=float, default=0.65)
    ap.add_argument("--hit-med", type=float, default=0.50)
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    if not args.fit_only:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from agent import Chat
        from object_names import list_objects
        from judge_boxes import judge, size_matched

        items = pick_c1(load_sample(args.sample), args.exclude, args.n, args.seed)
        (out / "c1_ids.json").write_text(json.dumps([[it["bench"], it["id"]] for it in items]))
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                            attn_implementation="sdpa").eval()
        chat, h = Chat(processor, model), SB.Helpers(SB.ClipText())
        done = {(r["bench"], r["id"]) for r in map(json.loads, rows_path.read_text().splitlines())} if rows_path.exists() else set()
        with open(rows_path, "a") as f:
            for n, it in enumerate(items, 1):
                if (it["bench"], it["id"]) in done:
                    continue
                it["latents"] = {k: str(Path(args.latents_root) / k / it["bench"] / f"{it['id']}.npy")
                                 for k in ("clip", "vggt", "dinov3")}
                objects, _ = list_objects(chat, it, 4, clean=True)
                photos = [Image.open(p).convert("RGB") for p in it["images"]]
                judged = {}
                for cond in ("correct", "noise"):
                    h.load({"images": it["images"], "latents": it["latents"], "blind": True, "flip_text_map": True,
                            "use": ["clip", "vggt", "dinov3"], "no_show": True,
                            "noise": f"{it['bench']}/{it['id']}" if cond == "noise" else None})
                    for r in measure(h, objects, args.z, len(it["images"])):
                        r.update(bench=it["bench"], id=it["id"], cond=cond)
                        if cond == "correct" and r["kind"] == "locate" and r["box"]:
                            key = (r["object"], r["view"], tuple(r["box"]))
                            if key not in judged:
                                rb = size_matched(r["box"], f"{it['bench']}/{it['id']}/calib/{r['object']}")
                                judged[key] = (judge(model, processor, photos[r["view"]], r["box"], r["object"])[1],
                                               judge(model, processor, photos[r["view"]], rb, r["object"])[1])
                            r["hit"], r["hit_random"] = judged[key]
                        f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{n}/{len(items)}] {it['bench']} {it['id']} objects={objects}", flush=True)
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    per_z, mot = fit(rows, args.z, args.hit_high, args.hit_med)
    feasible = {z: r for z, r in per_z.items() if r["thresholds"]}
    z_best = max(feasible, key=lambda z: (feasible[z]["result"]["high"], feasible[z]["result"]["medium"])) if feasible else None
    calib = {"per_z": {str(z): r for z, r in per_z.items()}, "motion": mot, "chosen_z": z_best,
             "c1": json.loads((out / "c1_ids.json").read_text()) if (out / "c1_ids.json").exists() else None}
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    print(json.dumps(calib["per_z"], indent=1), json.dumps(mot), sep="\n")
    if z_best is None:
        print("NO FEASIBLE THRESHOLDS: do not run the blend; look at calib.json (rows.jsonl has every number)")
        return
    th = feasible[z_best]["thresholds"]
    print("\n# paste into latentparse/sandbox.py")
    print(f"SUPPORT_Z = {z_best}\nKEEP_MIN = {th['KEEP_MIN']}\nIOU_GG = {th['IOU_GG']}\n"
          f"PEAK_HIGH, PEAK_MED = {th['PEAK_HIGH']}, {th['PEAK_MED']}\n"
          f"STANDOUT_HIGH, STANDOUT_MED = {th['STANDOUT_HIGH']}, {th['STANDOUT_MED']}\nMIN_INLIERS = {mot['min_inliers']}\n"
          f"PT_AGREE_HIGH = {mot['pt_agree_high']}")


if __name__ == "__main__":
    main()
