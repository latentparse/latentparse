#!/usr/bin/env python3
"""CPU tests of the tools in the sandbox: python3 test_tools.py [OUTDIR]

No torch, VLM or CLIP weights: CLIP's text tower is blend_synth.FakeClip and the sandbox runs in-process exactly as
sandbox._worker runs code (stdout captured, tracebacks into the output, 2500-character cap). Synthetic scenes
(blend_synth.py) have planted objects and known camera motion. Exits 1 on any failure.
  T1 Random Noise: no locate tier high or medium (<= 5% medium), no motion direction, no track "yes", for many seeds
  T2 planted objects found at tier high with the right box; an unplanted phrase is never high; the un-negated map fails
  T3 motion: pan left / right, zoom, static camera; pt_agree high when VGGT and DINOv3 see the same scene
  T4 a broken DINOv3 (noise, or another scene moving the other way) cannot overrule VGGT, and locate drops below high
  T5 Mismatched features (every latent from another scene): runs; confident but misplaced (undetectable by design)
  T6 every encoder subset x 1/3/5/8 photos: every Fixed-Sequence step runs without a traceback, each under the 2500
     cap, the evidence under 16000 characters; F holds only the encoders in use; no show(); timing
  T9 calib_blend fits thresholds on synthetic scenes
"""
import contextlib
import io
import json
import re
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy 2.0 + Accelerate on macOS: spurious matmul warnings
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "direct_input"))
args_in = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(args_in[0] if args_in else "/tmp/test_tools")
OUT.mkdir(parents=True, exist_ok=True)

# torch stub: agent.py and answer_qwen.py import torch at module level but these tests never call a model
torch = types.ModuleType("torch")
torch.bfloat16 = "bf16"
torch.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
torch.inference_mode = torch.no_grad = contextlib.nullcontext
torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
torch.nn = types.SimpleNamespace(Module=object, functional=types.SimpleNamespace(pad=None))
torch.Tensor = type("Tensor", (), {})  # not object: matplotlib's unit registry would take every value for a tensor
sys.modules.setdefault("torch", torch)

import sandbox as SB  # noqa: E402
from blend_synth import FakeClip, Scene, write_question  # noqa: E402
from sandbox import BLEND_TOOLS, TOOLS, Helpers  # noqa: E402

# The tests check the logic at the provisional thresholds the synthetic scenes were made for. The frozen values that
# calib_blend.py sets on c1 may differ; they are checked on real latents by the calibration itself, not here.
PROVISIONAL = dict(SUPPORT_Z=2.0, CORE_FRAC=0.04, KEEP_MIN=0.5, IOU_GG=0.3, AREA_MIN=0.02, AREA_MAX=0.40, PEAK_HIGH=3.5,
                   PEAK_MED=2.5, STANDOUT_HIGH=1.0, STANDOUT_MED=0.5, MIN_MATCHES=30, MIN_INLIERS=0.10, MOVING_MIN=0.30,
                   SAME_WAY=0.60, TRACK_MIN=5)
for _k, _x in PROVISIONAL.items():
    setattr(SB, _k, _x)
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


class LocalSandbox:
    """sandbox.Sandbox in-process: the same load / run contract and output handling as sandbox._worker."""

    def __init__(self, clip_text, helpers_cls=Helpers, max_output=SB.MAX_OUTPUT):
        self.h, self.max_output = helpers_cls(clip_text), max_output

    def load(self, q):
        self.h.load(q)
        self.ns = self.h.namespace()
        return "ok"

    def run(self, code):
        import traceback

        self.h.shows = []
        out = io.StringIO()
        np.set_printoptions(precision=3, suppress=True, threshold=60, edgeitems=3)
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(code, "<analyst>", "exec"), self.ns)
        except Exception:
            out.write("\n" + traceback.format_exc(limit=2))
        text = out.getvalue()
        if len(text) > self.max_output:
            text = text[:self.max_output] + f"\n... [output cut at {self.max_output} characters; print summaries, not arrays]"
        return text, list(self.h.shows)


OBJS = {"the red car": (0.55, 0.55, 0.80, 0.80), "a wooden chair": (0.15, 0.30, 0.32, 0.62)}
ALL3 = ["clip", "vggt", "dinov3"]


def helpers(images, latents, clip, use=ALL3, noise=None, flip=True):
    h = Helpers(clip)
    h.load({"images": images, "latents": latents, "blind": True, "flip_text_map": flip, "noise": noise,
            "use": list(use), "no_show": True})
    return h


def box_iou(a, b):
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u else 0.0


def planted_box(rect, pan):
    x0, y0, x1, y1 = rect
    return [max(0, x0 - pan), y0, min(1, x1 - pan), y1]


# ---------------------------------------------------------------------------------------------------------- T1 noise
print("\n== T1 Noise control")
tiers, dirs, tracks, tier_med = [], [], [], 0
for seed in range(20):
    for nv in (1, 3):
        sc = Scene(seed, OBJS)
        imgs, lat, clip = write_question(Helpers, OUT, f"t1_{seed}_{nv}", sc, [(640, 480)] * nv, [0.12 * v for v in range(nv)])
        h = helpers(imgs, lat, clip, noise=f"3dsr/t1_{seed}_{nv}")
        found = h.locate_all(list(OBJS) + ["a lamp"])
        tiers += [r["tier"] for r in found.values()]
        for a in range(nv):
            for b in range(nv):
                if a != b:
                    m = h.motion(a, b)
                    dirs += [m["x"], m["y"], m["zoom"]]
                    for r in found.values():
                        if r["tier"] != "none" and r["view"] == a:
                            tracks.append(h.track(r, b)["found"])
n = len(tiers)
check(tiers.count("high") == 0 and tiers.count("medium") <= 0.05 * n,
      f"noise: no locate tier high, <= 5% medium ({dict((t, tiers.count(t)) for t in SB.TIERS)} of {n})")
check(all(d == "unreliable" for d in dirs), f"noise: every motion direction is 'unreliable' ({len(dirs)} checked, "
      f"others: {sorted(set(d for d in dirs if d != 'unreliable'))})")
check(not any(t.startswith("yes") for t in tracks), f"noise: track never 'yes' ({len(tracks)} checked: "
      f"{ {t: tracks.count(t) for t in set(tracks)} })")
a, b = SB.load_latent(lat["dinov3"], "3dsr/q/dinov3"), SB.load_latent(lat["dinov3"], "3dsr/q/vggt")
check(not np.allclose(a, b) and np.allclose(a, SB.load_latent(lat["dinov3"], "3dsr/q/dinov3")),
      "noise: seeded per question and latent (same key -> same noise, other key -> other noise)")
h = helpers(imgs, lat, clip, noise="3dsr/t1_seeds")
lay = h.layout(0)
check(max(lay["cka"].values()) <= 0.2, f"noise: layout CKA near 0 ({lay['cka']})")

# -------------------------------------------------------------------------------------------------------- T2 planted
print("\n== T2 planted objects")
UNPLANTED = ["a lamp", "the blue door", "a tree", "a cup"]
for nv in (1, 3):
    pans = [0.12 * v for v in range(nv)]
    planted, unplanted, ious, unflipped = [], [], [], []
    for seed in range(10):
        sc = Scene(30 + seed, OBJS)
        imgs, lat, clip = write_question(Helpers, OUT, f"t2_{nv}_{seed}", sc, [(640, 480)] * nv, pans)
        found = helpers(imgs, lat, clip).locate_all(list(OBJS) + UNPLANTED)
        for o, rect in OBJS.items():
            planted.append(found[o]["tier"])
            ious.append(box_iou(found[o].get("box", [0, 0, 0, 0]), planted_box(rect, pans[found[o]["view"]])))
        unplanted += [found[o]["tier"] for o in UNPLANTED]
        if seed < 3:
            unflipped += [helpers(imgs, lat, clip, flip=False).locate(o)["tier"] for o in OBJS]
    check(planted.count("high") == len(planted) and min(ious) >= 0.5,
          f"{nv} photo(s): planted objects all tier high ({planted.count('high')}/{len(planted)}), box IoU with the planted "
          f"box >= 0.5 (min {min(ious):.2f}, mean {np.mean(ious):.2f})")
    check(unplanted.count("high") <= 0.05 * len(unplanted),
          f"{nv} photo(s): unplanted phrases rarely high ({ {t: unplanted.count(t) for t in SB.TIERS} }); agreement alone "
          f"cannot tell identity, the CLIP peak and standout bars do")
    check("high" not in unflipped, f"{nv} photo(s): the un-negated map does not find them ({unflipped})")
    if nv == 1:
        h = helpers(imgs, lat, clip)
        rel = h.relate(h.locate("the red car"), h.locate("a wooden chair"))
        check(rel["dx"] < -0.2 and rel["dy"] < -0.05 and not rel["touch"], f"relate: chair left of and above the car ({rel})")
sc = Scene(8, {"the red car": (0.5, 0.5, 0.75, 0.8)})
imgs, lat, clip = write_question(Helpers, OUT, "t2_same", sc, [(640, 480)], [0.0])
clip.plant("a car", sc.clip_dir("the red car"))
found = helpers(imgs, lat, clip).locate_all(["the red car", "a car"])
check(sum("same_as" in r for r in found.values()) == 1 and sorted(r["tier"] for r in found.values()) == ["high", "low"],
      f"locate_all: two names on one region -> one keeps it, the other gets same_as and tier low ({found})")

# --------------------------------------------------------------------------------------------------------- T3 motion
print("\n== T3 motion")
sc = Scene(3, OBJS)
imgs, lat, clip = write_question(Helpers, OUT, "t3_pan", sc, [(640, 480)] * 3, [0.0, 0.12, 0.24])
h = helpers(imgs, lat, clip)
m = h.motion(0, 1)
check(m["x"] == "left 3/3" and m["y"] == "none 3/3" and m["pt_agree"] >= 0.5,
      f"pan right: content moves left in all three ({m['x']}, {m['y']}, pt_agree {m['pt_agree']})")
check(h.motion(1, 0)["x"] == "right 3/3", "reverse pair: right 3/3")
r = h.locate("the red car", v=0)
t = h.track(r, 1)
check(t["found"] == "yes" and abs(t["shift"][0] + 0.12) < 0.04 and abs(t["beyond_camera"][0]) < 0.04,
      f"track a static object: found yes, shift about -0.12, beyond_camera about 0 ({t})")
imgs, lat, clip = write_question(Helpers, OUT, "t3_zoom", sc, [(640, 480)] * 2, [0.0, 0.0], zooms=[1.0, 1.25])
m = helpers(imgs, lat, clip).motion(0, 1)
check(m["zoom"] == "grows 3/3" and abs(m["vggt"]["scale"] - 1.25) < 0.08, f"zoom in: grows 3/3, scale {m['vggt']['scale']}")
imgs, lat, clip = write_question(Helpers, OUT, "t3_static", sc, [(640, 480)] * 2, [0.0, 0.0])
m = helpers(imgs, lat, clip).motion(0, 1)
check(m["x"] == "none 3/3" and m["y"] == "none 3/3" and m["reliable"] == "vggt dinov3 clip",
      f"static camera: none 3/3, all reliable ({m['x']}, {m['reliable']})")
sc_obj = Scene(4, {"the red car": (0.3, 0.5, 0.5, 0.75)})
imgs, lat, clip = write_question(Helpers, OUT, "t3_obj", sc_obj, [(640, 480)] * 2, [0.0, 0.0])
sc_moved = Scene(4, {"the red car": (0.5, 0.5, 0.7, 0.75)})  # same scene, the car moved right by 0.2
for k in ("vggt", "dinov3", "clip"):
    x = np.load(lat[k])
    _, lat2, _ = write_question(Helpers, OUT, "t3_obj2", sc_moved, [(640, 480)] * 2, [0.0, 0.0], models=(k,))
    x[1] = np.load(lat2[k])[1]
    np.save(lat[k], x)
h = helpers(imgs, lat, clip)
t = h.track(h.locate("the red car", v=0), 1)
check(t["found"] == "yes" and t["beyond_camera"][0] > 0.12, f"an object that moved: beyond_camera x {t.get('beyond_camera')}")

# -------------------------------------------------------------------------------------------------- T4 broken DINOv3
print("\n== T4 a broken DINOv3")
sc = Scene(5, OBJS)
imgs, lat, clip = write_question(Helpers, OUT, "t4", sc, [(640, 480)] * 2, [0.0, 0.12])
d = np.load(lat["dinov3"])
np.save(OUT / "t4_dino_noise.npy", np.random.default_rng(0).standard_normal(d.shape).astype(np.float32))
lat_n = dict(lat, dinov3=str(OUT / "t4_dino_noise.npy"))
h = helpers(imgs, lat_n, clip)
m = h.motion(0, 1)
check(m["x"].startswith("left 2/2") and "dinov3" not in m["reliable"], f"DINOv3 = noise: VGGT and CLIP still give left ({m['x']})")
check(all(h.locate(o)["tier"] != "high" for o in OBJS), "DINOv3 = noise: no tier high (VGGT and DINOv3 must both keep)")
other = Scene(6, OBJS)
_, lat_o, _ = write_question(Helpers, OUT, "t4_other", other, [(640, 480)] * 2, [0.12, 0.0], models=("dinov3",))
m = helpers(imgs, dict(lat, dinov3=lat_o["dinov3"]), clip).motion(0, 1)
check(m["x"].startswith("left 2/3") and "dinov3 right" in m["x"] and m["pt_agree"] < 0.1,
      f"DINOv3 from a scene moving the other way: majority kept, dissent shown ({m['x']}, pt_agree {m['pt_agree']})")
np.save(OUT / "t4_clip_noise.npy", np.random.default_rng(1).standard_normal(np.load(lat["clip"]).shape).astype(np.float32))
m = helpers(imgs, dict(lat_n, clip=str(OUT / "t4_clip_noise.npy")), clip).motion(0, 1)
check(m["x"] == "left (only vggt)", f"only VGGT reliable: 'left (only vggt)' ({m['x']})")

# ---------------------------------------------------------------------------------------------------------- T5 wrong
print("\n== T5 Wrong control")
donor = Scene(11, {"the red car": (0.1, 0.1, 0.3, 0.3), "a wooden chair": (0.6, 0.6, 0.9, 0.9)})
imgs_d, lat_d, clip_d = write_question(Helpers, OUT, "t5_donor", donor, [(640, 480)] * 2, [0.0, 0.12])
h = helpers(imgs, lat_d, clip_d)
r = h.locate("the red car")
check(r["tier"] in SB.TIERS and box_iou(r.get("box", [0, 0, 0, 0]), planted_box(OBJS["the red car"], 0)) < 0.2,
      f"Wrong: runs; the box is where the donor's car is, not this scene's (tier {r['tier']}): agreement cannot see it")

# ----------------------------------------------------------------------------------------- T6 subsets x photos, length
print("\n== T6 every subset and photo count: fixed steps")
import agent  # noqa: E402  (with the torch stub)

SUBSETS = [ALL3, ["clip", "vggt"], ["vggt"], ["vggt", "dinov3"], ["clip", "dinov3"], ["dinov3"]]
LAYOUTS = {1: [(640, 480)], 3: [(640, 480), (480, 640), (640, 480)], 5: [(640, 480)] * 5, 8: [(640, 480)] * 8}
QUESTIONS = {}
for nv, sizes in LAYOUTS.items():
    for kind in ("random", "planted"):
        objs = {"the red car": (0.55, 0.55, 0.80, 0.80), "a wooden chair": (0.15, 0.30, 0.32, 0.62),
                "the lamp": (0.40, 0.10, 0.50, 0.40), "a dog": (0.70, 0.20, 0.85, 0.40)}
        sc = Scene(20 + nv, objs if kind == "planted" else {})
        imgs, lat, clip = write_question(Helpers, OUT, f"t6_{kind}_{nv}", sc, sizes, [0.05 * v for v in range(nv)])
        if kind == "planted":
            for o in objs:
                clip.plant(o, sc.clip_dir(o))
        QUESTIONS[(kind, nv)] = (imgs, lat, clip, list(objs))
worst = (0, None)
for use in SUBSETS:
    for (kind, nv), (imgs, lat, clip, objs) in QUESTIONS.items():
        sb = LocalSandbox(clip)
        sb.load({"images": imgs, "latents": lat, "blind": True, "flip_text_map": True, "use": use, "no_show": True})
        t0 = time.time()
        steps = agent.blend_steps(use, nv)
        outs = {e: sb.run(f"objects = {objs!r}\n" + c)[0] for e, c in steps.items()}
        dt = time.time() - t0
        ev = "\n\n".join(f"[{e}] ran:\nobjects = {objs!r}\n{steps[e]}\n[{e}] output:\n{o}" for e, o in outs.items())
        tag = f"{'+'.join(use)} {kind} {nv} photo(s)"
        bad = [e for e, o in outs.items() if "Traceback" in o or "output cut" in o]
        if bad:
            check(False, f"{tag}: {bad} -> {outs[bad[0]][-600:]}")
        worst = max(worst, (len(ev), tag), (max(len(o) for o in outs.values()), tag + " (one step)"))
        if len(ev) > 16000 or dt > 30:
            check(False, f"{tag}: evidence {len(ev)} chars, {dt:.1f} s")
check(not [f for f in failures if "photo(s): E" in f or "Traceback" in f], "every subset x photo count: no traceback, no cut")
print(f"   longest evidence / step: {worst}")
imgs, lat, clip, objs = QUESTIONS[("planted", 8)]
sb = LocalSandbox(clip)
sb.load({"images": imgs, "latents": lat, "blind": True, "flip_text_map": True, "use": ALL3, "no_show": True})
steps = agent.blend_steps(ALL3, 8)
outs = {e: sb.run(f"objects = {objs!r}\n" + c)[0] for e, c in steps.items()}
sizes = {e: len(o) for e, o in outs.items()}
ev = "\n\n".join(f"[{e}] ran:\nobjects = {objs!r}\n{steps[e]}\n[{e}] output:\n{o}" for e, o in outs.items())
check(max(sizes.values()) <= 2500 and len(ev) <= 16000, f"8 photos, 4 planted objects: step outputs {sizes}, evidence {len(ev)}")
(OUT / "evidence_8views_planted.txt").write_text(ev)
imgs, lat, clip, objs = QUESTIONS[("planted", 1)]
sb = LocalSandbox(clip)
sb.load({"images": imgs, "latents": lat, "blind": True, "flip_text_map": True, "use": ALL3, "no_show": True})
steps = agent.blend_steps(ALL3, 1)
(OUT / "evidence_1view_planted.txt").write_text("\n\n".join(
    f"[{e}] ran:\nobjects = {objs!r}\n{c}\n[{e}] output:\n{sb.run(f'objects = {objs!r}' + chr(10) + c)[0]}" for e, c in steps.items()))
for use in SUBSETS:
    imgs, lat, clip, objs = QUESTIONS[("random", 3)]
    sb = LocalSandbox(clip)
    sb.load({"images": imgs, "latents": lat, "blind": True, "flip_text_map": True, "use": use, "no_show": True})
    ok = sorted(sb.ns["F"]) == sorted(use) and "show" not in sb.ns and sb.ns["latents"] == [k for k in ALL3 if k in use]
    out = sb.run("text_map('a car', 0)")[0]
    check(ok and (("Traceback" in out) == ("clip" not in use)), f"{'+'.join(use)}: F holds only these, no show(), text_map "
          f"{'works' if 'clip' in use else 'refused'}")

print("\n== T9 calib_blend on synthetic scenes (hit = box IoU with the planted box >= 0.5)")
import calib_blend  # noqa: E402

rows = []
for seed in range(6):
    nv = 1 + seed % 2
    pans = [0.1 * v for v in range(nv)]
    sc = Scene(50 + seed, OBJS)
    imgs, lat, clip = write_question(Helpers, OUT, f"t9_{seed}", sc, [(640, 480)] * nv, pans)
    for cond in ("correct", "noise"):
        h = helpers(imgs, lat, clip, noise=f"3dsr/t9_{seed}" if cond == "noise" else None)
        for r in calib_blend.measure(h, list(OBJS) + UNPLANTED[:2], [1.5, 2.0], nv):
            r.update(bench="3dsr", id=str(seed), cond=cond)
            if cond == "correct" and r["kind"] == "locate" and r["box"]:
                r["hit"] = r["object"] in OBJS and box_iou(r["box"], planted_box(OBJS[r["object"]], pans[r["view"]])) >= 0.5
                r["hit_random"] = False
            rows.append(r)
per_z, mot = calib_blend.fit(rows, [1.5, 2.0], 0.65, 0.50)
check(any(r["thresholds"] for r in per_z.values()) and mot["min_inliers"] is not None and mot["noise_track_yes"] <= 0.02,
      f"calibration fit runs: {[(z, r['result']) for z, r in per_z.items()]}; motion {mot}")


print("\nFAILURES:", failures or "none")
sys.exit(1 if failures else 0)
