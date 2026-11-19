#!/usr/bin/env python3
"""CPU-only checks of agent.py --marks and marks.py: python3 test_marks.py [OUTDIR]

No torch (a stub), no model, no GPU: synthetic photos with drawn marks, synthetic latents (blend_synth.py), a scripted
analyst and a fake API client. Exits 1 on any failure.
  K1 MV-RoboBench: thin rectangle outlines in the exact palette colors on a noisy PNG -> one box per color and photo,
     the largest, exact to the pixel; an outline cut in two by a line of another color is one box; stray pixels, a blob
     under MIN_PIXELS, a speck under MIN_SIDE and a color off by 1 are ignored; a photo with no mark has none
  K2 SPAR points (kind "point"): filled red, dark green (0, 128, 0) and blue discs 41 px across on a JPEG -> each
     point's centre, several per color; a large saturated region, a thin saturated line, a small saturated fragment and
     a natural dark green disc are not points
  K2b SPAR boxes (kind "box"): thick red, dark green and yellow outlines on a JPEG -> one box per color, also the red one
     cut in two by a gray bar and the yellow one running off the photo; a filled saturated square is not a box; each
     kind finds nothing in the other kind's photos, and kind None finds both
  K3 any other benchmark: no marks, and no photo is opened
  K4 the analyst's first message: without marks exactly the text of the runs without marks (copied below); with marks
     that text and then the marks section
  K5 run_agent, Multi-Latent (--latents) and one-encoder (--latent) runs, Agent-Directed and Fixed-Sequence: the system
     prompt is the same with and without marks, the first message without marks is the old one and with marks gets the
     section
  K6 main() end to end over the API (fake client), blend Agent-Directed and Fixed-Sequence, on an MV-RoboBench and a SPAR sample:
     --marks logs log["marks"] and the analyst gets the section; a SPAR question naming points gets only the points, one
     naming bboxes only the boxes; a question with no mark and a run without --marks send the analyst exactly the old
     messages (--latent always injects its latent in the final answer, as an image over an API, so the one-encoder path
     is checked in K5 only: main() finds the marks before it branches)
  K7 sandbox: locate_all(phrases, v=k) locates in photo k (it raised TypeError before)
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy 2.0 + Accelerate on macOS: spurious matmul warnings
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "direct_input"))
args_in = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(args_in[0]) if args_in else Path(tempfile.mkdtemp(prefix="test_marks_"))
OUT.mkdir(parents=True, exist_ok=True)

# torch stub: agent.py and answer_qwen.py import torch at module level but nothing here calls a model
torch = types.ModuleType("torch")
torch.bfloat16 = "bf16"
torch.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
torch.inference_mode = torch.no_grad = contextlib.nullcontext
torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
torch.nn = types.SimpleNamespace(Module=object, functional=types.SimpleNamespace(pad=None))
torch.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", torch)
for _name in ("openai", "anthropic"):  # answer_gpt imports its SDK at module level; the fake client replaces it
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.SimpleNamespace(OpenAI=None, Anthropic=None, BadRequestError=Exception)

import agent  # noqa: E402
import api_chat  # noqa: E402
import marks  # noqa: E402
from blend_synth import Scene, write_question  # noqa: E402
from sandbox import Helpers  # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def outline(a, x0, y0, x1, y1, rgb, t=2):
    """A rectangle outline t pixels thick covering columns x0..x1-1 and rows y0..y1-1."""
    a[y0:y1, x0:x0 + t] = a[y0:y1, x1 - t:x1] = rgb
    a[y0:y0 + t, x0:x1] = a[y1 - t:y1, x0:x1] = rgb


def frac_box(x0, y0, x1, y1, w=640, h=480):
    return [round(x0 / w, 3), round(y0 / h, 3), round(x1 / w, 3), round(y1 / h, 3)]


# ------------------------------------------------------------------------------------------------ K1 MV-RoboBench
print("== K1 MV-RoboBench boxes (PNG)")
C = marks.BOX_COLORS
rng = np.random.default_rng(0)
mv = []
for k in range(3):
    gray = rng.integers(0, 236, (480, 640, 1))  # a noisy, unsaturated scene: no palette color
    a = np.clip(gray + rng.integers(0, 21, (480, 640, 3)), 0, 255).astype(np.uint8)
    if k == 0:
        outline(a, 397, 120, 436, 206, C["red"])
        a[420:470:7, 300:600:7] = C["red"]  # stray pixels of a mark color, none touching another
        a[50:53, 50:53] = C["yellow"]  # 9 pixels: under MIN_PIXELS
        a[0:8, 0:6] = C["yellow"]  # a 6x8 speck in the corner: under MIN_SIDE
        outline(a, 100, 300, 220, 400, (C["pink"][0] + 1, C["pink"][1], C["pink"][2]))  # off by 1: not the color
        outline(a, 450, 300, 570, 380, C["purple"])
        a[280:400, 500:502] = C["blue"]  # a blue line cutting the purple outline in two, itself 2 px wide
    elif k == 1:
        outline(a, 300, 200, 420, 330, C["red"])
        outline(a, 600, 400, 612, 412, C["red"], t=1)  # a second, smaller red outline: only the largest counts
        outline(a, 40, 300, 200, 470, C["blue"])
        outline(a, 500, 10, 630, 100, C["green"])
    p = OUT / f"mvrb_{k + 1}.png"
    Image.fromarray(a).convert("RGBA" if k == 1 else "RGB").save(p)
    mv.append(str(p))
got = marks.detect_marks(mv, "mvrb")
want = [(1, "red", frac_box(397, 120, 436, 206)), (1, "purple", frac_box(450, 300, 570, 380)),
        (2, "red", frac_box(300, 200, 420, 330)), (2, "blue", frac_box(40, 300, 200, 470)),
        (2, "green", frac_box(500, 10, 630, 100))]
check([(m["photo"], m["color"], m["box"]) for m in got] == want and all(m["kind"] == "box" for m in got),
      f"boxes found, exact: {[(m['photo'], m['color'], m['box']) for m in got]}")
check(got[0]["center"] == [round(416.5 / 640, 3), round(163 / 480, 3)], f"centre of the box: {got[0]['center']}")
check(marks.detect_marks(mv[2:], "mvrb") == [], "a photo without marks: none")

# ------------------------------------------------------------------------------------------------------- K2 SPAR
print("\n== K2 SPAR points (JPEG)")
W, H = 1296, 968
RED, GREEN, BLUE, YELLOW = (254, 0, 0), (0, 128, 0), (0, 0, 254), (255, 255, 0)  # SPAR's green is dark
POINTS = {0: [("red", RED, 531, 639), ("green", GREEN, 259, 290), ("green", GREEN, 907, 484)],
          1: [("red", RED, 1037, 194), ("blue", BLUE, 648, 436)]}


def scene():
    """A smooth, unsaturated photo with some noise."""
    yy, xx = np.mgrid[0:H, 0:W]
    base = 110 + 50 * np.sin(xx / 97.0) * np.cos(yy / 71.0)
    return np.clip(np.stack([base, base * 0.9 + 10, base * 0.8 + 20], -1) + rng.normal(0, 8, (H, W, 3)), 0, 255).astype(np.uint8)


sp = []
for k in range(2):
    a = scene()
    if k == 0:
        a[600:800, 50:350] = (220, 30, 30)  # a large saturated region: too big for a point
        a[100:104, 700:900] = RED           # a thin saturated line: not compact
        a[300:312, 1100:1112] = RED         # a small saturated fragment: under POINT_PIXELS
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    for _, rgb, cx, cy in POINTS[k]:
        d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=rgb)  # 41 px across, as SPAR's points
    if k == 0:
        d.ellipse([1130, 780, 1170, 820], fill=(60, 110, 60))  # a natural dark green: not SPAR's green
    p = OUT / f"spar_{k + 1}.jpg"
    im.save(p, quality=85)
    sp.append(str(p))
got = marks.detect_marks(sp, "spar", "point")
want = [(k + 1, c, ((cx + 0.5) / W, (cy + 0.5) / H)) for k in POINTS for c, _, cx, cy in POINTS[k]]
check([(m["photo"], m["color"]) for m in got] == [(k, c) for k, c, _ in want] and all(m["kind"] == "point" for m in got),
      f"points found, by photo, color, left to right: {[(m['photo'], m['color'], m['center']) for m in got]}")
check(len(got) == len(want) and all(abs(m["center"][0] - x) <= 0.004 and abs(m["center"][1] - y) <= 0.004
                                    for m, (_, _, (x, y)) in zip(got, want)), "each centre within 0.004 of the drawn one")
check(all(m["box"][0] < m["center"][0] < m["box"][2] and m["box"][1] < m["center"][1] < m["box"][3]
          and m["box"][2] - m["box"][0] <= marks.MAX_POINT / W for m in got), "each point's box holds its centre and is point-sized")

print("\n== K2b SPAR boxes (JPEG)")
a = scene()
outline(a, 200, 150, 500, 450, RED, t=12)
a[100:500, 340:360] = (120, 120, 120)  # a gray bar cutting the red outline in two
outline(a, 700, 200, 1000, 400, GREEN, t=10)
a[600:612, 1100:] = a[888:900, 1100:] = a[600:900, 1100:1112] = YELLOW  # an outline running off the right edge
a[650:800, 300:450] = BLUE  # a filled saturated square: not an outline
p = OUT / "spar_box.jpg"
Image.fromarray(a).save(p, quality=85)
sb = [str(p)]
got = marks.detect_marks(sb, "spar", "box")
want = [("red", (200, 150, 500, 450)), ("green", (700, 200, 1000, 400)), ("yellow", (1100, 600, W, 900))]
check([(m["photo"], m["color"], m["kind"]) for m in got] == [(1, c, "box") for c, _ in want]
      and all(max(abs(g - f) for g, f in zip(m["box"], (x0 / W, y0 / H, x1 / W, y1 / H))) <= 0.005
              for m, (_, (x0, y0, x1, y1)) in zip(got, want)),
      f"one box per color, each side within 0.005 of the drawn one: {[(m['color'], m['box']) for m in got]}")
check(marks.detect_marks(sb, "spar", "point") == [] and marks.detect_marks(sp, "spar", "box") == [],
      "no points in the box photo, no boxes in the point photos")
check(marks.detect_marks([sp[0], sb[0]], "spar") == marks.detect_marks(sp[:1], "spar", "point") + [dict(m, photo=2) for m in got],
      "kind None: both")

# ----------------------------------------------------------------------------------------------- K3 other benchmarks
print("\n== K3 other benchmarks")
check(all(marks.detect_marks(["/nonexistent/photo.png"], b) == [] for b in ("3dsr", "mmsi", "mindcube", "vsi", "aa")),
      "no marks and no photo opened for 3dsr, mmsi, mindcube, vsi, aa")
check(marks.detect_marks(mv, "mmsi") == [], "marks drawn in an MMSI photo are not looked for")

# ------------------------------------------------------------------------------------------------- K4 first message
print("\n== K4 the analyst's first message")


def old_first_text(item, n_views, grid_line, blind):  # run_agent's first message without marks, copied verbatim
    return (f"{n_views} photo(s){' (not shown to you)' if blind else ''}. Grids: {grid_line}.\n"
            f"Question: {item['question']}\nOptions: " + "; ".join(f"{k}. {v}" for k, v in item["options"]))


item = {"question": "Photo 1: left gripper view.\nWhich object is outlined with a red bounding box?",
        "options": [("A", "The yellow bounding box."), ("B", "The red bounding box.")]}
GL = "CLIP (16, 16) per photo; VGGT and DINOv3 [(28, 37), (28, 37), (28, 37)]"
check(all(agent.analyst_question(item, 3, GL, blind, m) == old_first_text(item, 3, GL, blind)
          for blind in (True, False) for m in (None, [])), "no marks (None or []): the first message of the runs without marks")
M = [{"photo": 3, "color": "red", "kind": "box", "box": [0.621, 0.25, 0.684, 0.43], "center": [0.652, 0.34]},
     {"photo": 1, "color": "green", "kind": "point", "box": [0.403, 0.651, 0.418, 0.669], "center": [0.41, 0.66]}]
SECTION = ("\nMarks in the photo(s):\n- Photo 3: red box, x 0.62-0.68, y 0.25-0.43\n- Photo 1: green point at (0.41, 0.66)\n"
           + agent.MARKS_NOTE)
check(agent.analyst_question(item, 3, GL, True, M) == old_first_text(item, 3, GL, True) + SECTION,
      "with marks: the old text, then the section")
check("given, not guessed" in agent.MARKS_NOTE and "box(x0, y0, x1, y1, model, v)" in agent.MARKS_NOTE
      and "never to guess where an object is" in agent.ANALYST_BOXES, "the note says marked positions are given; the "
      "blind prompt's sentence on box() is unchanged")
print("   section:" + SECTION.replace("\n", "\n   "))

# --------------------------------------------------------------------------------------------------- K5 run_agent
print("\n== K5 run_agent: blend and one-encoder, Agent-Directed and Fixed-Sequence")
class StubSandbox:
    """Loads nothing and prints nothing: these checks are about the messages, not the tools."""

    def load(self, q):
        return "ok"

    def run(self, code):
        return "", []

    def close(self):
        pass


class ScriptedChat:
    """Objects for the object-listing call, FINDINGS for the analyst at once. Records every message list it is sent."""

    def __init__(self):
        self.seen = []

    def __call__(self, messages, max_new_tokens, think=False, prefix="", **kw):
        self.seen.append(messages)
        if "short noun phrase" in str(messages[0]["content"]):
            return json.dumps({"objects": ["the cup"]}), 10
        return 'FINDINGS: {"findings": [], "summary": "s", "regions": []}', 100


def make_args(latents, latent, all_tools):
    return types.SimpleNamespace(pixels=False, report_from=None, stop_after_code=False, latents_root=None, latent=latent,
                                 flip_text_map=bool(latents), all_tools=all_tools, latents=latents, max_objects=4,
                                 index_chars=16000, analyst_tokens=1000, max_steps=4, time_budget=600, cond3_latents=[],
                                 answer_formats=["pad"], marks=False)


lat = {"clip": str(OUT / "k5_clip.npy"), "vggt": str(OUT / "k5_vggt.npy"), "dinov3": str(OUT / "k5_dinov3.npy")}
np.save(lat["clip"], np.zeros((2, 16, 16, 4), np.float32))
np.save(lat["vggt"], np.zeros((2, 28, 37, 4), np.float32))
np.save(lat["dinov3"], np.zeros((2, 28, 37, 4), np.float32))
k5 = {"bench": "mvrb", "id": "k5", "type": "t", "images": mv[:2], "latents": lat, "question": item["question"],
      "options": item["options"]}


def analyst_msgs(module, it, args):
    chat, log = ScriptedChat(), {"calls": [], "rounds": []}
    module.run_agent(it, StubSandbox(), chat, args, log)
    return next(msgs for msgs in chat.seen if "short noun phrase" not in str(msgs[0]["content"]))


for latents, latent in ((["clip", "vggt", "dinov3"], None), (None, "vggt"), (None, "dinov3")):
    for all_tools in (False, True):
        tag = f"{'blend' if latents else latent} {'fixed' if all_tools else 'free'}"
        args = make_args(latents, latent, all_tools)
        off = analyst_msgs(agent, k5, args)
        on = analyst_msgs(agent, dict(k5, marks=M), args)
        grid_line = off[1]["content"][-1]["text"].split("Grids: ", 1)[1].split(".\n", 1)[0]
        old = old_first_text(k5, 2, grid_line, True)
        check(off[1]["content"] == [{"type": "text", "text": old}], f"{tag}: no marks -> the old first message")
        check(on[0] == off[0] and on[1]["content"] == [{"type": "text", "text": old + SECTION}] and on[2:] == off[2:],
              f"{tag}: marks -> same system prompt and later messages, first message + section")

# ------------------------------------------------------------------------------------------- K6 main() end to end
print("\n== K6 main() over the API (fake client): --marks on an MV-RoboBench and a SPAR sample")


class FakeAPI:
    """The OpenAI client's responses.create(): objects for the object call, FINDINGS for the analyst, "B" for every
    final answer (16 tokens). Records every call."""

    def __init__(self):
        self.calls = []
        self.responses = self

    def create(self, **kw):
        self.calls.append(kw)
        system = " ".join(m["content"] for m in kw["input"] if m["role"] == "system")
        if "short noun phrase" in system:
            text = json.dumps({"objects": ["the cup"]})
        elif kw["max_output_tokens"] <= 256:
            text = "B"
        else:
            text = 'FINDINGS: {"findings": [], "summary": "s", "regions": []}'
        return types.SimpleNamespace(output_text=text, usage=types.SimpleNamespace(input_tokens=7, output_tokens=1))


sd, lr = OUT / "k6_sample", OUT / "k6_latents"
for d in (sd, lr):
    shutil.rmtree(d, ignore_errors=True)
(sd / "img").mkdir(parents=True)
QS = {"q_marked": ("mvrb", mv[:2], item["question"]), "q_plain": ("mvrb", mv[2:], item["question"]),
      "s_point": ("spar", [sp[0], sb[0]], "How far is the chair (red point) from the lamp (green point)?"),
      "s_box": ("spar", [sp[0], sb[0]], "Is the sofa (green bbox) left of the table (red bbox)?")}
rows = {"mvrb": [], "spar": []}
for qid, (bench, photos, q) in QS.items():
    names = []
    for p in photos:
        shutil.copy(p, sd / "img" / f"{qid}_{Path(p).name}")
        names.append(f"img/{qid}_{Path(p).name}")
    for k, shape in (("clip", (16, 16, 4)), ("vggt", (28, 37, 4)), ("dinov3", (28, 37, 4))):
        (lr / k / bench).mkdir(parents=True, exist_ok=True)
        np.save(lr / k / bench / f"{qid}.npy", np.zeros((len(photos),) + shape, np.float32))
    rows[bench].append({"id": qid, "type": "t", "q": q, "opts": [list(o) for o in item["options"]], "ans": "B",
                        "imgs": names})
for bench, r in rows.items():
    (sd / f"samples_{bench}_t.json").write_text(json.dumps({bench: r}))
providers, sandbox_cls = dict(api_chat.PROVIDERS), agent.Sandbox
CMD = ["--sample", str(sd), "--latents-root", str(lr), "--model", "gpt-6-luna", "--blind", "--boxes", "--single-pass",
       "--index-chars", "16000", "--answer-formats", "pad", "--cond3-latents"]
seen = {}
try:
    agent.Sandbox = StubSandbox
    for tag, extra in (("free", ["--latents", "clip", "vggt", "dinov3"]), ("fixed", ["--latents", "clip", "vggt", "dinov3", "--all-tools"])):
        for flag in ([], ["--marks"]):
            fake = FakeAPI()
            api_chat.PROVIDERS["gpt-"] = lambda model, fake=fake: api_chat.OpenAIChat(model, client=fake)
            run = OUT / f"k6_{tag}{'_marks' if flag else ''}"
            shutil.rmtree(run, ignore_errors=True)
            sys.argv = ["agent.py", "--out", str(run)] + CMD + extra + flag
            with contextlib.redirect_stdout(io.StringIO()):
                agent.main()
            logs = {p.stem: json.loads(p.read_text()) for p in (run / "transcripts").glob("*.json")}
            analyst = {}  # question -> the analyst's request (the call whose system prompt is the analyst's)
            for c in fake.calls:
                sys_text = " ".join(m["content"] for m in c["input"] if m["role"] == "system")
                if "You are the analyst" in sys_text:
                    first = next(m for m in c["input"] if m["role"] == "user")["content"][0]["text"]
                    analyst[next(qid for qid, (_, photos, q) in QS.items()
                                 if first.startswith(f"{len(photos)} photo(s)") and f"Question: {q}\n" in first)] = c["input"]
            seen[(tag, bool(flag))] = (logs, analyst)
finally:
    api_chat.PROVIDERS.clear()
    api_chat.PROVIDERS.update(providers)
    agent.Sandbox = sandbox_cls
for tag in ("free", "fixed"):
    (logs0, an0), (logs1, an1) = seen[(tag, False)], seen[(tag, True)]
    want = marks.detect_marks([str(sd / "img" / f"q_marked_{Path(p).name}") for p in mv[:2]], "mvrb")
    check(len(logs0) == len(logs1) == 4 and not any("error" in x for x in list(logs0.values()) + list(logs1.values())),
          f"{tag}: both runs answer all four questions without error")
    check(not any("marks" in x for x in logs0.values()), f"{tag}: without --marks no \"marks\" in any transcript")
    check(logs1["mvrb_q_marked"].get("marks") == want and len(want) == 5 and logs1["mvrb_q_plain"].get("marks") == [],
          f"{tag}: --marks logs the marks ({len(want)} on the marked question, [] on the other)")
    first1 = an1["q_marked"][1]["content"][0]["text"]
    check(first1 == an0["q_marked"][1]["content"][0]["text"] + agent.marks_text(want) and "- Photo 2: green box, x " in first1,
          f"{tag}: --marks adds the section to the marked question's first message only")
    check(an1["q_plain"] == an0["q_plain"] and an1["q_marked"][0] == an0["q_marked"][0],
          f"{tag}: the unmarked question's analyst request and every system prompt are unchanged")
    check(logs1["mvrb_q_marked"]["final"].keys() == logs0["mvrb_q_marked"]["final"].keys(), f"{tag}: the same answer runs")
    for qid, kind, kinds in (("s_point", "point", "points"), ("s_box", "box", "boxes")):  # the same photos, points and boxes
        want = marks.detect_marks([str(sd / "img" / f"{qid}_{Path(p).name}") for p in QS[qid][1]], "spar", kind)
        check(logs1[f"spar_{qid}"].get("marks") == want and len(want) >= 3 and all(m["kind"] == kind for m in want)
              and an1[qid][1]["content"][0]["text"] == an0[qid][1]["content"][0]["text"] + agent.marks_text(want),
              f"{tag}: SPAR, the question names {kinds} -> only the {len(want)} {kinds}, logged and in the first message")
print("   first message with marks:\n   " + seen[("free", True)][1]["q_marked"][1]["content"][0]["text"].replace("\n", "\n   "))

# --------------------------------------------------------------------------------------------- K7 locate_all(v=)
print("\n== K7 locate_all(phrases, v=k)")
OBJS = {"the red car": (0.55, 0.55, 0.80, 0.80), "a wooden chair": (0.15, 0.30, 0.32, 0.62)}
sc = Scene(9, OBJS)
imgs, lat7, clip = write_question(Helpers, OUT, "k7", sc, [(640, 480)] * 2, [0.0, 0.12])
for o in OBJS:
    clip.plant(o, sc.clip_dir(o))
h = Helpers(clip)
h.load({"images": imgs, "latents": lat7, "blind": True, "flip_text_map": True, "use": ["clip", "vggt", "dinov3"],
        "no_show": True})
for v in (0, 1):
    try:
        r = h.locate_all(list(OBJS), v=v)
        one = {o: h.locate(o, v=v) for o in OBJS}
        ok = all(r[o]["view"] == v and r[o].get("box") == one[o].get("box") for o in OBJS)
    except TypeError as e:
        ok, r = False, e
    check(ok, f"locate_all(..., v={v}): every phrase located in photo {v + 1}, as locate(phrase, v={v}) does")
check(set(h.locate_all(list(OBJS))) == set(OBJS), "locate_all without v: as before")

print(f"\n{len(failures)} failure(s)" + "".join(f"\n  {f}" for f in failures))
sys.exit(1 if failures else 0)
