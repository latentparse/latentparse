#!/usr/bin/env python3
"""CPU test of agent.py in every setting of the paper: python3 test_agent.py [--print]

Each setting runs main() end to end on a small synthetic sample (MMSI-Bench and SPAR-Bench-Tiny questions with 1-3
photos), with a fake API model as analyst and answering VLM and a sandbox that records what it loads. Every request
the VLM receives (images replaced by their size), every sandbox load, transcript and answer row is hashed, and the
hash must equal GOLDEN, which was recorded with the code of the paper runs. So the analyst prompt, the first message,
the tool steps, the report and the answering prompt are exactly those of the paper in every setting. No torch, no
weights, no network. --print prints the hashes instead of checking them. Exits 1 on any difference."""
import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PRINT = "--print" in sys.argv
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "direct_input"))

torch = types.ModuleType("torch")
torch.bfloat16 = "bf16"
torch.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
torch.inference_mode = torch.no_grad = contextlib.nullcontext
torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
torch.nn = types.SimpleNamespace(Module=object, functional=types.SimpleNamespace(pad=None))
torch.Tensor = type("Tensor", (), {})
sys.modules["torch"] = torch
for _name in ("openai", "anthropic"):
    sys.modules[_name] = types.SimpleNamespace(OpenAI=None, Anthropic=None, BadRequestError=Exception)

import agent  # noqa: E402
import api_chat  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="equiv_"))
rng = np.random.default_rng(0)
W, H = 640, 480
rows = round(H * 518 / W / 14)


def photo(path, seed, mark=None):
    r = np.random.default_rng(seed)
    im = Image.fromarray((r.random((H // 8, W // 8, 3)) * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    if mark:
        ImageDraw.Draw(im).ellipse(mark, fill=(255, 0, 0))
    im.save(path)


sd, lr = TMP / "sample", TMP / "lat"
(sd / "img").mkdir(parents=True)
spec = {"mmsi": [("0", 1, "Motion (Cam.)"), ("1", 2, "MSR"), ("2", 3, "Positional Relationship (Cam.–Obj.)")],
        "spar": [("200", 1, "t"), ("700", 2, "t")]}
files = {}
for bench, qs in spec.items():
    out = []
    for qid, nv, typ in qs:
        names = []
        for v in range(nv):
            p = sd / "img" / f"{bench}_{qid}_{v}.png"
            photo(p, sum(map(ord, f"{bench}{qid}{v}")), mark=[300, 220, 330, 250] if bench == "spar" and v == 0 else None)
            names.append(f"img/{p.name}")
        for k, shape in (("clip", (16, 16, 6)), ("vggt", (rows, 37, 6)), ("dinov3", (rows, 37, 6))):
            (lr / k / bench).mkdir(parents=True, exist_ok=True)
            np.save(lr / k / bench / f"{qid}.npy", rng.standard_normal((nv,) + shape).astype(np.float32))
        q = ("From the observer's view in the first image, is the chair (red point) left of the lamp?" if bench == "spar"
             else f"Question {qid}: where did the camera move between the photos?")
        out.append({"id": qid, "type": typ, "q": q, "opts": [["A", "left"], ["B", "right"], ["C", "up"], ["D", "down"]],
                    "ans": "B", "imgs": names})
    key = {"mmsi": "mmsi", "spar": "spar"}[bench]
    files[bench] = out
(sd / "samples_mmsi_t.json").write_text(json.dumps({"mmsi": files["mmsi"]}))
(sd / "samples_spar_t.json").write_text(json.dumps({"spar": files["spar"]}))


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.responses = self

    def create(self, **kw):
        self.calls.append(kw)
        system = " ".join(m["content"] for m in kw["input"] if m["role"] == "system" and isinstance(m["content"], str))
        n = len(self.calls)
        if "short noun phrase" in system:
            text = json.dumps({"objects": ["the chair", "the lamp"]})
        elif kw["max_output_tokens"] <= 256:
            text = "AB"[n % 2]
        elif n % 3 == 1:
            text = "```python\nprint(motion(0, 1) if len(F['vggt']) > 1 else layout(0))\n```"
        else:
            text = ('FINDINGS: {"findings": [{"text": "camera moves right", "source": "latent", "evidence": ["E1"], '
                    '"reliability": "high", "why": "both encoders"}], "summary": "s", '
                    '"regions": [{"object": "the chair", "view": 1, "box": [0.4, 0.4, 0.6, 0.6]}]}')
        return types.SimpleNamespace(output_text=text, usage=types.SimpleNamespace(input_tokens=7, output_tokens=1))


class RecordingSandbox:
    loaded = []

    def load(self, q):
        RecordingSandbox.loaded.append(copy.deepcopy(q))
        return "ok"

    def run(self, code):
        return f"ran {len(code)} chars", []

    def close(self):
        pass


DROP = {"overlay", "chan_norm", "only_tools", "drop_tools"}  # sandbox keys of options the paper never used


def scrub(x):
    if isinstance(x, str) and x.startswith("data:image"):
        import base64
        from io import BytesIO
        im = Image.open(BytesIO(base64.b64decode(x.split(",", 1)[1])))
        return f"<image {im.size[0]}x{im.size[1]}>"
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if not any(s in k for s in ("second", "time", "elapsed"))
                and not (k in DROP and not v)}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, str):
        return x.replace(str(TMP), "<TMP>")
    return x


P = ["--blind", "--boxes", "--single-pass", "--index-chars", "16000"]
C3 = ["--answer-formats", "pad", "--cond3-latents"]
M = ["--latents", "clip", "vggt", "dinov3"]
CONFIGS = {
    "multi_agent_directed_mmsi": P + C3 + M + ["--bench", "mmsi"],
    "multi_fixed_sequence_mmsi": P + C3 + M + ["--bench", "mmsi", "--all-tools"],
    "multi_mismatched_mmsi": P + C3 + M + ["--bench", "mmsi", "--wrong-latent"],
    "multi_noise_mmsi": P + C3 + M + ["--bench", "mmsi", "--noise-latent"],
    "multi_marks_spar": P + C3 + M + ["--bench", "spar", "--marks"],
    "multi_marks_fixed_spar": P + C3 + M + ["--bench", "spar", "--marks", "--all-tools"],
    "one_vggt_mmsi": P + ["--answer-formats", "image", "--latent", "vggt", "--bench", "mmsi"],
    "one_clip_mmsi": P + ["--answer-formats", "image", "--latent", "clip", "--bench", "mmsi"],
    "one_dinov3_mmsi": P + ["--answer-formats", "image", "--latent", "dinov3", "--bench", "mmsi"],
    "one_vggt_fixed_mmsi": P + ["--answer-formats", "image", "--latent", "vggt", "--bench", "mmsi", "--all-tools"],
    "one_dinov3_marks_spar": P + ["--answer-formats", "image", "--latent", "dinov3", "--bench", "spar", "--marks"],
    "source_question_mmsi": P + C3 + ["--report-from", "question", "--bench", "mmsi"],
    "source_photos_mmsi": P + C3 + ["--report-from", "photos", "--bench", "mmsi"],
    "source_pixels_mmsi": P + C3 + ["--latents", "clip", "vggt", "--pixels", "--bench", "mmsi"],
    "camera_head_answers": "after:multi_agent_directed_mmsi",
}
result = {}
providers, sandbox_cls = dict(api_chat.PROVIDERS), agent.Sandbox
for name, cmd in CONFIGS.items():
    fake = FakeAPI()
    api_chat.PROVIDERS["gpt-"] = lambda model, fake=fake: api_chat.OpenAIChat(model, client=fake)
    agent.Sandbox = RecordingSandbox
    RecordingSandbox.loaded = []
    run = TMP / "runs" / name
    if isinstance(cmd, str):
        src = TMP / "runs" / cmd.split(":")[1]
        shutil.copytree(src, run)
        for tp in (run / "transcripts").glob("*.json"):
            tr = json.loads(tp.read_text())
            tr["final"] = {}
            tr["notes"] = "- From photo 1 to photo 2 the camera turns 20 degrees right; its position moves right 0.90 (unit direction)."
            tp.write_text(json.dumps(tr))
        for d in run.glob("*/results.jsonl"):
            d.unlink()
        argv = CONFIGS[cmd.split(":")[1]] + ["--answers-only", "agent_only", "--notes-header", "camera_head"]
    else:
        argv = cmd
    sys.argv = ["agent.py", "--sample", str(sd), "--latents-root", str(lr), "--model", "gpt-6-luna", "--out", str(run)] + argv
    err = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errbuf := io.StringIO()):
            agent.main()
    except SystemExit as e:
        err = f"SystemExit {e}"
        try:
            err += " " + errbuf.getvalue()[-300:]
        except NameError:
            pass
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    trans = {p.name: scrub(json.loads(p.read_text())) for p in sorted((run / "transcripts").glob("*.json"))} if (run / "transcripts").exists() else {}
    res = {str(p.relative_to(run)): scrub([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
           for p in sorted(run.glob("*/results.jsonl"))}
    result[name] = {"error": err, "calls": scrub([{"input": c["input"], "max": c["max_output_tokens"]} for c in fake.calls]),
                    "loads": scrub(RecordingSandbox.loaded), "transcripts": trans, "results": res}
api_chat.PROVIDERS.clear()
api_chat.PROVIDERS.update(providers)
shutil.rmtree(TMP, ignore_errors=True)
hashes = {k: hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:16] for k, v in result.items()}
if PRINT:
    print(json.dumps(hashes, indent=1))
    sys.exit(0)
GOLDEN = {  # recorded with the code of the paper runs
    "multi_agent_directed_mmsi": "f189172fe8bbc89c",
    "multi_fixed_sequence_mmsi": "88b8185abb3091bd",
    "multi_mismatched_mmsi": "42dc947425226bc1",
    "multi_noise_mmsi": "68303d0f085457c8",
    "multi_marks_spar": "49313df23c0ca8ce",
    "multi_marks_fixed_spar": "82f6594fe1e386fc",
    "one_vggt_mmsi": "d93861841e1fd212",
    "one_clip_mmsi": "453306af9198b07f",
    "one_dinov3_mmsi": "7059fffeada41d74",
    "one_vggt_fixed_mmsi": "5674c13bcdd96885",
    "one_dinov3_marks_spar": "6f53816d1ac3674c",
    "source_question_mmsi": "5da4f6dc0ee60629",
    "source_photos_mmsi": "356a6eee6a6eb9c5",
    "source_pixels_mmsi": "dcc1f169a8af6a3a",
    "camera_head_answers": "8aa633408d61c029",
}
bad = [k for k in CONFIGS if hashes.get(k) != GOLDEN.get(k) or result[k]["error"]]
for k in CONFIGS:
    print(("   ok  " if k not in bad else "   FAIL ") + f"{k}: {hashes[k]}" + (f" (expected {GOLDEN.get(k)})" if k in bad else ""))
print(f"\n{len(bad)} failure(s)")
sys.exit(1 if bad else 0)
