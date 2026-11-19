#!/usr/bin/env python3
"""Photo only and direct input with a local VLM (Qwen3.8-27B, Gemma 4, InternVL3-38B): the photo(s) alone, or the
photo(s) with the latent features of one encoder, or of all three (blend), given directly.

Usage: python answer_qwen.py --sample DIR --model MODEL_DIR --out RESULTS_DIR \
           --runs baseline clip:image clip:pad blend:image [--latents-root DIR] [--limit N] [--max-soft-tokens 1120]

Each run writes RESULTS_DIR/<run>/results.jsonl (one row per question),
metrics.json and README.md. The model is loaded once for all runs.

Latent formats (the latent is the [views, h, w, dim] array from extract_latents.py):
- image: per view, the (h*w) x dim matrix as a grayscale image, one row per patch
  (row-major), one column per channel, min-max scaled to 0-255 over the whole
  matrix. Rows are padded with black up to a multiple of 32 so Qwen's 32-pixel
  tokens never resample the matrix.
- pad: per view, h*w tokens, one per patch. Each patch vector is L2-normalised,
  scaled to the mean norm of Qwen's input-embedding rows, zero-padded at the end
  to Qwen's hidden size, and written into the embeddings of <|vision_pad|>
  placeholder tokens. Per-patch normalising drops each patch's own norm.
- padglobal: like pad, but the whole latent (all views) is multiplied by one
  scalar so its mean patch norm equals the mean embedding norm. Relative patch
  norms are kept, so nothing but a single scale factor is lost.

Outlier-robust formats (suffix z). A handful of channels carry most of the latent's
energy (CLIP top-5: ~47%, VGGT: ~86%), which flattens min-max images to a few gray
levels and drowns every other channel in padding. These formats first z-score each
channel with its mean and std over all patches of all latents of that model in the
sample (latents-root/<kind>/stats.npz) and clip to +-3 std:
- imagez: the image format, but z mapped linearly from [-3, 3] to [0, 255].
- tilez: each patch's channels as one tile (1024 -> 32x32, 2048 -> 32x64 pixels),
  tiles laid out like the patch grid, so neighbouring patches stay neighbours.
- padz: padglobal on the z-scored latent.
- tokz (Qwen only): the z-scored latent as image tokens. A blank image of exactly
  (32h x 32w) pixels gives an h x w grid of image tokens with Qwen's 2D positions;
  its vision features are replaced by the latent vectors (zero-padded, scaled to the
  mean norm of the real photo tokens in the same prompt).
"""
import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import load_sample, parse_letter, question_text

LATENTS = {
    "clip": ("CLIP ViT-L/14", "image content and semantics, aligned with text"),
    "vggt": ("VGGT", "3D scene geometry: depth, surfaces and camera viewpoint"),
    "dinov3": ("DINOv3 ViT-L/16", "object parts and which parts of different photos show the same thing"),
}
# --runs blend:<format>: the three latents one after another, each in that format (the blend of the main table).
BLEND = ("clip", "vggt", "dinov3")
# Placeholder slots for padded latents: the first of these that the tokenizer maps to a
# single id (Qwen: <|vision_pad|>, Gemma 4: <mask>).
PLACEHOLDER_CANDIDATES = ["<|vision_pad|>", "<mask>"]
# Pre-filled start of Qwen's reply so the next token is the option letter. Without it Qwen
# starts explaining on MMSI and runs out of tokens before naming an answer.
ANSWER_PREFIX = "Answer:"  # no trailing space: Qwen's tokens carry the space before the letter (" B")


def latent_image(view):
    h, w, d = view.shape
    m = view.reshape(h * w, d)
    lo, hi = m.min(), m.max()
    return matrix_image(np.round((m - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8))


def matrix_image(px):
    rows = -(-px.shape[0] // 32) * 32
    px = np.pad(px, ((0, rows - px.shape[0]), (0, 0)))
    return Image.fromarray(px, mode="L").convert("RGB")


def latent_stats(root, kind):
    """Per-channel mean and std over every patch of every latent of one model in the sample."""
    path = Path(root) / kind / "stats.npz"
    if path.exists():
        s = np.load(path)
        return s["mean"], s["std"]
    total = sq = count = 0
    for f in sorted((Path(root) / kind).glob("*/*.npy")):
        x = np.load(f)
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        total, sq, count = total + x.sum(0), sq + (x ** 2).sum(0), count + x.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(sq / count - mean ** 2, 1e-12))
    tmp = path.with_name(f"stats.{time.time_ns()}.npz")
    np.savez(tmp, mean=mean, std=std, patches=count)
    tmp.rename(path)  # atomic, so parallel jobs never read a half-written file
    return mean, std


def noise_like(x, key):
    """Noise control: Gaussian noise of x's shape with each view's per-channel mean and std, seeded by key
    ("<bench>/<id>/<kind>"); the same numbers as latentparse/sandbox.load_latent gives the agent's --noise-latent."""
    import zlib

    x = x.astype(np.float32)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    mu, sd = x.mean((1, 2), keepdims=True), x.std((1, 2), keepdims=True)
    return mu + sd * rng.standard_normal(x.shape, dtype=np.float32)


def load_run_latent(root, kind, bench, qid, noise=None):
    """A run's latent for one question: one [views, h, w, dim] array, or for kind "blend" a dict of the three. With noise
    ("<bench>/<id>"), noise_like each latent instead."""
    load = lambda k: np.load(Path(root) / k / bench / f"{qid}.npy")
    if kind == "blend":
        return {k: noise_like(load(k), f"{noise}/{k}") if noise else load(k) for k in BLEND}
    return noise_like(load(kind), f"{noise}/{kind}") if noise else load(kind)


def run_stats(root, kind, fmt, cache):
    """Per-channel stats for the z formats, cached per latent in `cache` (for the blend, a dict per latent)."""
    if not fmt.endswith("z"):
        return None
    kinds = BLEND if kind == "blend" else (kind,)
    for k in kinds:
        if k not in cache:
            cache[k] = latent_stats(root, k)
    return {k: cache[k] for k in kinds} if kind == "blend" else cache[kind]


def zscore(view, stats):
    mean, std = stats
    return np.clip((view - mean) / std, -3, 3).astype(np.float32)


def z_gray(z):
    return np.round((z + 3) / 6 * 255).astype(np.uint8)


def pca_image(view, cell=14):
    """[h, w, d] -> RGB picture of the patch grid: the first three principal components over this photo's patches,
    each min-max scaled to 0-255, one cell x cell block per patch."""
    h, w, d = view.shape
    m = view.reshape(h * w, d).astype(np.float64)
    m = m - m.mean(0)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    p = m @ vt[:3].T
    p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-12)
    img = Image.fromarray(np.round(p * 255).astype(np.uint8).reshape(h, w, 3), mode="RGB")
    return img.resize((w * cell, h * cell), Image.NEAREST)


def tile_image(z):
    """[h, w, d] z-scores -> one (32 x d/32) tile per patch, tiles laid out like the patch grid."""
    h, w, d = z.shape
    th, tw = 32, d // 32
    px = z_gray(z).reshape(h, w, th, tw).transpose(0, 2, 1, 3).reshape(h * th, w * tw)
    return Image.fromarray(px, mode="L").convert("RGB")


class ImageSlotInjector:
    """Replaces the vision features of chosen images with latent vectors.

    Wraps the inner model's get_image_features, so the latent keeps the 2D positions
    Qwen computes from the blank image's grid.
    """

    def __init__(self, model):
        self.inner = model.model
        self.orig = self.inner.get_image_features
        self.inner.get_image_features = self
        self.slots = {}

    def __call__(self, *args, **kwargs):
        out = self.orig(*args, **kwargs)
        if not self.slots:
            return out
        embeds = list(out.pooler_output)
        real = torch.cat([e for i, e in enumerate(embeds) if i not in self.slots])
        target = real.float().norm(dim=1).mean()
        for i, v in self.slots.items():
            if v.shape[0] != embeds[i].shape[0]:
                raise RuntimeError(f"image {i}: {embeds[i].shape[0]} image tokens but {v.shape[0]} latent vectors")
            v = v.to(embeds[i].device).float()
            v = torch.nn.functional.pad(v, (0, embeds[i].shape[1] - v.shape[1]))
            embeds[i] = (v * (target / v.norm(dim=1).mean().clamp_min(1e-12))).to(embeds[i].dtype)
        out.pooler_output = tuple(embeds)
        return out


class LatentInjector:
    """Forward hook on the input embeddings that swaps placeholder rows for latent vectors."""

    def __init__(self, emb, placeholder, placeholder_id):
        self.placeholder = placeholder
        self.placeholder_id = placeholder_id
        self.hidden = emb.weight.shape[1]
        self.vectors = None
        # Mean norm of what the embedding layer actually outputs: Gemma scales its
        # embeddings inside forward(), so the raw weight norms would be wrong there.
        with torch.no_grad():
            ids = torch.arange(min(emb.weight.shape[0], 50000), device=emb.weight.device)
            self.target_norm = emb(ids).float().norm(dim=1).mean().item()
        emb.register_forward_hook(self)

    def global_scale(self, latent):
        """One scalar for the whole latent: mean patch norm -> mean embedding norm."""
        norms = np.linalg.norm(latent.reshape(-1, latent.shape[-1]), axis=1)
        return self.target_norm / max(float(norms.mean()), 1e-12)

    def tokens(self, view, scale=None):
        h, w, d = view.shape
        v = torch.from_numpy(view.reshape(h * w, d)).float()
        if scale is None:
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12) * self.target_norm
        else:
            v = v * scale
        return torch.nn.functional.pad(v, (0, self.hidden - d))

    def __call__(self, module, inputs, output):
        ids = inputs[0]
        mask = ids == self.placeholder_id
        if self.vectors is None or not mask.any():
            return output
        if mask.sum() != self.vectors.shape[0]:
            raise RuntimeError(f"{mask.sum().item()} placeholders but {self.vectors.shape[0]} latent tokens")
        output = output.clone()
        output[mask] = self.vectors.to(output.device, output.dtype)
        return output


def build_content(item, latent, kind, fmt, injector, stats=None, photos=True):
    """Prompt content, placeholder vectors (pad formats) and {image index: vectors} (tokz).

    photos=False leaves the photos out: the model gets only the latent (or, for the baseline,
    only the question)."""
    content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in item["images"]] if photos else []
    vectors, slots = [], {}
    # The blend gives the three latents one after another, each introduced and drawn as on its own.
    for kind, latent, stats in ([(k, latent[k], stats and stats[k]) for k in BLEND] if kind == "blend" else [(kind, latent, stats)]):
        if latent is None:
            continue
        name, what = LATENTS[kind]
        if photos:
            intro = f"\nYou are also given raw latent features from {name}, a frozen vision model that encodes {what}."
        else:
            intro = (f"You are given raw latent features of {len(item['images'])} photo(s) from {name}, a frozen vision model "
                     f"that encodes {what}. The photos themselves are not shown.")
        content.append({"type": "text", "text": intro + " They come straight from its encoder, without any decoding."})
        z = zscore(latent, stats) if fmt.endswith("z") else None
        for k, view in enumerate(latent, 1):
            h, w, d = view.shape
            if fmt == "image":
                content.append({"type": "text", "text": f"\nLatent of photo {k}: a {h}x{w} grid of patches with {d} channels, drawn as "
                                                         f"a grayscale matrix. Each row is one patch in row-major order (top-left to "
                                                         f"bottom-right), each column one channel; brightness is the value, min-max "
                                                         f"scaled to 0-255. Rows past {h * w} are black padding."})
                content.append({"type": "image", "image": latent_image(view)})
            elif fmt == "imagez":
                content.append({"type": "text", "text": f"\nLatent of photo {k}: a {h}x{w} grid of patches with {d} channels, drawn as "
                                                         f"a grayscale matrix. Each row is one patch in row-major order (top-left to "
                                                         f"bottom-right), each column one channel. Each channel is standardised; "
                                                         f"brightness maps -3 to +3 standard deviations onto black to white. "
                                                         f"Rows past {h * w} are black padding."})
                content.append({"type": "image", "image": matrix_image(z_gray(z[k - 1].reshape(h * w, d)))})
            elif fmt == "tilez":
                content.append({"type": "text", "text": f"\nLatent of photo {k}: a {h}x{w} grid of patches laid out like the photo. "
                                                         f"Each patch is a 32x{d // 32} tile whose pixels are its {d} channels in "
                                                         f"row-major order (channel 0 at the tile's top-left). Each channel is "
                                                         f"standardised; brightness maps -3 to +3 standard deviations onto black to white."})
                content.append({"type": "image", "image": tile_image(z[k - 1])})
            elif fmt == "pcargb":
                content.append({"type": "text", "text": f"\nLatent of photo {k}: a {h}x{w} grid of patches laid out like the photo. "
                                                         f"The colour of each patch shows its first three principal components over "
                                                         f"the patches of this photo as red, green and blue, so patches with similar "
                                                         f"features have similar colours."})
                content.append({"type": "image", "image": pca_image(view)})
            elif fmt == "tokz":
                content.append({"type": "text", "text": f"\nLatent of photo {k}: the next image slot is not a picture. It holds the latent "
                                                         f"as a {h}x{w} grid of image tokens laid out like the photo, one token per patch "
                                                         f"carrying its {d} standardised feature values (zero-padded to the model width)."})
                slots[sum(c["type"] == "image" for c in content)] = torch.from_numpy(z[k - 1].reshape(h * w, d))
                content.append({"type": "image", "image": Image.new("RGB", (32 * w, 32 * h), (128, 128, 128))})
            else:
                content.append({"type": "text", "text": f"\nLatent of photo {k}: the next {h * w} tokens are one patch each of a {h}x{w} "
                                                         f"grid in row-major order (top-left to bottom-right), each holding the patch's "
                                                         f"{d} feature values (rescaled, zero-padded to the model width): "
                                                         + injector.placeholder * (h * w)})
                src = z if fmt == "padz" else latent
                scale = injector.global_scale(src) if fmt in ("padglobal", "padz") else None
                vectors.append(injector.tokens(src[k - 1], scale))
    content.append({"type": "text", "text": "\n" + question_text(item)})
    return content, (torch.cat(vectors) if vectors else None), slots


# Benchmarks whose Wrong donors stay exactly as in every run so far (one generator for all their groups, sorted order).
SHARED_DONOR_RNG = ("3dsr", "mmsi", "spar")


def donor_scene(it):
    """Questions that must not lend each other a Wrong latent: for MindCube the scene folder of the views
    (img/mindcube/<setting>/<scene>/; 741 of its 1,050 questions share their exact photos with another), else the photos."""
    return str(Path(it["images"][0]).parent) if it["bench"] == "mindcube" else tuple(it["images"])


def wrong_latent_donors(items, seed=0):
    """Map each question to a different question of the same benchmark whose latent it borrows.

    Questions are grouped by benchmark and number of views, shuffled with a fixed seed and
    rotated by one, so nobody keeps its own latent and the latent has as many views as the
    question has photos. A question alone in its group borrows from the question with the
    nearest view count.

    3DSR, MMSI and SPAR keep exactly these donors. Any other benchmark (MindCube) gets a generator
    of its own, seeded the same way, so its donors do not depend on what else is in the sample dir,
    and never borrows from its own scene (donor_scene): the shuffled group is laid out scene by
    scene (in order of first appearance) and rotated by the size of the largest scene, which puts
    every donor in another scene when no scene holds more than half the group (else a warning).
    Each latent is still lent exactly once.

    The shared generator walks the groups in sorted order, 3DSR first, so the MMSI donors of every run so far
    (sample dir with both files: 0 -> 792, 140 -> 975 on the full dir) need the 3DSR questions in the same items;
    without them a warning is printed.
    """
    import random
    import sys

    groups = collections.defaultdict(list)
    for it in items:
        groups[(it["bench"], len(it["images"]))].append(it["id"])
    benches = {b for b, _ in groups}
    if "mmsi" in benches and "3dsr" not in benches:
        print("warning: MMSI questions without 3DSR ones, so the MMSI Wrong donors differ from those of every run so far "
              "(drawn over a sample dir holding both, e.g. 0 -> 792, 140 -> 975)", file=sys.stderr)
    donors = {}
    rng = random.Random(seed)
    for (bench, n), ids in sorted(groups.items()):
        if bench not in SHARED_DONOR_RNG:
            continue
        if len(ids) > 1:
            order = sorted(ids)
            rng.shuffle(order)
            donors.update({(bench, a): b for a, b in zip(order, order[1:] + order[:1])})
        else:
            others = [it for it in items if it["bench"] == bench and it["id"] != ids[0]]
            nearest = min(others, key=lambda it: (abs(len(it["images"]) - n), it["id"]))
            donors[(bench, ids[0])] = nearest["id"]
    scene = {(it["bench"], it["id"]): donor_scene(it) for it in items if it["bench"] not in SHARED_DONOR_RNG}
    own_rng = {}
    for (bench, n), ids in sorted(groups.items()):
        if bench in SHARED_DONOR_RNG:
            continue
        r = own_rng.setdefault(bench, random.Random(seed))
        if len(ids) > 1:
            order = sorted(ids)
            r.shuffle(order)
            blocks = collections.defaultdict(list)  # scenes in order of first appearance
            for q in order:
                blocks[scene[(bench, q)]].append(q)
            order = [q for b in blocks.values() for q in b]
            k = max(map(len, blocks.values()))
            if 2 * k > len(order):
                print(f"warning: {bench}, {n} views: one scene holds {k} of {len(order)} questions, so some Wrong "
                      f"donors come from the question's own scene", file=sys.stderr)
            k = min(k, len(order) // 2)
            donors.update({(bench, a): b for a, b in zip(order, order[k:] + order[:k])})
        else:
            others = [it for it in items if it["bench"] == bench and it["id"] != ids[0]]
            others = [it for it in others if scene[(bench, it["id"])] != scene[(bench, ids[0])]] or others
            nearest = min(others, key=lambda it: (abs(len(it["images"]) - n), it["id"]))
            donors[(bench, ids[0])] = nearest["id"]
    return donors


def summarize(rows):
    def acc(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None

    by_bench = collections.defaultdict(list)
    by_type = collections.defaultdict(list)
    for r in rows:
        by_bench[r["bench"]].append(r)
        by_type[f"{r['bench']}/{r['type']}"].append(r)
    return {
        "n": len(rows),
        "accuracy": acc(rows),
        "by_benchmark": {b: {"n": len(rs), "accuracy": acc(rs),
                             "chance": round(sum(1 / len(r["options"]) for r in rs) / len(rs), 4),
                             "unparsed": sum(r["prediction"] is None for r in rs)} for b, rs in by_bench.items()},
        "by_type": {t: {"n": len(rs), "accuracy": acc(rs)} for t, rs in sorted(by_type.items())},
        "seconds_per_question": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="baseline, or <clip|vggt|dinov3|blend>:<format>[:wrong|:noise] (blend = all three latents; wrong = "
                         "another question's latent; noise = Gaussian noise with each view's per-channel mean and std, as "
                         "agent.py --noise-latent); formats: "
                         "image, pad, padglobal, imagez, tilez, padz, tokz (Qwen only), pcargb (PCA color image)")
    ap.add_argument("--latents-root", help="dir holding <kind>/<bench>/<id>.npy")
    ap.add_argument("--limit", type=int, help="questions per benchmark (for a smoke test)")
    ap.add_argument("--ids", nargs="+", help="only these question ids (debugging); wrong-latent donors stay as in the full sample")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1, help=">1: sample this many replies and take the majority letter")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-soft-tokens", type=int,
                    help="Gemma 4 only: image token budget for every image (70/140/280/560/1120)")
    ap.add_argument("--attn", default="sdpa", help="attention implementation (sdpa, flex_attention, eager)")
    ap.add_argument("--no-photo", action="store_true",
                    help="leave the photos out: latent only, and the baseline becomes question text only")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    # DEVICE_MAP=auto spreads a model too big for one GPU (InternVL3-38B) over the visible GPUs, as in agent.py
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map=os.environ.get("DEVICE_MAP", "cuda"),
                                                        attn_implementation=args.attn).eval()
    tok = processor.tokenizer
    placeholder = next(p for p in PLACEHOLDER_CANDIDATES if len(tok.encode(p, add_special_tokens=False)) == 1)
    injector = LatentInjector(model.get_input_embeddings(), placeholder, tok.convert_tokens_to_ids(placeholder))
    slot_injector = ImageSlotInjector(model) if any(r.split(":")[1:2] == ["tokz"] for r in args.runs) else None
    image_kwargs = {"images_kwargs": {"max_soft_tokens": args.max_soft_tokens}} if args.max_soft_tokens else {}
    items = load_sample(args.sample, args.limit)
    donors = wrong_latent_donors(items)
    if args.ids:
        items = [it for it in items if it["id"] in args.ids]

    stats = {}
    for run in args.runs:
        kind, fmt, ctrl = (run.split(":") + ["", ""])[:3]
        if ctrl not in ("", "wrong", "noise"):
            ap.error(f"run {run}: the third field is wrong or noise")
        st = run_stats(args.latents_root, kind, fmt, stats)
        name = ("baseline" if kind == "baseline" else f"{kind}_{fmt}" + (f"_{ctrl}" if ctrl else "")) + (f"_sc{args.samples}" if args.samples > 1 else "")
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        with open(out / "results.jsonl", "w") as f, torch.inference_mode():
            for it in items:
                src = donors[(it["bench"], it["id"])] if ctrl == "wrong" else it["id"]
                noise = f"{it['bench']}/{it['id']}" if ctrl == "noise" else None
                latent = None if kind == "baseline" else load_run_latent(args.latents_root, kind, it["bench"], src, noise)
                content, vectors, slots = build_content(it, latent, kind, fmt, injector, st, photos=not args.no_photo)
                prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                       tokenize=False, enable_thinking=False) + ANSWER_PREFIX
                images = [c["image"] for c in content if c["type"] == "image"]
                inputs = processor(text=[prompt], images=images or None, return_tensors="pt",
                                   **(image_kwargs if images else {})).to(model.device)
                injector.vectors = vectors
                if slots:
                    slot_injector.slots = slots
                t0 = time.time()
                sc = dict(do_sample=True, temperature=args.temperature, num_return_sequences=args.samples) if args.samples > 1 else dict(do_sample=False)
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **sc)
                seconds = time.time() - t0
                votes = [parse_letter(processor.decode(g[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(),
                                      [k for k, _ in it["options"]]) for g in gen] if args.samples > 1 else None
                injector.vectors = None
                if slot_injector:
                    slot_injector.slots = {}
                reply = processor.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                letters = [k for k, _ in it["options"]]
                pred = parse_letter(reply, letters)
                if votes:
                    counts = collections.Counter(v for v in votes if v)
                    pred = max(counts, key=lambda x: (counts[x], -votes.index(x))) if counts else None
                row = {"bench": it["bench"], "id": it["id"], "type": it["type"], "options": it["options"],
                       "latent_from": None if kind == "baseline" else ("noise" if noise else src),
                       "answer": it["answer"], "reply": reply, "prediction": pred, "correct": pred == it["answer"],
                       "seconds": round(seconds, 2), "prompt_tokens": int(inputs["input_ids"].shape[1])}
                if votes:
                    row["votes"] = votes
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        metrics = summarize(rows)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        (out / "README.md").write_text(
            f"# {name}\n\nVLM: `{args.model}` (bf16, greedy, thinking off, reply pre-filled with "
            f"`{ANSWER_PREFIX.strip()}`, max {args.max_new_tokens} new tokens, latent slot `{placeholder}`"
            + (f", {args.max_soft_tokens} tokens per image" if args.max_soft_tokens else "") + f", attention `{args.attn}`)\n"
            f"Sample: `{args.sample}`\nPhotos: {'not shown' if args.no_photo else 'shown'}\n"
            f"Latent: {kind}" + (f", format `{fmt}` (see answer_qwen.py docstring)" if kind != "baseline" else "")
            + (" - **wrong latent**: each question gets the latent of another question (see `latent_from`)" if ctrl == "wrong"
               else " - **noise**: each latent is Gaussian noise with its views' per-channel mean and std" if ctrl == "noise"
               else "")
            + f"\n\nAccuracy: {metrics['accuracy']} on {metrics['n']} questions\n")
        print(name, json.dumps({b: v["accuracy"] for b, v in metrics["by_benchmark"].items()}), flush=True)


if __name__ == "__main__":
    main()
