#!/usr/bin/env python3
"""LatentParse: an analyst VLM parses the latent features of frozen vision encoders and writes a report; the answering
VLM answers from the photo(s) and that report.

Usage (the arguments of every paper run are in scripts/blend_job.sh and scripts/one_encoder_job.sh):
  python agent.py --sample DIR --latents-root DIR --model MODEL --out DIR --blind --boxes --single-pass \
      --index-chars 16000 --latents clip vggt dinov3 [--all-tools] [--marks] [--bench mmsi] [--shard i/n]

Per question:
  1. The analyst never sees the photo(s). It gets the question, the options and the latent features in a Python sandbox
     (sandbox.py) with the tools locate, locate_all, support, iou, motion, track, relate and layout.
     Agent-Directed (default): it writes code with these tools, at most --max-steps code steps, then replies with FINDINGS.
     Fixed-Sequence (--all-tools): every tool runs once with fixed arguments (blend_steps) and it writes FINDINGS from the
     output. Both modes run once, in a single pass.
  2. The report (FINDINGS: findings with a reliability each, a summary and boxes) goes to the answering VLM in a fresh
     context, with the photo(s) and the question, in the prompt of the photo-only baseline (answer_qwen.py).
     agent_only is LatentParse; the other answer runs (agent_reliable, agent_boxes, agent_consboxes, ...) vary what of
     the report the answering VLM gets.

Inputs:
  --latents clip vggt dinov3   the Multi-Latent features (any subset with VGGT or DINOv3)
  --latent clip|vggt|dinov3    one encoder alone with the basic tools (the + CLIP, + DINOv3 and + VGGT rows)
  --report-from question|photos, --pixels   the source controls: the report from the question, from the photos, or
                               from the photos' pixels on the CLIP and VGGT grids
  --wrong-latent, --noise-latent   the Mismatched and Random Noise controls
  --marks                      MV-RoboBench and SPAR-Bench-Tiny: the boxes or points drawn into the photos are listed
  --layer ENC:SPEC             the encoder-layer study
  --model gpt-6-luna | claude-haiku-5-5   a closed VLM over its API as analyst and answering VLM (api_chat.py)
  --answers-only RUN ... [--notes-header camera_head]   add answer runs to finished transcripts, e.g. the VGGT
                               camera poses written as text by cam_head_notes.py

Writes OUT/transcripts/<bench>_<id>.json (everything the analyst saw and did) and OUT/<answer run>/results.jsonl and
metrics.json. Questions with a finished transcript are skipped, so a job can be resubmitted after a crash.
"""
import argparse
import collections
import gc
import io
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "direct_input"))
from answer_qwen import (ANSWER_PREFIX, PLACEHOLDER_CANDIDATES, LatentInjector, build_content, latent_stats, summarize,
                         wrong_latent_donors)  # noqa: E402
from common import load_sample, parse_letter, question_text  # noqa: E402
from sandbox import PT_AGREE_HIGH, Sandbox, load_latent  # noqa: E402

import api_chat  # noqa: E402

ANALYST_BLIND = """You are the analyst in a two-agent system. You do NOT see the photo(s). You get a spatial question and the raw latent features of the photo(s) from two frozen vision models, loaded in a Python sandbox:
- F["clip"][v]: CLIP ViT-L/14, shape (16, 16, 1024): a 16x16 grid of patches over the whole of photo v (the photo was squeezed to a square, so the grid covers the full image). CLIP encodes what things are and links image regions to words.
- F["vggt"][v]: VGGT, shape (h, w, 2048): a grid of 14-pixel patches, 37 columns across the photo width. VGGT is a 3D reconstruction model; its features encode depth, surfaces and camera viewpoint, and for several photos of one scene they are computed jointly. vggt_valid[v] marks real image patches (False = padding).
Nothing is decoded: there is no depth map, no 3D points and no camera pose. Any evidence, including where each object is, must come from these features.

Your job is NOT to answer the question. Your job is to extract evidence from the latents that bears on it, check it, and say how reliable it is. A separate critic checks your work and a separate model answers.

Kinds of analysis (combine and adapt them, and write your own numpy/scipy code when useful). Every map or mask lives on one grid: CLIP (16, 16) or photo v's VGGT grid (h, w).
1. Locate the objects the question names from the latents. text_map(phrase, v) is a (16, 16) map of how much each patch matches a phrase. It is noisy: check that the object stands out as a compact blob, and compare with a related and an unrelated phrase. top_region(map, frac) picks a map's strongest blob; kmeans(model, v, k) splits the patches into segments; grow(model, seed_mask, v) extends a mask by feature similarity. box(x0, y0, x1, y1, model, v) selects the cells of a latent grid inside a rectangle in 0-1 coordinates across the grid (0 = left/top, 1 = right/bottom): use it only around a blob you found in a map, or for a coarse part of the image (e.g. the bottom quarter), never to guess where an object is. to_vggt(map_or_mask, v) and to_clip(map_or_mask, v) move maps and masks between grids (masks stay masks). Always show() a mask before relying on it.
2. Geometry statistics from VGGT: maps, ev = pca("vggt", v, k, mask) gives maps[i], the (h, w) map of component i, and explained variances; region_stats(map, mask, v) gives mean/std/median/area/centroid of a map inside a mask on the same grid. Latent directions have no fixed meaning or sign: calibrate a direction inside this scene first (does it change smoothly from the bottom of the image to the top, as floor to far wall would? does it separate an occluder from what it hides?) before reading it as depth, height or viewpoint.
3. Several photos: match(va, vb, model, mask_a) gives mutual nearest-neighbour patch matches from photo va to vb and their mean shift (x, y as fractions of the photo); sim_to(model, mask, v, mask_view) maps the similarity of photo v's patches to a region taken from photo mask_view.
4. Statistics: region_stats, corr(map_a, map_b, mask), kmeans(model, v, k), describe(map), grid(model, v), photo_size, n_views.
5. Look: show(map_or_mask, title, v) draws a map or mask on its own grid (there is no photo underneath) and attaches the picture to your next observation (2 per step).

Rules:
- Reply with exactly one ```python block per turn. Print short summaries (numbers, small dicts), never whole arrays. Variables persist between turns. Each execution gets an id (E1, E2, ...).
- Check regions before relying on them. Report each difference with its size relative to the spread, and say when it is within noise.
- PCA scores, similarities and cluster ids have arbitrary offset and scale. Never take ratios of them or turn them into metres, angles or "times farther": compare only orderings and differences relative to the spread.
- If an object could not be located reliably from the latents, say the comparison is inconclusive; do not draw it from a region that may be wrong.
- Every finding must come from your code on F (source "latent"). Do not describe what the scene probably looks like.
- When done, reply instead with a line starting with FINDINGS: followed by JSON (each finding at most 60 words):
{"findings": [{"text": "<finding with its numbers, the model and statistic, and the region>", "source": "latent", "evidence": ["E3"], "reliability": "high|medium|low", "why": "<reason for the reliability>"}], "summary": "<at most 80 words on what the latent measurements show that bears on the question; do not pick an option>"}"""

def _swap(text, pairs):
    """Apply exact replacements, each of which must match once (keeps derived prompts in step with their source)."""
    for old, new in pairs:
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    return text


SHOW_ON_GRID = [("overlay on photo v; you see it next turn", "drawn on its own grid (no photo underneath); you see it next turn"),
                ("overlays a map or mask on photo v and attaches", "draws a map or mask on its own grid (no photo underneath) and attaches")]

# --boxes (with --blind): the analyst also gives a box for each object it located in the latents; the boxes are drawn
# on the photo(s) for the model that answers.
ANALYST_BOXES = _swap(ANALYST_BLIND, [
    ("4. Statistics: region_stats, corr(map_a, map_b, mask),",
     "4. Statistics: region_stats, bbox(mask, v) (the [x0, y0, x1, y1] box around a mask in photo fractions), corr(map_a, map_b, mask),"),
    ("- When done, reply instead with a line starting with FINDINGS:",
     "- In \"regions\", give bbox(mask, v) for each object the question names that you located in the latents (leave out any you could not locate). They are drawn on the photo(s) for the model that answers, so a wrong box misleads it.\n- When done, reply instead with a line starting with FINDINGS:"),
    ('do not pick an option>"}', 'do not pick an option>", "regions": [{"label": "<object>", "view": 0, "box": [0.1, 0.2, 0.4, 0.6]}]}'),
])
# One pass: no critic and no second round.
SINGLE_PASS = [("A separate critic checks your work and a separate model answers.",
                "There is no critic and no second round: after a few code steps, give your FINDINGS. A separate model answers.")]
# --latent clip|vggt|dinov3: the analyst gets that latent alone.
ONLY_NOTE = {
    "clip": ("\n\nIn this run only the CLIP latent is loaded: F has no \"vggt\" entry, so pca, kmeans, grow, sim_to and match "
             "take model=\"clip\"."),
    "vggt": ("\n\nIn this run only the VGGT latent is loaded: F has no \"clip\" entry and text_map is not available (it needs "
             "CLIP), so locate objects with kmeans, pca, grow and sim_to on model=\"vggt\"."),
    "dinov3": ("\n\nIn this run only the DINOv3 latent is loaded: F[\"dinov3\"][v] holds DINOv3 ViT-L/16 patch tokens, shape "
               "(h, 37, 1024), on exactly the VGGT grid (vggt_valid[v] is False on padding patches). F has no \"clip\" or "
               "\"vggt\" entry and text_map is not available (it needs CLIP), so locate objects with kmeans, pca, grow and "
               "sim_to on model=\"dinov3\"."),
}
def one_encoder_prompt(latent):
    """The analyst's system prompt in a --latent run (one encoder alone, the + CLIP, + DINOv3 and + VGGT rows)."""
    text = _swap(ANALYST_BOXES, SINGLE_PASS) + ONLY_NOTE[latent]
    for a, b in SHOW_ON_GRID:
        text = text.replace(a, b)
    return text


BOXES_HEADER = ("\nThe boxes drawn on the photo(s) mark objects that another model located from the latent features alone, "
                "without seeing the photo(s). They may be wrong.\n")

# --all-tools with CLIP among --latents: the rule box per object (not shown to the analyst), the strongest text_map
# blob in the photo where the map peaks highest; logged for the agent_ruleboxes answer runs.
RULE_CODE = '''best = []
for o in objects:
    ms = [text_map(o, v) for v in range(n_views)]
    v = max(range(n_views), key=lambda i: ms[i].max())
    best.append({"object": o, "view": v, "box": bbox(top_region(ms[v]), v)})
print(repr(best))'''

# --latent vggt --all-tools: the Fixed-Sequence twin of the Agent-Directed one-encoder run. Same system prompt, sandbox (VGGT only,
# the basic tools, show() on the latent's own grid), answer runs and arguments; only the code differs: every basic tool
# runs once with fixed arguments (basic_steps) instead of the analyst choosing, and the maps they show are attached.
# The rule for "regions" is the system prompt's own (only objects located in the latents): the boxes basic_steps prints
# are segments and blobs, not named objects.
BASIC_ASK = ("Every sandbox tool below was already run once on the latents, with fixed arguments (the code and its output; "
             "the maps they drew with show() follow the output). Do not write code. Using only this evidence, reply with "
             "FINDINGS: and the JSON; cite the execution ids and say what is uncertain. In \"regions\", give bbox(mask, v) "
             "for each object the question names that you located in the latents (leave out any you could not locate).\n\n")


def basic_steps(n_views, model="vggt"):
    """The fixed steps of a --latent MODEL --all-tools run: {id: code}, in priority order (the evidence is cut from the
    end). E1 matches between photos (several photos only), E2 k-means segments, E3 principal components, E4 one seed
    region with the remaining tools. Every basic tool but text_map (it needs CLIP) runs at least once. CLIP and DINOv3
    get the VGGT steps word for word on their own latent, so the one-latent rows differ only in the latent:
    DINOv3 sits on the VGGT grid and keeps vggt_valid; CLIP's 16x16 grid has no padding, so every cell counts."""
    steps = _basic_steps_vggt(n_views)
    if model == "vggt":
        return steps
    name = {"clip": "CLIP", "dinov3": "DINOv3"}[model]

    def port(code):
        code = code.replace('"vggt"', f'"{model}"').replace("'vggt'", f"'{model}'").replace("VGGT", name)
        return re.sub(r"vggt_valid\[[^\]]+\]", "np.ones((16, 16), bool)", code) if model == "clip" else code

    steps = {k: port(c) for k, c in steps.items()}
    if model == "clip":  # E4's last line moves the region to the other grid and back: the VGGT grid for CLIP
        steps["E4"] = steps["E4"].replace(
            "on CLIP's 16x16 grid the grown region covers {to_clip(g, v).mean():.2f} of the photo; back on the CLIP grid ",
            "on VGGT's grid the grown region covers {to_vggt(g, v).mean():.2f} of the cells; back on CLIP's 16x16 grid "
        ).replace("{(to_vggt(to_clip(g, v), v) & g).sum() / g.sum():.2f}", "{(to_clip(to_vggt(g, v), v) & g).sum() / g.sum():.2f}")
    return steps


def _basic_steps_vggt(n_views):
    """basic_steps on VGGT, the code the VGGT-only fixed runs used."""
    steps = {}
    if n_views > 1:
        steps["E1"] = (
            "pairs = sorted(((a, b) for a in range(n_views) for b in range(a + 1, n_views)), key=lambda p: (p[1] - p[0], p))[:7]\n"
            "for a, b in pairs:  # VGGT: mutual nearest-neighbour patch matches between photos (real patches only), mean shift\n"
            "    print(f\"photo {a + 1} -> {b + 1}:\", match(a, b, \"vggt\", mask_a=vggt_valid[a]))\n"
            "lab = kmeans(\"vggt\", 0, k=5, mask=vggt_valid[0])\n"
            "for c in range(5):  # photo 1's k-means segments (as in E2): their matches in photo 2, and photo 2's most similar region\n"
            "    m = lab == c\n"
            "    if m.any():\n"
            "        t = top_region(np.where(vggt_valid[1], sim_to(\"vggt\", m, 1, mask_view=0), -1.0))\n"
            "        print(f\"photo 1 segment {c} -> photo 2: {match(0, 1, 'vggt', mask_a=m)}; most similar region in photo 2: \"\n"
            "              f\"box {bbox(t, 1)}\")")
    steps["E2"] = (
        "for v in range(min(n_views, 4)):  # VGGT: k-means segments of each photo (real patches only)\n"
        "    lab = kmeans(\"vggt\", v, k=5, mask=vggt_valid[v])\n"
        "    parts = []\n"
        "    for c in range(5):\n"
        "        m = lab == c\n"
        "        if m.any():\n"
        "            s = region_stats(m.astype(float), m, v)\n"
        "            parts.append(f\"{c}: box {bbox(m, v)}, area {m.sum() / vggt_valid[v].sum():.2f}, centroid {s['centroid_xy']}\")\n"
        "    print(f\"photo {v + 1} segments: \" + \"; \".join(parts))\n"
        "    if v < 2:\n"
        "        show(lab, f\"photo {v + 1}: VGGT k-means segments 0-4\", v)")
    steps["E3"] = (
        "for v in range(min(n_views, 4)):  # VGGT principal components; component 1 against height in the image\n"
        "    maps, ev = pca(\"vggt\", v, k=3)\n"
        "    pc1 = np.where(vggt_valid[v], maps[0], float(maps[0][vggt_valid[v]].min()))\n"
        "    h, w = grid(\"vggt\", v)\n"
        "    rows = np.repeat(np.linspace(0, 1, h)[:, None], w, 1)\n"
        "    lo, hi = region_stats(pc1, box(0, 0.5, 1, 1, \"vggt\", v), v), region_stats(pc1, box(0, 0, 1, 0.5, \"vggt\", v), v)\n"
        "    print(f\"photo {v + 1}: explained variance {ev}; PC1 {describe(pc1)}; corr(PC1, row from top) \"\n"
        "          f\"{corr(pc1, rows, vggt_valid[v]):.2f}; PC1 mean lower half {lo['mean']:.2f} vs upper half {hi['mean']:.2f}; \"\n"
        "          f\"PC1 top region box {bbox(top_region(pc1), v)}\")\n"
        "    if v < 2:\n"
        "        show(pc1, f\"photo {v + 1}: VGGT principal component 1\", v)")
    steps["E4"] = (
        "v = 0  # one seed region: photo 1's smallest k-means segment (as in E2) covering at least 3% of the photo\n"
        "lab = kmeans(\"vggt\", v, k=5, mask=vggt_valid[v])\n"
        "sizes = {c: int((lab == c).sum()) for c in range(5)}\n"
        "c = min((c for c in sizes if sizes[c] >= 0.03 * vggt_valid[v].sum()), key=sizes.get)\n"
        "seed = lab == c\n"
        "s = sim_to(\"vggt\", seed, v)\n"
        "g = grow(\"vggt\", seed, v) & vggt_valid[v]\n"
        "print(f\"grid {grid('vggt', v)}; seed = segment {c}, box {bbox(seed, v)}, area {seed.sum() / vggt_valid[v].sum():.2f}\")\n"
        "print(f\"grown from the seed: box {bbox(g, v)}, area {g.sum() / vggt_valid[v].sum():.2f}; similarity to the seed \"\n"
        "      f\"inside it {region_stats(s, g, v)}\")\n"
        "print(f\"similarity to the seed: {describe(np.where(vggt_valid[v], s, float(s[vggt_valid[v]].min())))}\")\n"
        "print(f\"on CLIP's 16x16 grid the grown region covers {to_clip(g, v).mean():.2f} of the photo; back on the VGGT grid \"\n"
        "      f\"it keeps {(to_vggt(to_clip(g, v), v) & g).sum() / g.sum():.2f} of its cells\")")
    return steps

# --latents (the blend: any of clip, vggt, dinov3 with VGGT or DINOv3 among them): one analyst prompt and one sandbox
# namespace (old tools + support, iou, locate, locate_all, motion, track, relate, layout; no show(), so no picture reaches
# the analyst) for Agent-Directed and Fixed-Sequence (--all-tools) runs alike. Fixed runs get blend_steps() instead of writing code.
NAME = {"clip": "CLIP", "vggt": "VGGT", "dinov3": "DINOv3"}
BLEND_LATENT = {
    "clip": '- F["clip"][v]: CLIP ViT-L/14 patch tokens, shape (16, 16, 1024): a 16x16 grid over the whole photo (squeezed '
            'to a square). CLIP links image regions to words.',
    "vggt": '- F["vggt"][v]: VGGT aggregator tokens, shape (h, 37, 2048). VGGT is a 3D reconstruction model: its features '
            'encode surfaces, depth and viewpoint, computed jointly for all photos of a scene.',
    "dinov3": '- F["dinov3"][v]: DINOv3 ViT-L/16 patch tokens, shape (h, 37, 1024), on exactly the VGGT grid (same cells, '
              'same vggt_valid). DINOv3 is a self-supervised image model: features of objects, parts and surfaces, computed '
              'for each photo on its own (it never sees the other photos and has no notion of depth).',
}


# --layer ENC:SPEC (layer ablation, set in main): F[ENC] holds block SPEC (1-24) or, with "cat", blocks 5, 12, 18 and 24,
# each L2-normalized per patch and concatenated, instead of the last block; the analyst's description of F says which.
LAYER = {}
LAST_DIM = {"clip": 1024, "vggt": 2048, "dinov3": 1024}


def blend_latent_line(k):
    """BLEND_LATENT[k], with the block and the channel count of a --layer run."""
    if k not in LAYER:
        return BLEND_LATENT[k]
    spec, d = LAYER[k], LAST_DIM[k]
    what = ("blocks 5, 12, 18 and 24 of 24, each L2-normalized per patch and concatenated" if spec == "cat"
            else f"block {spec} of 24 instead of the last")
    line = BLEND_LATENT[k]
    for end in (":", ".", ","):
        line = line.replace(f", {d}){end}", f", {4 * d if spec == 'cat' else d}){end}")
    return f"{line} Here F[\"{k}\"] holds {what}."


def blend_prompt(use, fixed, see=False):
    """The analyst's system prompt in a --latents run (`use`: the latents in F; fixed: --all-tools; see: --see-photos)."""
    geo = [k for k in ("vggt", "dinov3") if k in use]
    shared = "VGGT and DINOv3 share a grid of" if len(geo) == 2 else f"{NAME[geo[0]]}'s grid has"
    lines = ["You are the analyst. You see the photo(s) and a spatial question, and you get the raw latent features of "
             "the photo(s) from frozen vision models, loaded in a Python sandbox:" if see else
             "You are the analyst. You do NOT see the photo(s). You get a spatial question and the raw latent features of "
             "the photo(s) from frozen vision models, loaded in a Python sandbox:"]
    lines += [blend_latent_line(k) for k in ("clip", "vggt", "dinov3") if k in use]
    lines += ["Nothing is decoded: there is no depth map, no 3D points and no camera pose. "
              + ("CLIP's grid is 16x16 over the whole photo; " if "clip" in use else "")
              + f"{shared} 14-pixel cells, 37 columns across the photo width (a tall photo is centre-cropped; "
              "vggt_valid[v] is False on padding rows). Positions are photo fractions: x from the left, y from the top, both "
              "0-1. Printouts number the photos 1, 2, ...; in code and in \"regions\", view = photo number - 1.",
              "",
              "Your job is NOT to answer. Measure evidence in the latents that bears on the question, check it across "
              "latents, and say how reliable it is. There is no critic and no second round; a separate model answers from "
              "the photo(s) and your findings."
              + (" Use the photo(s) to see what the question refers to and where those objects are (for example to give "
                 "box() their position), then measure with the latents. The model that answers sees the same photo(s), "
                 "so do not report what you see in them: every finding must come from latent numbers in the code output."
                 if see else ""),
              "",
              "Tools that compare latents (start with these):"]
    if "clip" in use:
        lines.append("- locate(phrase, v=None): CLIP's text map (already negated, so the named object scores highest) finds "
                     "the object in the photo where it peaks (or in photo v); " + " and ".join(NAME[k] for k in geo)
                     + " each check on their own which cells belong with CLIP's blob. Prints photo, tier, box, area, "
                     "centroid, peak, standout (peak minus the best value elsewhere), keep (share of the core of CLIP's blob "
                     "each latent keeps)" + (", iou_vggt_dinov3" if len(geo) == 2 else "") + " and peaks per photo. tier is high, "
                     "medium, low or none. locate_all(phrases) runs it for each phrase and marks two names that land on "
                     "one region (same_as).")
    lines.append("- motion(va, vb): how the content moves from photo va to photo vb in each latent, from mutual patch "
                 "matches: n, dx, dy (median shift of the moving matches; + = right / down), static (share not moving), "
                 "inl (share that one shift + scale + rotation explains; noise stays under 0.2), scale (> 1: content "
                 "grows), rot (degrees). reliable lists the latents with enough consistent matches (an unreliable one "
                 "shows only n and inl); x, y and zoom give the direction of the reliable latents and how many agree "
                 "(\"left 3/3\", \"left 2/3 (clip right)\", \"left (only vggt)\", \"disagree\", \"mixed\", \"unreliable\")"
                 + ("; pt_agree is the share of cells VGGT and DINOv3 both match whose targets lie within one cell of each "
                    "other (chance is about 0.01)." if len(geo) == 2 else "."))
    lines.append("- track(region, vb): where a located region (a locate result) lands in photo vb in each geometry latent "
                 "(n, at, spread); found yes / no / unclear; only with yes: shift, beyond_camera (shift minus the camera "
                 "motion: the object's own movement, or parallax)"
                 + (" and text_at vs text_peak (CLIP's map at the landing place vs its peak there)." if "clip" in use else "."))
    lines.append("- relate(a, b): region b relative to region a in one photo: dx, dy between centroids, size (box area "
                 "ratio), iou, touch.")
    lines.append("- layout(v): k-means segments " + ("VGGT and DINOv3 share" if len(geo) == 2 else f"of {NAME[geo[0]]}")
                 + ", and cka: how alike each pair of latents' maps are (0 unrelated, 1 the same structure).")
    lines.append("- support(model, seed, v): the cells a latent itself groups with a seed region (the seed is not forced "
                 "in). iou(a, b, v): overlap of two masks of photo v (either grid).")
    lines.append("Single-latent tools: " + ("text_map(phrase, v, pooled=False), " if "clip" in use else "")
                 + "top_region(map, frac), sim_to(model, mask, v, mask_view), grow(model, seed, v) (its result always "
                 "contains the seed, so it never confirms a region), kmeans(model, v, k, mask), pca(model, v, k, mask), "
                 "match(va, vb, model, mask_a), region_stats(map, mask, v), corr(map_a, map_b, mask), box(x0, y0, x1, y1, "
                 "model, v), bbox(mask, v), to_vggt / to_clip(map_or_mask, v), describe(map), grid(model, v); photo_size, "
                 "n_views and latents (the latents in F). numpy and scipy also work.")
    if "clip" not in use:
        lines.append("There is no CLIP latent in this run, so no object can be found by name (text_map, locate and "
                     "locate_all are not available): report camera motion and segments, and leave \"regions\" empty.")
    if len(geo) == 1:
        lines.append(f"Only one geometry latent ({NAME[geo[0]]}) is in this run, so nothing is cross-checked between VGGT "
                     "and DINOv3: locate tiers stop at medium, and a direction or a track from one latent is at most medium.")
    lines += ["",
              "Rules for reading the numbers:",
              "1. Rely on a location, a motion or a relation only when the latents agree, and set each finding's reliability "
              "from that agreement:",
              "   - high: locate tier high; a motion direction that every reliable latent gives, with at least two of them "
              "(\"left 3/3\", \"left 2/2\")" + (f" and pt_agree >= {PT_AGREE_HIGH} when VGGT and DINOv3 are both reliable" if len(geo) == 2
                                                 else "") + "; track found yes.",
              "   - medium: locate tier medium; a direction with a dissenting latent (\"left 2/3 (...)\") or from one latent "
              "(\"only ...\"); track \"yes (only ...)\".",
              "   - low: anything else. \"disagree\", \"mixed\", \"unreliable\", tier low or none, and track no or unclear are "
              "not evidence: say the latents are inconclusive on that point. Never settle a disagreement by picking the "
              "latent you prefer.",
              "2. Agreement shows that a region is one coherent thing in several latents, not that it is the named object; "
              "that rests on CLIP's peak and standout.",
              "3. Camera motion: content moving right from photo a to photo b means the camera turned or moved left; "
              "content moving down means it tilted or moved up; zoom \"grows\" means it moved toward the scene. A shift "
              "under one cell (0.027) is no movement.",
              "4. State a relation between two objects only when both have tier high or medium; left/right and above/below "
              "come from their centroids in one photo.",
              "5. The latents carry no reliable signal for orientation: which way an object faces or points, or whether "
              "two objects are parallel or face the same way. Do not infer these; say they are not measured.",
              "6. PCA scores, similarities and cluster ids have arbitrary offset and scale: compare orderings and "
              "differences relative to the spread, never ratios, metres or angles.",
              "7. If nothing reaches medium reliability, say the latents are inconclusive. Every finding must come from "
              "the latent numbers (source \"latent\"); do not describe what the scene probably looks like.",
              ""]
    if fixed:
        lines.append("In this run the tools have already been run for you with fixed arguments; their code and output "
                     "follow in the next message. Do not write code.")
    else:
        lines.append("Write Python: exactly one ```python block per turn, at most 4 code steps; variables persist between "
                     "steps and each run gets an id (E1, E2, ...). Print short summaries (the tools above print compact "
                     "lines), never whole arrays. Then give your FINDINGS.")
    lines += ["In \"regions\", give the box of each object the question names whose tier is high, or medium when track "
              "found it in another photo (found yes); view = photo number - 1. Leave out every other object. The boxes are drawn on the "
              "photo(s) for the model that answers, so a wrong box misleads it.",
              "When done, reply with a line starting with FINDINGS: followed by JSON (at most 6 findings, most decisive "
              "first, each at most 60 words):",
              '{"findings": [{"text": "<finding with its numbers, the tool and the latents>", "source": "latent", '
              '"evidence": ["E2"], "reliability": "high|medium|low", "why": "<which latents agree or disagree>"}], '
              '"summary": "<at most 80 words on what the latents show that bears on the question; do not pick an option>", '
              '"regions": [{"label": "<object>", "view": 0, "box": [0.1, 0.2, 0.4, 0.6]}]}']
    return "\n".join(lines)


# --pixels with --latents clip vggt --see-photos (photo-only control of the --see-photos variant): the blend
# prompt of a CLIP + VGGT run that sees the photos, with only the input changed, so the method (single pass, at most 4 code
# steps, the same tools and rules) is the same and the numbers come from the photo's pixels on the same two grids.
# locate, locate_all and text_map need CLIP features, so objects are placed with box() from the marks or the photos, and
# "regions" stays empty (nothing is located from features). BLEND_PIXELS holds the pairs of every pixel run; those that
# depend on the analyst not seeing the photo(s) are in BLEND_PIXELS_BLIND.
BLEND_PIXELS = [
    (BLEND_LATENT["clip"],
     '- F["clip"][v]: shape (16, 16, 588): a 16x16 grid of patches over the whole photo (squeezed to a square); each patch '
     'holds its 14x14 RGB pixels, values 0-1. It is named after CLIP\'s grid but holds pixels, not features.'),
    (BLEND_LATENT["vggt"],
     '- F["vggt"][v]: shape (h, 37, 588): a grid of 14-pixel patches, 37 columns across the photo width; each patch holds '
     'its 14x14 RGB pixels, values 0-1. It is named after VGGT\'s grid but holds pixels, not features.'),
    ("Nothing is decoded: there is no depth map, no 3D points and no camera pose. CLIP's grid is 16x16 over the whole "
     "photo; VGGT's grid has",
     "These are pixels only: there are no model features, no depth map, no 3D points and no camera pose. The clip grid "
     "is 16x16 over the whole photo; the vggt grid has"),
    ("Measure evidence in the latents that bears on the question, check it across latents,",
     "Measure evidence in the pixel grids that bears on the question, check it across the two grids,"),
    ("Tools that compare latents (start with these):", "Tools that compare the grids (start with these):"),
    ("in each latent, from mutual patch matches", "in each grid, from mutual patch matches"),
    ("reliable lists the latents with enough", "reliable lists the grids with enough"),
    ("the direction of the reliable latents and how many agree", "the direction of the reliable grids and how many agree"),
    ('("left 3/3", "left 2/3 (clip right)", "left (only vggt)", "disagree", "mixed", "unreliable")',
     '("left 2/2", "left (only vggt)", "disagree", "mixed", "unreliable")'),
    ("- track(region, vb): where a located region (a locate result) lands in photo vb in each geometry latent",
     "- track(mask, vb, va): where a region of photo va (a mask, e.g. from box()) lands in photo vb on the vggt grid"),
    (" and text_at vs text_peak (CLIP's map at the landing place vs its peak there).", "."),
    ("- relate(a, b): region b relative to region a in one photo:", "- relate(a, b, v): region b relative to region a (two masks of photo v):"),
    ("k-means segments of VGGT, and cka: how alike each pair of latents' maps are",
     "k-means segments of the vggt grid, and cka: how alike the two grids' maps are"),
    ("the cells a latent itself groups with a seed region", "the cells a grid itself groups with a seed region"),
    ("Single-latent tools: text_map(phrase, v, pooled=False), top_region", "Single-grid tools: top_region"),
    ("latents (the latents in F)", "latents (the grids in F)"),
    ("only when the latents agree", "only when the grids agree"),
    ('   - high: locate tier high; a motion direction that every reliable latent gives, with at least two of them '
     '("left 3/3", "left 2/2"); track found yes.',
     '   - high: a motion direction that both grids give ("left 2/2").'),
    ('   - medium: locate tier medium; a direction with a dissenting latent ("left 2/3 (...)") or from one latent '
     '("only ..."); track "yes (only ...)".',
     '   - medium: a direction from one grid ("only ..."); track "yes (only vggt)".'),
    ('"unreliable", tier low or none, and track no or unclear are not evidence: say the latents are inconclusive on '
     'that point. Never settle a disagreement by picking the latent you prefer.',
     '"unreliable", and track no or unclear are not evidence: say the pixels are inconclusive on that point. Never '
     'settle a disagreement by picking the grid you prefer.'),
    ("5. The latents carry no reliable signal", "5. The pixel grids carry no reliable signal"),
    ('say the latents are inconclusive. Every finding must come from the latent numbers (source "latent")',
     'say the pixels are inconclusive. Every finding must come from the pixel numbers (source "pixel")'),
    ('In "regions", give the box of each object the question names whose tier is high, or medium when track found it in '
     'another photo (found yes); view = photo number - 1. Leave out every other object. The boxes are drawn on the '
     'photo(s) for the model that answers, so a wrong box misleads it.',
     'Leave "regions" empty: no object is located from features in this run.'),
    ('"<finding with its numbers, the tool and the latents>", "source": "latent"',
     '"<finding with its numbers, the tool and the grids>", "source": "pixel"'),
    ('"<which latents agree or disagree>"', '"<which grids agree or disagree>"'),
    ('"<at most 80 words on what the latents show', '"<at most 80 words on what the pixel numbers show'),
    ('"regions": [{"label": "<object>", "view": 0, "box": [0.1, 0.2, 0.4, 0.6]}]}', '"regions": []}'),
]
# Without --see-photos (the pixel control of the blind method) the analyst has no photo: the same sentences
# with the photo left out, and no object can be found by name, as a blend run without CLIP is told.
BLEND_PIXELS_BLIND = [
    ("You get a spatial question and the raw latent features of the photo(s) from frozen vision models, loaded in a "
     "Python sandbox:",
     "You get a spatial question and the photo(s) as raw pixels, cut into two patch grids and loaded in a Python sandbox:"),
    ("Only one geometry latent (VGGT) is in this run, so nothing is cross-checked between VGGT and DINOv3: locate tiers "
     "stop at medium, and a direction or a track from one latent is at most medium.",
     "text_map, locate and locate_all are not available (they need CLIP features), so no object can be found by name: "
     "place objects with box() from the marks, if any are listed, and report camera motion and segments. track and "
     "layout run on the vggt grid alone, so a track is at most medium."),
    ("one coherent thing in several latents, not that it is the named object; that rests on CLIP's peak and standout.",
     "one coherent thing in both grids, not that it is the named object; that rests on where the mark puts it."),
    ("only when both have tier high or medium;", "only when both are placed by a mark;"),
]


def blend_prompt_pixels():
    """The analyst's system prompt in a --pixels --latents clip vggt run (the Pixels source control, Agent-Directed)."""
    text = blend_prompt(["clip", "vggt"], fixed=False)
    start = text.index("- locate(phrase, v=None):")
    text = text[:start] + text[text.index("(same_as).\n", start) + len("(same_as).\n"):]  # locate needs CLIP features
    return _swap(text, BLEND_PIXELS + BLEND_PIXELS_BLIND)


# --report-from question | photos: what does the gain of a blind --latents run come from? The analyst has no
# latents, no sandbox and no code, and writes its report in one reply, from the question and options alone (question) or
# from them and the photo(s) (photos). Its prompt is made of the blend prompt's own lines (REPORT_LINES: the opening, the
# job, where a finding comes from, the code, the boxes and the reply) with only the input changed (REPORT_SWAPS); the
# latents, the tools and the rules for reading their numbers are left out.
REPORT_LINES = ("Your job is NOT to answer.", "7. If nothing reaches medium reliability", "Write Python:", 'In "regions",',
                "When done, reply", '{"findings":')
REPORT_REPLY = [  # the same for both: no code, no boxes, one reply
    ("Write Python: exactly one ```python block per turn, at most 4 code steps; variables persist between steps and each "
     "run gets an id (E1, E2, ...). Print short summaries (the tools above print compact lines), never whole arrays. Then "
     "give your FINDINGS.",
     "Do not write code: give your FINDINGS in one reply."),
    ('In "regions", give the box of each object the question names whose tier is high, or medium when track found it in '
     'another photo (found yes); view = photo number - 1. Leave out every other object. The boxes are drawn on the '
     'photo(s) for the model that answers, so a wrong box misleads it.',
     'Leave "regions" empty: no object is located in this run.'),
    ("When done, reply with a line", "Reply with a line"),
    (', "evidence": ["E2"]', ""),
    ('"regions": [{"label": "<object>", "view": 0, "box": [0.1, 0.2, 0.4, 0.6]}]}', '"regions": []}'),
]
REPORT_SWAPS = {
    "question": REPORT_REPLY + [
        ("You get a spatial question and the raw latent features of the photo(s) from frozen vision models, loaded in a "
         "Python sandbox:",
         "You get a spatial question only: no latent features of the photo(s), no sandbox and no tools."),
        ("Measure evidence in the latents that bears on the question, check it across latents, and say how reliable it is.",
         "Reason from the question alone about what bears on the answer, and say how reliable it is."),
        ('7. If nothing reaches medium reliability, say the latents are inconclusive. Every finding must come from the '
         'latent numbers (source "latent")',
         'Every finding must come from the question and its options (source "question")'),
        ('"<finding with its numbers, the tool and the latents>", "source": "latent"', '"<finding>", "source": "question"'),
        ('"<which latents agree or disagree>"', '"<what in the question it rests on>"'),
        ("on what the latents show that bears on the question", "on what the question itself tells that bears on the answer"),
    ],
    "photos": REPORT_REPLY + [
        (", and you get the raw latent features of the photo(s) from frozen vision models, loaded in a Python sandbox:",
         ", and you get no latent features of the photo(s), no sandbox and no tools."),
        ("Measure evidence in the latents that bears on the question, check it across latents, and say how reliable it is.",
         "Report what you see in the photo(s) that bears on the question, and say how reliable it is."),
        ('7. If nothing reaches medium reliability, say the latents are inconclusive. Every finding must come from the '
         'latent numbers (source "latent"); do not describe what the scene probably looks like.',
         'Every finding must come from what you see in the photo(s) (source "photo").'),
        ('"<finding with its numbers, the tool and the latents>", "source": "latent"', '"<finding>", "source": "photo"'),
        ('"<which latents agree or disagree>"', '"<what in the photo(s) shows it>"'),
        ("on what the latents show that bears on the question", "on what the photo(s) show that bears on the question"),
    ],
}


def report_prompt(source):
    """The analyst's system prompt in a --report-from run (source: question or photos)."""
    blend = ["clip", "vggt", "dinov3"]
    first = blend_prompt(blend, fixed=False, see=source == "photos").split("\n")[0]
    job, rule, code, regions, reply, form = [ln for ln in blend_prompt(blend, fixed=False).split("\n")
                                             if ln.startswith(REPORT_LINES)]
    return _swap("\n".join([first, "", job, rule, "", code, regions, reply, form]), REPORT_SWAPS[source])


TOOL_LINES = {"- locate(": {"locate", "locate_all"}, "- motion(": {"motion"}, "- track(": {"track"},
              "- relate(": {"relate"}, "- layout(": {"layout"}, "- support(": {"support", "iou"}, "Single-latent tools:": {"basic"}}


def filter_tool_lines(prompt, only, drop):
    """--only-tools / --drop-tools: leave out the prompt lines of the tools the analyst does not get."""
    if not only and not drop:
        return prompt
    keep = set(only) if only else None
    out = []
    for line in prompt.split("\n"):
        tools = next((v for k, v in TOOL_LINES.items() if line.startswith(k)), None)
        if tools is not None and (keep is not None and not tools & keep or drop and tools <= set(drop)):
            continue
        out.append(line)
        if line.startswith("Tools that compare latents"):
            out.append("Only the tools listed below are available in this run.")
    return "\n".join(out)


BLEND_ASK = ("The sandbox tools were run once on the latents with fixed arguments; the code and output of each block "
             "follow (blocks that do not apply to this question were not run): E1 camera motion between photos, E2 the "
             "named objects, E3 relations between located objects in one photo, E4 located objects carried into the other "
             "photos, E5 how the latents' maps relate in each photo, E6 the remaining tools once. Using only this evidence "
             "and the rules above, reply with FINDINGS: and the JSON, citing the execution ids. Copy boxes into \"regions\" "
             "only from E2.\n\n")


def blend_steps(use, n_views):
    """The fixed steps of a --latents --all-tools run: {id: code}, in priority order (the evidence is cut from the end).
    Ids are fixed (E1 motion, E2 objects, E3 relations, E4 tracking, E5 layout, E6 the remaining tools) and a step that
    does not apply (one photo; no CLIP) is left out."""
    geo = [k for k in ("vggt", "dinov3") if k in use]
    clip, multi = "clip" in use, n_views > 1
    steps = {}
    if multi:
        steps["E1"] = (
            "pairs = sorted(((a, b) for a in range(n_views) for b in range(a + 1, n_views)), key=lambda p: (p[1] - p[0], p))[:7]\n"
            "for a, b in pairs:  # camera motion: patch matches in each latent, and whether the latents agree\n"
            "    print(f\"photo {a + 1}->{b + 1}:\", motion(a, b))")
    if clip:
        steps["E2"] = (
            "found = locate_all(objects)\n"
            "for o, r in found.items():  # CLIP finds each object; " + " and ".join(NAME[k] for k in geo)
            + " each check which cells belong with it\n"
            "    print(f'\"{o}\":', r)")
        steps["E3"] = (
            "ok = {o: r for o, r in found.items() if r[\"tier\"] in (\"high\", \"medium\")}\n"
            "print(\"located (tier high or medium):\", list(ok) or \"none\")\n"
            "names = list(ok)\n"
            "for i, a in enumerate(names):  # pairs of located objects in one photo: b relative to a\n"
            "    for b in names[i + 1:]:\n"
            "        ra, rb = ok[a], ok[b]\n"
            "        if rb[\"view\"] != ra[\"view\"]:  # look for b in a's photo, else for a in b's photo\n"
            "            rb = locate(b, ra[\"view\"])\n"
            "            if rb[\"tier\"] not in (\"high\", \"medium\"):\n"
            "                ra, rb = locate(a, ok[b][\"view\"]), ok[b]\n"
            "        if ra[\"tier\"] in (\"high\", \"medium\") and rb[\"tier\"] in (\"high\", \"medium\"):\n"
            "            print(f'photo {ra[\"view\"] + 1}, \"{b}\" relative to \"{a}\":', relate(ra, rb))\n"
            "        else:\n"
            "            print(f'\"{a}\" and \"{b}\": not located together in one photo')")
        if multi:
            steps["E4"] = (
                "lines = 0\n"
                "for o, r in ok.items():  # each located object carried into the other photos by patch matches\n"
                "    for vb in sorted((u for u in range(n_views) if u != r[\"view\"]), key=lambda u: (abs(u - r[\"view\"]), u))[:3]:\n"
                "        if lines < 8:\n"
                "            print(f'\"{o}\" photo {r[\"view\"] + 1}->{vb + 1}:', track(r, vb))\n"
                "            lines += 1")
    steps["E5"] = ("for v in range(min(n_views, 3)):  # how the latents' maps relate: shared segments, CKA between latents\n"
                   "    print(layout(v))")
    g = geo[0]
    code = ["# the remaining tools, each once: on the first object's CLIP blob, or on the centre of photo 1"]
    if clip:
        code += ["v = found[objects[0]][\"view\"] if objects else 0",
                 "m = text_map(objects[0] if objects else \"an object\", v)",
                 "blob = top_region(m)",
                 "seed = to_vggt(blob, v) & vggt_valid[v]",
                 f"seed = seed if seed.sum() >= 2 else box(0.4, 0.4, 0.6, 0.6, \"{g}\", v)",
                 "print(\"text_map:\", describe(m), \"| blob:\", region_stats(m, blob, v))"]
    else:
        code += ["v = 0",
                 f"pc = pca(\"{g}\", v, 1)[0][0]",
                 "print(\"first principal component:\", describe(pc), \"| its top blob:\", region_stats(pc, top_region(pc), v))",
                 f"seed = box(0.4, 0.4, 0.6, 0.6, \"{g}\", v)"]
    code += [f"lower = box(0, 0.5, 1, 1, \"{g}\", v)",
             "print(\"seed box\", bbox(seed, v), \"| seed vs its round trip through the CLIP grid, IoU\", "
             "iou(seed, to_vggt(to_clip(seed, v), v), v))"]
    for k in geo:
        code.append(f"print(\"{k}: grid\", grid(\"{k}\", v), \"| grow area\", round(float(grow(\"{k}\", seed, v).mean()), 3), "
                    f"\"vs support area\", round(float(support(\"{k}\", seed, v).mean()), 3), \"| PCA ev\", pca(\"{k}\", v, 3)[1], "
                    f"\"| k-means sizes\", sorted(np.bincount(kmeans(\"{k}\", v, 5, mask=vggt_valid[v])[vggt_valid[v]]).tolist(), reverse=True))")
    sims = [f"sim_to(\"{k}\", seed, v)" for k in geo]
    pairs_c = [f"\"vggt~dinov3\": round(corr({sims[0]}, {sims[1]}, vggt_valid[v]), 2)"] if len(geo) == 2 else []
    if clip:
        pairs_c.append(f"\"clip text_map~{g}\": round(corr(to_vggt(m, v), {sims[0]}, vggt_valid[v]), 2)")
    else:
        pairs_c.append(f"\"{g} PC1~{g}\": round(corr(pc, {sims[0]}, vggt_valid[v]), 2)")
    code.append(f"print(\"corr of maps\", {{{', '.join(pairs_c)}}}, \"| mean similarity to the seed in the lower half\", "
                f"round(float({sims[0]}[lower].mean()), 2))")
    if not clip:
        code.append("print(\"lower half relative to the centre:\", relate(seed, lower, v))")
    if multi:
        code.append("print(\"match photo 1->2:\", {k: match(0, 1, k) for k in latents})")
        if not clip:
            code.append("print(\"centre of photo 1 in photo 2:\", track(seed, 1, v))")
    steps["E6"] = "\n".join(code)
    return steps


# Not shown to the analyst: the boxes of the objects locate() puts at tier high or medium (log["located3"], judge sets
# e3_high / e3_medium); run in both modes, after the analyst, on a freshly loaded sandbox.
RULE3_CODE = ('print(repr([{"object": o, "view": r["view"], "box": r["box"], "tier": r["tier"]} '
              'for o, r in locate_all(objects).items() if r["tier"] in ("high", "medium")]))')


NOTES_HEADER = ("\nNotes from another model's analysis of the latent features (CLIP and VGGT) of the photo(s). "
                "They may contain errors:\n")
def latents_header(use):
    names = [NAME[k] for k in ("clip", "vggt", "dinov3") if k in use]
    return (f"\nNotes from another model's analysis of the latent features ({', '.join(names[:-1]) + ' and ' + names[-1] if len(names) > 1 else names[0]}) "
            "of the photo(s). They may contain errors:\n")
NOTES_HEADER_CAMERA = ("\nNotes with the relative camera poses that VGGT's camera head predicts for the photo(s). "
                       "They may contain errors:\n")
NOTES_HEADER_PIXELS = "\nNotes from another model's analysis of the pixels of the photo(s). They may contain errors:\n"
REPORT_HEADER = {"question": "\nNotes from another model's analysis of the question, made without seeing the photo(s). "
                             "They may contain errors:\n",
                 "photos": "\nNotes from another model's analysis of the photo(s). They may contain errors:\n"}
CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.S)
# Qwen3.8 often writes code as its native tool call instead of a ```python block (3 of 10 Agent-Directed smoke-test questions)
TOOL_CALL_RE = re.compile(r"<function=python>\s*(?:<parameter=\w+>\s*)?(.*?)\s*(?:</parameter>\s*)?</function>", re.S)


def shrink(img, side=768):
    """Analyst and critic see photos at most `side` px; the final answer uses the originals, as the baseline does."""
    img = img.copy()
    img.thumbnail((side, side))
    return img


class StopAfterCode:
    """Stops generation once the reply holds a complete code block that no FINDINGS: precedes (--stop-after-code)."""

    def __init__(self, tokenizer, start):
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=True)
        code = CODE_RE.search(text) or TOOL_CALL_RE.search(text)
        done = bool(code) and "FINDINGS:" not in text[:code.start()]
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


class Chat:
    """Multi-turn chat with Qwen; images live inside the message content."""

    def __init__(self, processor, model, image_kwargs=None):
        self.processor, self.model = processor, model
        self.image_kwargs = image_kwargs or {}

    def __call__(self, messages, max_new_tokens, think=False, prefix="", stop_after_code=False):
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                    enable_thinking=think) + prefix
        images = [c["image"] for m in messages if isinstance(m["content"], list) for c in m["content"] if c["type"] == "image"]
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt", **self.image_kwargs).to(self.model.device)
        stop = {}
        if stop_after_code:
            from transformers import StoppingCriteriaList
            stop = {"stopping_criteria": StoppingCriteriaList([StopAfterCode(self.processor.tokenizer, inputs["input_ids"].shape[1])])}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **stop)
        text = self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if think and "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip(), int(inputs["input_ids"].shape[1])


class BatchChat:
    """--batch K: the local model behind the interface of an API chat (__call__ as Chat, answer() as api_chat's classes),
    for K questions that run in threads (run_batched). A call waits; once every thread is waiting or has no question left,
    serve() generates the waiting calls with the shortest reply limit, ordered by question and left-padded in one batch.
    Short calls (object names, final answers) so pass first, and the analyst's long replies are written only when every
    thread is at one: K replies at once. The batches of a run depend only on its questions and K, not on timing."""

    def __init__(self, processor, model, workers):
        self.processor, self.model = processor, model
        self.cond, self.local = threading.Condition(), threading.local()  # local.n: the question a thread is on
        self.waiting, self.running = [], workers  # calls without a reply yet; threads neither waiting nor done
        self.batches = self.splits = 0  # batches generated; batches that failed and were split in two

    def __call__(self, messages, max_new_tokens, think=False, prefix=""):
        call = {"n": getattr(self.local, "n", 0), "messages": messages, "prefix": prefix, "tokens": max_new_tokens, "think": think}
        with self.cond:
            self.waiting.append(call)
            self.running -= 1
            self.cond.notify_all()
            while "reply" not in call and "error" not in call:
                self.cond.wait()
        if "error" in call:
            raise call["error"]
        return call["reply"], call["prompt_tokens"]

    def answer(self, content):
        """(reply, prompt tokens, extra fields of the result row) of a final answer, asked as final_answers() asks a
        local model: the reply pre-filled with "Answer:", thinking off, 16 new tokens."""
        reply, ntok = self([{"role": "user", "content": content}], 16, prefix=ANSWER_PREFIX)
        return reply, ntok, {}

    def done(self):
        """A thread has no question left."""
        with self.cond:
            self.running -= 1
            self.cond.notify_all()

    def serve(self):
        """Generate, one batch per step, until every thread is done (the main thread runs this)."""
        while True:
            with self.cond:
                while self.running:
                    self.cond.wait()
                if not self.waiting:
                    return
                tokens, think = min((c["tokens"], c["think"]) for c in self.waiting)
                calls = sorted((c for c in self.waiting if (c["tokens"], c["think"]) == (tokens, think)), key=lambda c: c["n"])
                self.waiting = [c for c in self.waiting if (c["tokens"], c["think"]) != (tokens, think)]
            self.generate(calls, tokens, think)
            with self.cond:
                self.running += len(calls)
                self.cond.notify_all()

    def generate(self, calls, max_new_tokens, think):
        """One batch. A batch that fails, above all for lack of GPU memory, is split in two and tried again, so only a
        call that fails on its own gets the error, as it would without --batch."""
        error = self._generate(calls, max_new_tokens, think)
        if error is None:
            return
        if isinstance(error, torch.OutOfMemoryError):
            gc.collect()
            torch.cuda.empty_cache()
        if len(calls) == 1:
            calls[0]["error"] = error
            return
        self.splits += 1
        self.generate(calls[:len(calls) // 2], max_new_tokens, think)
        self.generate(calls[len(calls) // 2:], max_new_tokens, think)

    def _generate(self, calls, max_new_tokens, think):
        """Fill each call's reply and prompt tokens, as Chat does for one call; returns the error if the batch failed."""
        self.batches += 1
        try:
            prompts = [self.processor.apply_chat_template(c["messages"], add_generation_prompt=True, tokenize=False,
                                                          enable_thinking=think) + c["prefix"] for c in calls]
            images = [x["image"] for c in calls for m in c["messages"] if isinstance(m["content"], list)
                      for x in m["content"] if x["type"] == "image"]  # in the order of the prompts, as the processor reads them
            inputs = self.processor(text=prompts, images=images or None, padding=True, padding_side="left",
                                    return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        except Exception as e:  # noqa: BLE001
            return e.with_traceback(None)  # without the traceback, whose frames would keep the batch on the GPU
        start = inputs["input_ids"].shape[1]  # left-padded: every reply starts here
        for c, row, ntok in zip(calls, out, inputs["attention_mask"].sum(1).tolist()):
            text = self.processor.decode(row[start:], skip_special_tokens=True)
            if think and "</think>" in text:
                text = text.split("</think>", 1)[1]
            c["reply"], c["prompt_tokens"] = text.strip(), int(ntok)
        return None


def run_batched(items, question, new_sandbox, chat, workers):
    """--batch K: K threads, each with a sandbox of its own, take the questions in order and run question(n, item,
    sandbox); their model calls meet in chat.serve() (BatchChat). An error outside a question's own handling stops the
    run once the threads are done, as it stops a run without --batch."""
    todo, lock, errors = iter(enumerate(items, 1)), threading.Lock(), []

    def worker():
        sandbox = None
        try:
            sandbox = new_sandbox()
            while True:
                with lock:
                    n, it = next(todo, (None, None))
                if it is None:
                    break
                chat.local.n = n
                question(n, it, sandbox)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            if sandbox:
                sandbox.close()
            chat.done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    chat.serve()
    for t in threads:
        t.join()
    print(f"batch {workers}: {chat.batches} batches, {chat.splits} split in two (too large for the GPU memory, or an error), "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    if errors:
        raise errors[0]


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def echo_code(code):
    """Free-form steps only: the prompt says the tools print, but they return their result, so a bare call such as
    locate_all([...]) printed nothing. Every top-level bare expression whose value is not None is printed, as in a
    notebook, and `from tools import *` (the tools are already in the namespace) is dropped instead of failing."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # the sandbox reports the error itself
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and all(
                (getattr(node, "module", None) or a.name).split(".")[0] == "tools" for a in node.names):
            continue
        if isinstance(node, ast.Expr) and not (isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "print"):
            body += ast.parse(f"__v = ({ast.unparse(node.value)})\nif __v is not None:\n    print(__v)").body
            continue
        body.append(node)
    tree.body = body
    return ast.unparse(tree)


def notes_text(findings, reliable_only=False):
    """Only latent measurements reach the answering model; visual impressions stay in the transcript.
    reliable_only (agent_reliable): only findings rated high or medium, and nothing at all (not even the summary) when
    there is none, so the answer is then made exactly as in the photo-only baseline."""
    latent = [f for f in findings.get("findings", []) if str(f.get("source", "latent")).lower() != "visual"]
    if reliable_only:
        latent = [f for f in latent if str(f.get("reliability", "")).lower() in ("high", "medium")]
        if not latent:
            return ""
    lines = [f"- {f.get('text', '')} (reliability: {f.get('reliability', '?')})" for f in latent]
    # The summary can restate low-reliability claims, so the reliable-only notes leave it out
    text = "\n".join(lines) + (f"\nSummary: {findings['summary']}" if findings.get("summary") and not reliable_only else "")
    return text[:1500]


def sandbox_question(item, args):
    """What the sandbox loads for one question (the same in run_agent and for the hidden rule boxes)."""
    # The pixel grids are cut from the photos, so a --pixels sandbox keeps them (no show(), so no picture comes out).
    q = {"images": item["images"], "latents": item["latents"], "blind": not args.pixels, "pixels": args.pixels,
         "noise": item.get("noise"), "clip_text": item.get("clip_text"), "only": args.latent,
         "flip_text_map": args.flip_text_map, "basic_tools": bool(args.latent)}
    if args.latents:
        q.update(use=list(args.latents), no_show=True)  # F holds only these latents; no show(), so no picture at all
    return q


def blend_boxes(item, sandbox, objects, args, log):
    """--latents, both modes: the hidden consensus boxes (RULE3_CODE -> log["located3"]) on a freshly loaded sandbox,
    so nothing the Agent-Directed analyst defined can change them."""
    import ast

    log["located3"] = []
    if "clip" not in args.latents or args.pixels or not objects:  # the consensus boxes need CLIP's text map
        return
    sandbox.load(sandbox_question(item, args))
    out, _ = sandbox.run(f"objects = {objects!r}\n" + RULE3_CODE)
    try:
        log["located3"] = ast.literal_eval(out.strip())
    except (ValueError, SyntaxError):
        log["rule3_error"] = out[-500:]


# --marks (MV-RoboBench's boxes, SPAR's points or boxes, found by marks.py): the marks drawn into the photos,
# which the blind analyst cannot see, listed in its first message after the question and options. The system prompts are
# unchanged (the blind one says box() must never guess where an object is: a marked position is given, and the note says
# so). Without --marks, or for a question with no mark, the first message is exactly as before.
MARKS_NOTE = ("These marks are drawn in the photo(s) and are what the question refers to. Their positions are "
              "given, not guessed, so a marked object counts as located: box(x0, y0, x1, y1, model, v) with these "
              "coordinates (v = photo number - 1; for a point, a box of +-0.04 around it) gives the latent cells of the "
              "marked object, usable with the other tools.")
# The frame rule is added only when the question names a reference view (SPAR: "the first image is the
# observer's perspective") and every mark lies in another photo: there the analyst had read a mark's position in its own photo
# as a position in the observer's view. Everywhere else (MV-RoboBench never names a reference view) the note is MARKS_NOTE
# word for word, which did better there.
REF_VIEW_RE = re.compile(r"observer|primary (?:viewpoint|perspective)|first image", re.I)
MARKS_NOTE_FRAME = ("These marks are drawn in the photo(s) and are what the question refers to. Their positions are "
                    "given, not guessed, so a marked object counts as located: box(x0, y0, x1, y1, model, v) with these "
                    "coordinates (v = photo number - 1; for a point, a box of +-0.04 around it) gives the latent cells of "
                    "the marked object as a boolean mask (print mask.sum(), not the mask), usable with the other tools. A "
                    "mark's position holds only in the photo it is drawn in: when the question asks about a relation seen "
                    "from another photo (for example the first photo as the observer's view), carry the marked cells there "
                    "with track(mask, vb, va=v) or locate the object in that photo, and never read a position in one photo "
                    "as a position in another.")


def marks_text(marks, question=""):
    """The marks section of the analyst's first message, from marks.detect_marks(); "" without marks. The frame rule is
    added when the question names a reference view (photo 1) and no mark lies in it."""
    if not marks:
        return ""
    lines = [f"- Photo {m['photo']}: {m['color']} box, x {m['box'][0]:.2f}-{m['box'][2]:.2f}, y {m['box'][1]:.2f}-{m['box'][3]:.2f}"
             if m["kind"] == "box" else f"- Photo {m['photo']}: {m['color']} point at ({m['center'][0]:.2f}, {m['center'][1]:.2f})"
             for m in marks]
    frame = bool(REF_VIEW_RE.search(question or "")) and all(m["photo"] != 1 for m in marks)
    return "\nMarks in the photo(s):\n" + "\n".join(lines) + "\n" + (MARKS_NOTE_FRAME if frame else MARKS_NOTE)


def analyst_question(item, n_views, grid_line, blind, marks=None):
    """The text of the analyst's first message: photo count, grids, question and options, then (--marks) the marks.
    grid_line None (--report-from): there are no grids."""
    return (f"{n_views} photo(s){' (not shown to you)' if blind else ''}." + (f" Grids: {grid_line}." if grid_line else "")
            + f"\nQuestion: {item['question']}\nOptions: " + "; ".join(f"{k}. {v}" for k, v in item["options"]) + marks_text(marks, item["question"]))


def run_agent(item, sandbox, chat, args, log):
    """The analysis of one question, in a single pass. Agent-Directed: the analyst writes code against the latent features
    in the sandbox (at most --max-steps code steps) and ends with FINDINGS. --all-tools (Fixed-Sequence): it reads the
    output of the fixed steps instead. --report-from: no features and no sandbox. Returns the findings dict."""
    if args.report_from:  # no latents and no sandbox: the report is written in one reply
        return report_findings(item, chat, args, log)
    # The analyst never sees the photo(s); only the model that answers gets them, as the photo-only baseline does.
    n_views = len(item["images"])
    lat = {k: np.load(item["latents"][k], mmap_mode="r") for k in ("clip", "vggt")}  # DINOv3 shares the VGGT grid
    grids = {k: [tuple(x[v % len(x)].shape[:2]) for v in range(n_views)] for k, x in lat.items()}  # as the sandbox aligns views
    sandbox.load(sandbox_question(item, args))
    if args.latents and args.pixels:
        grid_line = f"clip {grids['clip'][0]} per photo; vggt {grids['vggt']}"
    elif args.latents:
        geo = [NAME[k] for k in ("vggt", "dinov3") if k in args.latents]
        grid_line = (f"CLIP {grids['clip'][0]} per photo; " if "clip" in args.latents else "") + f"{' and '.join(geo)} {grids['vggt']}"
    else:
        grid_line = f"CLIP {grids['clip'][0]} per photo; VGGT{' and DINOv3' if args.latent == 'dinov3' else ''} {grids['vggt']}"
    question = analyst_question(item, n_views, grid_line, True, item.get("marks"))
    if args.latents and args.pixels:
        question = question.replace("the latent cells of the marked object", "the grid cells of the marked object")
    first = [{"type": "text", "text": question}]
    if args.latents and args.pixels:
        analyst_prompt = blend_prompt_pixels()
    elif args.latents:
        # One prompt for both modes; it never mentions show(), which is not in the namespace.
        analyst_prompt = blend_prompt(args.latents, fixed=args.all_tools)
    else:
        analyst_prompt = one_encoder_prompt(args.latent)
    history = [{"role": "system", "content": analyst_prompt}, {"role": "user", "content": first}]
    if args.all_tools:
        return all_tools_findings(item, sandbox, chat, args, log, history)
    executions, findings, steps, asks = {}, None, 0, 0
    start = time.time()
    while True:
        # After max_steps code steps, or once the time budget is used, ask (at most twice) for findings only.
        if steps >= args.max_steps or time.time() - start > args.time_budget:
            if asks == 2:
                break
            asks += 1
            history.append({"role": "user", "content": [{"type": "text", "text":
                            "Step or time limit reached. Do not write code. Reply only with FINDINGS: and the JSON, "
                            "using the evidence you already have (say what is still uncertain)."}]})
        kw = {"stop_after_code": True} if args.stop_after_code and not asks else {}
        reply, ntok = chat(history, args.analyst_tokens, **kw)
        history.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        log["calls"].append({"role": "analyst", "round": 1, "prompt_tokens": ntok, "reply": reply})
        if "FINDINGS:" in reply and (parsed := parse_json(reply.split("FINDINGS:", 1)[1])):
            findings = parsed
            break
        code = CODE_RE.search(reply) or TOOL_CALL_RE.search(reply)
        if not code and isinstance(parsed := parse_json(reply), dict) and "findings" in parsed:
            findings = parsed  # the findings JSON without its FINDINGS: prefix
            break
        if asks:
            continue  # code after the limit is not run
        steps += 1
        if not code:
            history.append({"role": "user", "content": [{"type": "text", "text": "Reply with one ```python block, or FINDINGS: followed by valid JSON."}]})
            continue
        eid = f"E{len(executions) + 1}"
        out, shows = sandbox.run(echo_code(code.group(1)))
        executions[eid] = {"code": code.group(1), "output": out, "shows": [t for t, _ in shows], "round": 1}
        obs = [{"type": "text", "text": f"[{eid}] output:\n{out or '(no output)'}"}]
        for title, png in shows:  # only --latent runs draw maps (show()), on the latent's own grid
            obs += [{"type": "text", "text": f"\n[{eid}] shown: {title}"},
                    {"type": "image", "image": Image.open(io.BytesIO(png)).convert("RGB")}]
        history.append({"role": "user", "content": obs})
    if findings is None:
        findings = {"findings": [], "summary": "The analyst produced no valid findings."}
    verdict = ({"score": 0, "tasks": [], "why": "single pass: no critic"} if findings.get("findings") else
               {"score": 0, "tasks": ["Write FINDINGS that cite the evidence ids of your executions."],
                "why": "No findings were produced this round."})
    log["rounds"].append({"round": 1, "findings": findings, "critic": verdict})
    if time.time() - start > args.time_budget:
        log["stopped"] = f"time budget of {args.time_budget}s used after round 1"
    log["executions"] = executions
    log["used_round"] = 1
    return findings


def report_findings(item, chat, args, log):
    """--report-from: the analyst writes FINDINGS in one reply, from the question and options (question) or from them and
    the photo(s), shrunk as for --see-photos (photos). No sandbox and no code; asked once more if the reply is not valid
    FINDINGS, as in --replay. No critic."""
    sees = args.report_from == "photos"
    photos = [shrink(Image.open(p).convert("RGB")) for p in item["images"]] if sees else []
    first = [{"type": "image", "image": im} for im in photos] + [{"type": "text", "text":
             analyst_question(item, len(item["images"]), None, not sees)}]
    history = [{"role": "system", "content": report_prompt(args.report_from)}, {"role": "user", "content": first}]
    findings = None
    for _ in range(2):
        reply, ntok = chat(history, args.analyst_tokens)
        log["calls"].append({"role": "analyst", "round": 1, "prompt_tokens": ntok, "reply": reply})
        if "FINDINGS:" in reply and (findings := parse_json(reply.split("FINDINGS:", 1)[1])):
            break
        history = history + [{"role": "assistant", "content": [{"type": "text", "text": reply}]},
                             {"role": "user", "content": [{"type": "text", "text": "Reply only with FINDINGS: followed by valid JSON."}]}]
    findings = findings or {"findings": [], "summary": "The analyst produced no valid findings."}
    log["rounds"].append({"round": 1, "findings": findings, "critic": {"score": 0, "why": "report only: no critic"}})
    log["executions"], log["used_round"] = {}, 1
    return findings


def all_tools_findings(item, sandbox, chat, args, log, history):
    """--all-tools (Fixed-Sequence): the same sandbox, analyst prompt and output as the Agent-Directed run, but every tool
    runs once with fixed arguments (blend_steps for --latents, basic_steps for --latent; objects named by one text-only
    call) instead of the analyst choosing tools and writing code. The analyst then writes FINDINGS once and decides the
    boxes itself. The rule box per object (RULE_CODE, not shown to the analyst) is logged in log["located"]."""
    import ast

    import object_names

    if args.latent:  # the fixed twin of the Agent-Directed one-latent run: no object list (the Agent-Directed run has none)
        objects, head = [], ""
    else:
        objects, reply = object_names.list_objects(chat, item, args.max_objects, clean=True)
        log["calls"].append({"role": "objects", "round": 1, "reply": reply})
        head = f"objects = {objects!r}\n"
    log["objects"] = objects
    executions, shown = {}, []
    if args.latents:
        code_by_id = blend_steps(args.latents, len(item["images"]))  # fixed ids; steps that do not apply are left out
    else:
        code_by_id = basic_steps(len(item["images"]), args.latent)
    for eid, code in code_by_id.items():
        out, shows = sandbox.run(head + code)
        executions[eid] = {"code": head + code, "output": out, "round": 1}
        if shows:  # only --latent runs draw maps; the analyst sees them as in the Agent-Directed run
            executions[eid]["shows"] = [t for t, _ in shows]
            shown += [(eid, t, Image.open(io.BytesIO(png)).convert("RGB")) for t, png in shows]
    log["located"] = []
    if args.latents and "clip" in args.latents:  # the rule box needs CLIP's text_map
        out, _ = sandbox.run(head + RULE_CODE)
        try:
            log["located"] = ast.literal_eval(out.strip()) if objects else []
        except (ValueError, SyntaxError):
            log["rule_error"] = out[-500:]
    evidence = "\n\n".join(f"[{e}] ran:\n{x['code']}\n[{e}] output:\n{x['output'] or '(no output)'}" for e, x in executions.items())
    log["evidence_chars"], log["evidence_cut"] = len(evidence), len(evidence) > args.index_chars
    ask = BLEND_ASK if args.latents else BASIC_ASK
    maps = [c for e, t, img in shown for c in ({"type": "text", "text": f"\n[{e}] shown: {t}"}, {"type": "image", "image": img})]
    history = history + [{"role": "user", "content": [{"type": "text", "text": ask + evidence[:args.index_chars]}] + maps}]
    findings = None
    for _ in range(2):
        reply, ntok = chat(history, args.analyst_tokens)
        log["calls"].append({"role": "analyst", "round": 1, "prompt_tokens": ntok, "reply": reply})
        if "FINDINGS:" in reply and (findings := parse_json(reply.split("FINDINGS:", 1)[1])):
            break
        if args.latent and isinstance(parsed := parse_json(reply), dict) and "findings" in parsed:
            findings = parsed  # as its Agent-Directed twin accepts it: the findings JSON without its FINDINGS: prefix
            break
        history = history + [{"role": "assistant", "content": [{"type": "text", "text": reply}]},
                             {"role": "user", "content": [{"type": "text", "text": "Reply only with FINDINGS: followed by valid JSON."}]}]
    findings = findings or {"findings": [], "summary": "The analyst produced no valid findings."}
    log["rounds"].append({"round": 1, "findings": findings, "critic": {"score": 0, "why": "all tools: no critic"}})
    log["executions"], log["used_round"] = executions, 1
    if args.latents:
        blend_boxes(item, sandbox, objects, args, log)
    return findings


def valid_regions(regions, n_views):
    """The analyst's regions that are well-formed boxes in photo fractions."""
    out = []
    for r in regions if isinstance(regions, list) else []:
        try:
            v, (x0, y0, x1, y1) = int(r.get("view", 0)), [float(b) for b in r["box"]]
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if 0 <= v < n_views and 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1:
            out.append({"label": str(r.get("label", ""))[:30], "view": v, "box": [x0, y0, x1, y1]})
    return out


def draw_boxes(img, regions):
    """The photo with each region's box and label drawn on it."""
    from PIL import ImageDraw, ImageFont

    img = img.copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(14, W // 40))
    except TypeError:
        font = ImageFont.load_default()
    for i, r in enumerate(regions):
        color = ("#ff2d2d", "#ffd400", "#00e5ff", "#7cff00", "#ff00e5")[i % 5]
        x0, y0, x1, y1 = r["box"]
        draw.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline=color, width=max(3, W // 200))
        draw.text((x0 * W + 4, y0 * H + 2), r["label"], fill=color, font=font, stroke_width=2, stroke_fill="black")
    return img


ANSWER_RUN_RE = re.compile(r"agent_(nophoto_)?(only|reliable|(?:clip|vggt|dinov3)_(?:pad|image|imagez))")
# How the final answer gets the raw latent, as in the direct-input runs: padded tokens, or a min-max / channel-normalized image.
ANSWER_FORMATS = {"pad": "padglobal", "image": "image", "imagez": "imagez"}


def answer_runs(args, only=None):
    """The final-answer runs of a question: (name, latent injected or None, with notes). agent_only is LatentParse: the
    photo(s), the report and the question."""
    runs = [("agent_only", None, True)] + [(f"agent_{k}_{f}", k, True) for k in args.cond3_latents for f in args.answer_formats]
    if args.pixels:
        runs = [("agent_only", None, True)]  # the Pixels source control: no latent anywhere
    # The analyst's boxes drawn on the photo(s), with and without its findings.
    runs += [("agent_boxes", None, True), ("agent_boxes_nonotes", None, False)]
    if args.all_tools and not args.latent:
        # The rule boxes (strongest text_map blob per object), with and without the same findings.
        runs += [("agent_ruleboxes", None, True), ("agent_ruleboxes_nonotes", None, False)]
    if args.latents or args.report_from:
        # Only the findings rated high or medium, and the consensus boxes (locate tier high or medium; both modes) with
        # and without the findings.
        runs.insert(1, ("agent_reliable", None, True))
        if args.latents and "clip" in args.latents and not args.pixels:
            runs += [("agent_consboxes", None, True), ("agent_consboxes_nonotes", None, False)]
    if only:
        runs = [(name, None if m[2] in ("only", "reliable") else m[2].split("_")[0], True) for name in only
                for m in [ANSWER_RUN_RE.fullmatch(name)]]
    return runs


def final_answers(item, notes, chat_model, processor, injector, args, only=None, regions=None, rule_regions=None,
                  findings=None, cons_regions=None):
    """The answering VLM: the photo(s), the report and the question, in the prompt of the photo-only baseline.
    `only`: run just these names (see ANSWER_RUN_RE), each with the findings; nophoto runs leave the photo(s) out.
    agent_reliable: only the findings rated high or medium (`findings`), and no notes at all when there is none, so the
    prompt is then exactly the photo-only baseline's. agent_consboxes: the hidden consensus boxes (`cons_regions`)."""
    rows = {}
    reliable = notes_text(findings or {}, reliable_only=True)
    runs = answer_runs(args, only)
    for name, kind, with_notes in runs:
        run_notes = notes
        if name.endswith("_reliable"):
            run_notes, with_notes = reliable, bool(reliable)
        latent = load_latent(item["latents"][kind], item.get("noise") and f"{item['noise']}/{kind}") if kind else None
        fmt = ANSWER_FORMATS[name.rsplit("_", 1)[1]] if kind else "padglobal"
        stats = latent_stats(args.latents_root, kind) if fmt == "imagez" else None
        content, vectors, _ = build_content(item, latent, kind, fmt, injector, stats, photos="_nophoto_" not in name)
        if with_notes:
            header = (NOTES_HEADER_CAMERA if getattr(args, "notes_header", None) == "camera_head"
                      else REPORT_HEADER[args.report_from] if args.report_from else NOTES_HEADER_PIXELS if args.pixels
                      else latents_header(args.latents) if args.latents else NOTES_HEADER)
            content.insert(len(content) - 1, {"type": "text", "text": header + run_notes})
        if name.startswith(("agent_boxes", "agent_ruleboxes", "agent_consboxes")):
            rule = rule_regions if name.startswith("agent_ruleboxes") else cons_regions
            src = regions if name.startswith("agent_boxes") else [{"label": r.get("object", ""), "view": r.get("view", 0),
                                                                    "box": r.get("box")} for r in rule or []]
            boxes = valid_regions(src, len(item["images"]))
            for v, c in enumerate(content[:len(item["images"])]):
                c["image"] = draw_boxes(c["image"], [r for r in boxes if r["view"] == v])
            if boxes:  # with no box drawn the prompt stays that of the same run without boxes
                content.insert(len(content) - 1, {"type": "text", "text": BOXES_HEADER})
        extra = {}
        if processor is None:  # a closed VLM: the same content, sent as its direct-input runner sends it (api_chat.py)
            t0 = time.time()
            reply, ntok, extra = chat_model.answer(content)
        else:
            prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=False) + ANSWER_PREFIX
            images = [c["image"] for c in content if c["type"] == "image"]
            # Gemma's image budget only when there are images, as answer_qwen.py passes it
            kw = {"images_kwargs": {"max_soft_tokens": args.max_soft_tokens}} if images and getattr(args, "max_soft_tokens", None) else {}
            inputs = processor(text=[prompt], images=images or None, return_tensors="pt", **kw).to(chat_model.device)
            injector.vectors = vectors
            t0 = time.time()
            with torch.inference_mode():
                gen = chat_model.generate(**inputs, max_new_tokens=16, do_sample=False)
            injector.vectors = None
            reply = processor.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            ntok = int(inputs["input_ids"].shape[1])
        pred = parse_letter(reply, [k for k, _ in item["options"]])
        rows[name] = {"bench": item["bench"], "id": item["id"], "type": item["type"], "options": item["options"],
                      "latent_from": item["id"] if kind else None, "answer": item["answer"], "reply": reply,
                      "prediction": pred, "correct": pred == item["answer"], "seconds": round(time.time() - t0, 2),
                      "prompt_tokens": ntok, **extra}
    return rows


def pilot_subset(items, n, seed=0):
    """n questions per benchmark, spread over question types (round-robin over shuffled types).
    MindCube draws with a generator of its own, so its pilot does not depend on what else the sample dir holds."""
    rng = random.Random(seed)
    picked = []
    for bench in ("3dsr", "mmsi", "mindcube"):
        r = random.Random(seed) if bench == "mindcube" else rng
        by_type = collections.defaultdict(list)
        for it in items:
            if it["bench"] == bench:
                by_type[it["type"]].append(it)
        queues = [r.sample(v, len(v)) for _, v in sorted(by_type.items())]
        r.shuffle(queues)
        chosen = []
        while len(chosen) < n and any(queues):
            for q in queues:
                if q and len(chosen) < n:
                    chosen.append(q.pop())
        picked += chosen
    return picked


def main():
    ap = argparse.ArgumentParser(description="LatentParse: the analyst parses the latent features in a sandbox and writes a "
                                             "report; the answering VLM answers from the photo(s) and the report.")
    ap.add_argument("--sample", required=True, help="sample dir (data/build_*.py)")
    ap.add_argument("--latents-root", required=True, help="dir holding <encoder>/<bench>/<id>.npy (extract_latents.py)")
    ap.add_argument("--model", required=True, help="the VLM: a local model dir, or gpt-6-luna / claude-haiku-5-5 over the API")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", choices=["mmsi", "mvrb", "spar", "3dsr"], help="only this benchmark's questions (after the "
                    "Mismatched donors are picked, so they match a run over the whole sample)")
    ap.add_argument("--ids", nargs="+", help="only these question ids")
    ap.add_argument("--pilot", type=int, help="questions per benchmark, spread over the question types")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th question starting at i (parallel jobs, same --out)")
    # What the analyst reads: exactly one of --latents, --latent and --report-from.
    ap.add_argument("--latents", nargs="+", choices=["clip", "vggt", "dinov3"],
                    help="the encoders whose latent features the analyst reads; clip vggt dinov3 is LatentParse with the "
                         "Multi-Latent features (DINOv3 on the VGGT grid; VGGT or DINOv3 must be among them)")
    ap.add_argument("--latent", choices=["clip", "vggt", "dinov3"],
                    help="one encoder alone, with the basic tools (the + CLIP, + DINOv3 and + VGGT rows)")
    ap.add_argument("--report-from", choices=["question", "photos"],
                    help="source control: no features and no sandbox; the report is written in one reply from the question "
                         "alone (question) or from the question and the photo(s) (photos)")
    ap.add_argument("--all-tools", "--fixed-sequence", dest="all_tools", action="store_true",
                    help="Fixed-Sequence mode: every tool runs once with fixed arguments and the analyst writes its report "
                         "from the output (default: Agent-Directed, the analyst picks the tools and writes the code)")
    ap.add_argument("--wrong-latent", "--mismatched", dest="wrong_latent", action="store_true",
                    help="Mismatched control: every question reads the latent features of another question")
    ap.add_argument("--noise-latent", "--random-noise", dest="noise_latent", action="store_true",
                    help="Random Noise control: Gaussian noise with each feature map's per-channel mean and std")
    ap.add_argument("--pixels", action="store_true",
                    help="Pixels source control, with --latents clip vggt: the photos' pixels on the CLIP and VGGT grids "
                         "in place of the latent features")
    ap.add_argument("--marks", action="store_true",
                    help="MV-RoboBench and SPAR-Bench-Tiny: the boxes or points drawn into the photos, which the analyst "
                         "cannot see, are found by color (marks.py) and listed after the question")
    ap.add_argument("--layer", help="encoder layers, with --latents: ENC:SPEC, e.g. vggt:5; --latents-root/ENC holds block "
                    "SPEC (1-24) of that encoder, or with cat its blocks 5, 12, 18 and 24 (extract_latents.py --layer)")
    ap.add_argument("--last-latents-root", help="with --layer clip:...: the root of the last-layer latents, whose CLIP "
                    "tokens text_map keeps reading")
    ap.add_argument("--flip-text-map", action="store_true",
                    help="negate the CLIP text map (always on with --latents): last-layer CLIP patch tokens score the "
                         "named object lowest, so the negated map puts it on top")
    ap.add_argument("--max-steps", type=int, default=4, help="analyst code steps")
    ap.add_argument("--time-budget", type=float, default=600,
                    help="seconds per question for the analysis; when used up the analyst writes its findings")
    ap.add_argument("--analyst-tokens", type=int, default=1000)
    ap.add_argument("--max-objects", type=int, default=4, help="objects named per question for the boxes")
    ap.add_argument("--index-chars", type=int, default=8000, help="Fixed-Sequence: the characters of tool output shown")
    ap.add_argument("--stop-after-code", action=argparse.BooleanOptionalAction, default=None,
                    help="end each analyst reply at its first complete code block (on for InternVL unless given)")
    ap.add_argument("--cond3-latents", nargs="*", choices=["clip", "vggt", "dinov3"],
                    help="latent features also given directly to the answering VLM (agent_<encoder>_<format>); none in "
                         "--latents runs unless given")
    ap.add_argument("--answer-formats", nargs="+", choices=sorted(ANSWER_FORMATS), default=["pad"],
                    help="how --cond3-latents reach the answering VLM: pad (padded tokens), image (global min-max image) "
                         "or imagez (channel-normalized image)")
    ap.add_argument("--max-soft-tokens", type=int, help="image token budget of the answering VLM (Gemma: 560 unless given)")
    ap.add_argument("--batch", type=int, default=1, metavar="K",
                    help="K questions side by side, their model calls generated together (a local model and no latent "
                         "in the final answer); compare only runs made with the same K")
    ap.add_argument("--answers-only", nargs="+", metavar="RUN",
                    help="reuse the findings in --out's transcripts and add only these final-answer runs")
    ap.add_argument("--notes-header", choices=["latents", "camera_head"], default="latents",
                    help="camera_head: the notes are the VGGT camera-head poses of cam_head_notes.py (with --answers-only)")
    ap.add_argument("--aggregate-only", action="store_true", help="only rebuild results.jsonl from the transcripts in --out")
    for flag in ("--blind", "--boxes", "--single-pass"):
        ap.add_argument(flag, action="store_true", help="always on: the analyst never sees the photo(s), gives a box for "
                        "each object it locates, and runs a single pass; kept so the commands of the paper runs work")
    args = ap.parse_args()
    args.blind = args.boxes = args.single_pass = True
    if args.answers_only and (bad := [r for r in args.answers_only if not ANSWER_RUN_RE.fullmatch(r)]):
        ap.error(f"unknown --answers-only runs: {bad}")
    if sum(map(bool, (args.latents, args.latent, args.report_from))) != 1 and not args.aggregate_only:
        ap.error("give exactly one of --latents, --latent and --report-from")
    if args.wrong_latent and args.noise_latent:
        ap.error("--wrong-latent and --noise-latent are separate controls")
    if args.pixels and (sorted(args.latents or []) != ["clip", "vggt"] or args.all_tools or args.layer
                        or args.wrong_latent or args.noise_latent):
        ap.error("--pixels (the Pixels source control) needs --latents clip vggt, without --all-tools, --layer, "
                 "--wrong-latent or --noise-latent")
    if args.report_from and (args.pixels or args.marks or args.all_tools or args.wrong_latent or args.noise_latent):
        ap.error("--report-from has no features and no sandbox: no --pixels, --marks, --all-tools, --wrong-latent or "
                 "--noise-latent")
    if args.notes_header == "camera_head" and not args.answers_only:
        ap.error("--notes-header camera_head goes with --answers-only (the notes come from cam_head_notes.py)")
    if args.latent:
        args.cond3_latents = [args.latent]
    if args.layer:
        enc, _, spec = args.layer.partition(":")
        if not args.latents or enc not in args.latents or not (spec == "cat" or spec.isdigit() and 1 <= int(spec) <= 24):
            ap.error("--layer is ENC:SPEC with ENC among --latents and SPEC a block 1-24 or cat")
        if enc == "clip" and not args.last_latents_root:
            ap.error("--layer clip:... needs --last-latents-root: text_map reads CLIP's last layer")
        LAYER[enc] = spec
    if args.latents:
        args.latents = [k for k in ("clip", "vggt", "dinov3") if k in args.latents]
        if not {"vggt", "dinov3"} & set(args.latents):
            ap.error("--latents needs vggt or dinov3 (motion, track and the cross-checks run on them)")
        args.flip_text_map = True  # the CLIP map is inverted on these latents; --latents runs always use the negated map
    if args.cond3_latents is None:
        args.cond3_latents = [] if args.latents or args.report_from else ["clip", "vggt"]
    if args.report_from and args.cond3_latents:
        ap.error("--report-from runs have no latent to put in the final answer (--cond3-latents)")
    if args.latents and (bad := [k for k in args.cond3_latents if k not in args.latents]):
        ap.error(f"--cond3-latents {bad} are not among --latents")
    api = api_chat.api_class(args.model)
    if api and args.cond3_latents and (bad := [f for f in args.answer_formats if f not in api_chat.IMAGE_FORMATS]):
        ap.error(f"an API model takes latents as images only, not --answer-formats {bad}")
    if api and args.answers_only and (bad := [r for r in args.answers_only if r.endswith("_pad")]):
        ap.error(f"an API model takes latents as images only, not --answers-only {bad}")
    if args.max_soft_tokens is None and "gemma" in args.model.lower():
        args.max_soft_tokens = 560
    if args.stop_after_code is None:
        args.stop_after_code = "internvl" in args.model.lower()
    if api and args.stop_after_code:
        ap.error("--stop-after-code needs a local model")
    if args.batch < 1 or args.batch > 1 and (api or args.answers_only or args.stop_after_code or args.cond3_latents
                                             or args.max_soft_tokens):
        ap.error("--batch K needs K >= 1 and, above 1, a local model and no latent in the final answer (--cond3-latents "
                 "empty), without --answers-only, --stop-after-code or an image token budget (--max-soft-tokens, Gemma)")
    # A question's clock also runs while the other K - 1 are generated, so its time budget is K times as long: the
    # analyst then gets its code steps as without --batch instead of being cut short by the wait.
    args.time_budget *= args.batch

    out = Path(args.out)
    if args.aggregate_only:
        aggregate(out)
        return
    items = load_sample(args.sample)
    donors = wrong_latent_donors(items) if args.wrong_latent else {}  # before any filtering, so shards agree
    if args.pilot:
        items = pilot_subset(items, args.pilot)
    if args.ids:
        items = [it for it in items if it["id"] in args.ids]
    if args.bench:
        items = [it for it in items if it["bench"] == args.bench]
    for it in items:
        it["latent_from"] = donors.get((it["bench"], it["id"]), it["id"])
        it["latents"] = {k: str(Path(args.latents_root) / k / it["bench"] / f"{it['latent_from']}.npy")
                         for k in ("clip", "vggt") + (("dinov3",) if args.latent == "dinov3" or args.latents and "dinov3" in args.latents else ())}
        if args.noise_latent:
            it["noise"] = f"{it['bench']}/{it['id']}"
        if "clip" in LAYER:
            it["clip_text"] = str(Path(args.last_latents_root) / "clip" / it["bench"] / f"{it['latent_from']}.npy")
    shard, n_shards = map(int, args.shard.split("/"))
    items = items[shard::n_shards]

    if api:  # no local weights: the API chat is analyst and answerer (final_answers calls its answer())
        processor = injector = None
        model = chat = api(args.model)
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=os.environ.get("DEVICE_MAP", "cuda"),
                                                            attn_implementation="sdpa").eval()
        tok = processor.tokenizer
        placeholder = next(p for p in PLACEHOLDER_CANDIDATES if len(tok.encode(p, add_special_tokens=False)) == 1)
        injector = LatentInjector(model.get_input_embeddings(), placeholder, tok.convert_tokens_to_ids(placeholder))
        chat = Chat(processor, model)
        if args.batch > 1:  # the same model behind an API chat's interface: final_answers() asks it through answer()
            model = chat = BatchChat(processor, model, args.batch)
            processor = None
    if args.answers_only:
        add_answers(items, out, model, processor, injector, args)
        aggregate(out)
        return
    if args.marks:  # imported once, so a missing dependency stops the run here and not question by question
        from marks import detect_marks
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))

    def question(n, it, sandbox):
        """One question, from the analysis to its transcript; a finished one is skipped."""
        path = out / "transcripts" / f"{it['bench']}_{it['id']}.json"
        if path.exists():
            return
        t0 = time.time()
        log = {"bench": it["bench"], "id": it["id"], "type": it["type"], "answer": it["answer"], "latent_from": it["latent_from"],
               "calls": [], "rounds": []}
        try:
            if args.marks:  # the boxes or points drawn into the photos, listed for the analyst after the question
                kind = "point" if re.search(r"\((red|green|blue) point\)", it["question"]) else "box"  # SPAR names one kind
                try:  # a photo that cannot be read costs the marks, not the analysis
                    it["marks"] = log["marks"] = detect_marks(it["images"], it["bench"], kind)
                except Exception as e:
                    it["marks"], log["marks"], log["marks_error"] = [], [], repr(e)[:300]
            findings = run_agent(it, sandbox, chat, args, log)
            if args.latents and not args.all_tools and not args.pixels:  # Agent-Directed: the same hidden consensus boxes
                import object_names

                objects, reply = object_names.list_objects(chat, it, args.max_objects, clean=True)
                log["calls"].append({"role": "objects", "round": 1, "reply": reply})
                log["objects"] = objects
                blend_boxes(it, sandbox, objects, args, log)
            notes = notes_text(findings)
        except torch.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            log["error"] = f"out of memory: {e}"[:300]
            findings = {"findings": [], "summary": "The analysis failed (out of GPU memory)."}
            notes = notes_text(findings)
        except Exception as e:  # one bad question (malformed reply, missing latent) must not stop an overnight run
            import traceback

            log["error"] = traceback.format_exc()[-1500:]
            print(f"[error] {it['bench']} {it['id']}: {e!r}"[:300], flush=True)
            findings = {"findings": [], "summary": "The analysis failed."}
            notes = notes_text(findings)
        try:  # an unreadable photo or an OOM here must not kill the shard (rerunning would stop at the same question)
            final = final_answers(it, notes, model, processor, injector, args,
                                  # a report run locates nothing: no box is drawn, whatever "regions" its reply holds
                                  regions=None if args.report_from else findings.get("regions"),
                                  rule_regions=log.get("located") if args.all_tools else None,
                                  findings=findings, cons_regions=log.get("located3"))
        except Exception as e:
            import traceback

            if isinstance(e, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
            log["error"] = (log.get("error", "") + "\nfinal answer: " + traceback.format_exc())[-1500:]
            print(f"[error] {it['bench']} {it['id']} final answer: {e!r}"[:300], flush=True)
            final = {}
        log.update({"final_findings": findings, "notes": notes, "agent_seconds": round(time.time() - t0, 1),
                    "final": final})
        write_atomic(path, json.dumps(log, ensure_ascii=False, indent=1, default=str))
        print(f"[{n}/{len(items)}] {it['bench']} {it['id']} {log['agent_seconds']}s "
              f"answers={ {k: v['prediction'] for k, v in log['final'].items()} } gold={it['answer']}", flush=True)

    def new_sandbox():
        return None if args.report_from else Sandbox()  # a --report-from run has no sandbox

    if args.batch > 1:
        run_batched(items, question, new_sandbox, chat, args.batch)
    else:
        sandbox = new_sandbox()
        for n, it in enumerate(items, 1):
            question(n, it, sandbox)
        if sandbox:
            sandbox.close()
    aggregate(out)


def add_answers(items, out, model, processor, injector, args):
    """--answers-only: add the named final-answer runs to every transcript in `out`, from the findings it already has."""
    for n, it in enumerate(items, 1):
        path = out / "transcripts" / f"{it['bench']}_{it['id']}.json"
        if not path.exists():
            continue
        log = json.loads(path.read_text())
        todo = [r for r in args.answers_only if r not in log["final"]]
        if not todo:
            continue
        rows = final_answers(it, log["notes"], model, processor, injector, args, only=todo,
                             findings=log.get("final_findings"))
        log["final"].update(rows)
        write_atomic(path, json.dumps(log, ensure_ascii=False, indent=1, default=str))
        print(f"[{n}/{len(items)}] {it['bench']} {it['id']} answers={ {k: v['prediction'] for k, v in rows.items()} } "
              f"gold={it['answer']}", flush=True)


def write_atomic(path, text):
    """Write via a temporary file and rename, so a reader (aggregate in a sibling shard, a resume) never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def aggregate(out):
    """Rebuild <run>/results.jsonl, metrics.json and agent_stats.json from every transcript in `out`."""
    logs = [json.loads(p.read_text()) for p in sorted((out / "transcripts").glob("*.json"))]
    for run in sorted({r for lg in logs for r in lg["final"]}):
        rows = [lg["final"][run] for lg in logs if run in lg["final"]]
        (out / run).mkdir(exist_ok=True)
        with open(out / run / "results.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / run / "metrics.json").write_text(json.dumps(summarize(rows), indent=1))
    agent_stats = {"questions": len(logs),
                   "mean_agent_seconds": round(float(np.mean([lg["agent_seconds"] for lg in logs])), 1) if logs else None,
                   "no_findings": sum(not (lg.get("final_findings") or {}).get("findings") for lg in logs),
                   "errors": sum("error" in lg for lg in logs),
                   "missing_final": sum(not lg.get("final") for lg in logs)}  # left out of every run's metrics
    (out / "agent_stats.json").write_text(json.dumps(agent_stats, indent=1))
    print("agent:", agent_stats, flush=True)


if __name__ == "__main__":
    main()
