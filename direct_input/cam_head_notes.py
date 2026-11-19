#!/usr/bin/env python3
"""Decoded-pose baseline (review W1): VGGT's own camera head gives the relative poses of a question's photos, written as
text notes in agent.py transcripts, so `agent.py --answers-only agent_only --notes-header camera_head` answers from the
photos and these notes exactly as LatentParse answers from its report.

Usage: python cam_head_notes.py --sample DIR --out RUN_DIR [--types camera] [--shard i/n]
Writes RUN_DIR/transcripts/<bench>_<id>.json with notes, final_findings and an empty final.
Conventions (OpenCV, VGGT): x right, y down, z forward; extrinsics map world (= camera of photo 1) to each camera."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import load_sample

CAMERA_TYPES = {"Motion (Cam.)", "Positional Relationship (Cam.–Cam.)", "Positional Relationship (Cam.–Obj.)",
                "Positional Relationship (Cam.–Reg.)"}


def describe(R, t, a, b):
    """Camera of photo b relative to camera of photo a; R, t: world-to-camera of b expressed in camera a's frame."""
    fwd = R.T @ np.array([0.0, 0.0, 1.0])            # b's viewing direction in a's frame
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))    # + = turned right
    pitch = math.degrees(math.atan2(-fwd[1], math.hypot(fwd[0], fwd[2])))  # + = tilted up
    c = -R.T @ t                                      # b's camera centre in a's frame
    n = np.linalg.norm(c)
    parts = []
    if n > 1e-6:
        d = c / n
        for v, pos, neg in ((d[0], "right", "left"), (-d[1], "up", "down"), (d[2], "forward", "backward")):
            if abs(v) >= 0.2:
                parts.append(f"{pos if v > 0 else neg} {abs(v):.2f}")
    turn = "no turn" if abs(yaw) < 2 else f"turns {abs(yaw):.0f} degrees {'right' if yaw > 0 else 'left'}"
    tilt = "" if abs(pitch) < 2 else f" and tilts {abs(pitch):.0f} degrees {'up' if pitch > 0 else 'down'}"
    move = ("its position moves " + ", ".join(parts) + " (unit direction)") if parts else "its position barely moves"
    return f"From photo {a} to photo {b} the camera {turn}{tilt}; {move}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", choices=["camera", "all"], default="camera")
    ap.add_argument("--bench", default="mmsi")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    i0, n = map(int, args.shard.split("/"))
    items = [it for it in load_sample(args.sample) if it["bench"] == args.bench and (args.types == "all" or it["type"] in CAMERA_TYPES)][i0::n]
    out = Path(args.out) / "transcripts"
    out.mkdir(parents=True, exist_ok=True)
    for k, it in enumerate(items, 1):
        path = out / f"{it['bench']}_{it['id']}.json"
        if path.exists():
            continue
        imgs = load_and_preprocess_images(it["images"], mode="crop").to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(imgs)
        ext, _ = pose_encoding_to_extri_intri(pred["pose_enc"].float(), imgs.shape[-2:])
        ext = ext[0].cpu().numpy()  # [S, 3, 4], camera 1 = identity
        lines = []
        S = len(it["images"])
        pairs = [(0, j) for j in range(1, S)] + [(j, j + 1) for j in range(1, S - 1)]
        for a, b in pairs:
            Ra, ta = ext[a][:, :3], ext[a][:, 3]
            Rb, tb = ext[b][:, :3], ext[b][:, 3]
            R = Rb @ Ra.T                  # camera a -> camera b
            t = tb - R @ ta
            lines.append(describe(R, t, a + 1, b + 1))
        findings = {"findings": [{"text": s, "source": "latent", "reliability": "high"} for s in lines][:6],
                    "summary": "", "regions": []}
        notes = "\n".join(f"- {s}" for s in lines) if S > 1 else "- Only one photo: no relative camera pose."
        log = {"bench": it["bench"], "id": it["id"], "type": it["type"], "answer": it["answer"],
               "latent_from": it.get("latent_from"), "notes": notes[:1500], "final_findings": findings, "final": {},
               "calls": [], "rounds": [], "executions": {}, "used_round": 1, "agent_seconds": 0}
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(f"[{k}/{len(items)}] {it['bench']} {it['id']} {notes[:160]!r}", flush=True)


if __name__ == "__main__":
    main()
