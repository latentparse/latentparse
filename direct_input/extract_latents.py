#!/usr/bin/env python3
"""Extract the latent features (raw patch tokens) of CLIP, VGGT or DINOv3 for every question of a sample.

Usage: python extract_latents.py --model {clip,vggt,dinov3} --sample DIR --out DIR [--limit N] [--ids-file F] [--layer L]

Writes OUT/<bench>/<question id>.npy with shape [views, h, w, dim] (fp32) and OUT/meta.json.
Only encoder tokens are kept: no heads, no decoding, nothing trained.

- clip: openai/clip-vit-large-patch14, last_hidden_state[:, 1:] (patch tokens, before
  the projection to 768-d). Each view is resized to 224x224 without cropping, so
  the whole photo is encoded -> 16x16x1024. Views are encoded independently.
- vggt: facebook/VGGT-1B, last aggregator layer, patch tokens only (camera and
  register tokens dropped) -> (H/14)x(W/14)x2048. All views of a question go in
  together, as VGGT is designed for. Preprocessing is VGGT's own
  load_and_preprocess_images(mode="crop"): width 518, height scaled to a multiple of 14.
- dinov3: DINOv3 ViT-L/16 (timm copy of facebook/dinov3-vitl16-pretrain-lvd1689m), patch tokens
  after the final norm (class and register tokens dropped). Each view gets VGGT's crop and is
  rescaled so its 16 px patches fall on the VGGT grid -> (H/14)x37x1024, patch for patch with VGGT.

--layer (the layer ablation of Table tab:app_layers) reads block L of the 24 instead of the last one: CLIP's
hidden state after block L, VGGT's aggregator output of block L (frame and global attention), DINOv3's output of block L
with the final norm applied (as the last block is above). --layer cat L2-normalizes blocks 5, 12, 18 and 24 per patch and
concatenates them (4x the channels): the blocks VGGT's DPT heads read. Without --layer the output is exactly as before.
--ids-file limits the questions to the ids listed in F, one per line.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tf
from PIL import Image

from common import load_sample


CAT_BLOCKS = (5, 12, 18, 24)  # --layer cat: the blocks VGGT's DPT heads read, counted from 1


def pick(blocks, layer):
    """blocks[k - 1]: block k's patch tokens. layer: "1"-"24", or "cat" for CAT_BLOCKS, each L2-normalized per patch and
    concatenated."""
    want = CAT_BLOCKS if layer == "cat" else (int(layer),)
    missing = [k for k in want if blocks[k - 1] is None]
    if missing:
        raise ValueError(f"block(s) {missing} not kept by the model (VGGT keeps only {CAT_BLOCKS})")
    if layer == "cat":
        return torch.cat([tf.normalize(blocks[k - 1], dim=-1) for k in CAT_BLOCKS], dim=-1)
    return blocks[int(layer) - 1]


def clip_extractor(device, layer=None):
    from transformers import CLIPImageProcessor, CLIPVisionModel

    name = "openai/clip-vit-large-patch14"
    proc = CLIPImageProcessor.from_pretrained(name)
    model = CLIPVisionModel.from_pretrained(name, dtype=torch.float32).to(device).eval()

    def run(paths):
        imgs = [Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC) for p in paths]
        px = proc(images=imgs, do_resize=False, do_center_crop=False, return_tensors="pt")["pixel_values"]
        if layer is None:
            tokens = model(pixel_values=px.to(device)).last_hidden_state[:, 1:, :]  # [S, 256, 1024]
        else:  # hidden_states[0] is the embedding, [k] block k's output; [24] equals last_hidden_state
            hs = model(pixel_values=px.to(device), output_hidden_states=True).hidden_states
            tokens = pick([h[:, 1:, :] for h in hs[1:]], layer)
        return tokens.reshape(len(paths), 16, 16, -1)

    meta = {"model": name, "tensor": "last_hidden_state[:, 1:] (patch tokens, pre-projection)",
            "layer": "last", "preprocess": "resize to 224x224, no crop", "grid": "16x16", "dim": 1024}
    return run, meta


def vggt_extractor(device, layer=None):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    name = "facebook/VGGT-1B"
    model = VGGT.from_pretrained(name).to(device).eval()

    def run(paths):
        imgs = load_and_preprocess_images(paths, mode="crop").to(device)  # [S, 3, H, W]
        tokens_list, ps = model.aggregator(imgs[None])
        if layer is None:
            tokens = tokens_list[-1][0, :, ps:, :]  # [S, h*w, 2048]
        else:  # one entry per block, 24 in all; VGGT keeps only blocks 5, 12, 18 and 24 (its DPT heads') and None elsewhere
            tokens = pick([None if t is None else t[0, :, ps:, :] for t in tokens_list], layer)
        h, w = imgs.shape[-2] // 14, imgs.shape[-1] // 14
        return tokens.reshape(len(paths), h, w, -1)

    meta = {"model": name, "tensor": "aggregator output, last layer, patch tokens only",
            "layer": "last (index -1)", "preprocess": "load_and_preprocess_images(mode='crop'), width 518",
            "grid": "(H/14)x(W/14), varies with aspect ratio", "dim": 2048}
    return run, meta


def dinov3_extractor(device, layer=None):
    import timm
    import torch.nn.functional as F
    from vggt.utils.load_fn import load_and_preprocess_images

    # timm's copy of facebook/dinov3-vitl16-pretrain-lvd1689m (the facebook repo is gated)
    name = "timm/vit_large_patch16_dinov3.lvd1689m"
    model = timm.create_model("hf-hub:" + name, pretrained=True, dynamic_img_size=True).to(device).eval()
    p, skip = 16, model.num_prefix_tokens  # class + 4 register tokens
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def run(paths):
        imgs = load_and_preprocess_images(paths, mode="crop").to(device)  # VGGT's crop: [S, 3, 14h, 518]
        h, w = imgs.shape[-2] // 14, imgs.shape[-1] // 14
        px = F.interpolate(imgs, size=(h * p, w * p), mode="bicubic", align_corners=False)
        if layer is None:
            tokens = model.forward_features((px - mean) / std)[:, skip:, :]  # [S, h*w, 1024]
        else:  # block outputs with the final norm, patch tokens only, as forward_features gives for the last block
            want = CAT_BLOCKS if layer == "cat" else (int(layer),)
            got = model.forward_intermediates((px - mean) / std, indices=[k - 1 for k in want], norm=True,
                                              output_fmt="NLC", intermediates_only=True)
            blocks = {k: g for k, g in zip(want, got)}
            tokens = pick([blocks.get(k) for k in range(1, 25)], layer)
        assert tokens.shape[1] == h * w, (tokens.shape, h, w, skip)  # exactly one token per VGGT cell
        return tokens.reshape(len(paths), h, w, -1)

    meta = {"model": name, "tensor": "forward_features (final norm), patch tokens only",
            "layer": "last", "preprocess": "VGGT crop (width 518), rescaled to 16 px patches on the same grid",
            "grid": "(H/14)x37, the VGGT grid", "dim": 1024}
    return run, meta


EXTRACTORS = {"clip": clip_extractor, "vggt": vggt_extractor, "dinov3": dinov3_extractor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(EXTRACTORS), required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids-file", help="only the question ids listed in this file, one per line")
    ap.add_argument("--layer", choices=[str(k) for k in range(1, 25)] + ["cat"],
                    help="block 1-24 instead of the last, or cat (blocks 5, 12, 18 and 24, L2-normalized, concatenated)")
    args = ap.parse_args()

    device = "cuda"
    run, meta = EXTRACTORS[args.model](device, args.layer)
    if args.layer:
        meta.update({"layer": args.layer, "tensor": meta["tensor"].replace("last layer", f"block {args.layer}") + (
            f", blocks {CAT_BLOCKS} L2-normalized per patch and concatenated" if args.layer == "cat" else "")})
    out = Path(args.out)
    items = load_sample(args.sample, args.limit)
    if args.ids_file:
        keep = set(Path(args.ids_file).read_text().split())
        items = [it for it in items if it["id"] in keep]
    t0, shapes = time.time(), {}
    with torch.inference_mode():
        for it in items:
            lat = run(it["images"]).float().cpu().numpy()
            dest = out / it["bench"] / f"{it['id']}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, lat)
            shapes[f"{it['bench']}/{it['id']}"] = list(lat.shape)
    meta.update({"n_questions": len(items), "seconds": round(time.time() - t0, 1), "shapes": shapes})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{args.model}: {len(items)} questions in {meta['seconds']}s, e.g. {next(iter(shapes.values()))}")


if __name__ == "__main__":
    main()
