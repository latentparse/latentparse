#!/usr/bin/env python3
"""Photo only and direct input with an OpenAI model (GPT-6 Luna), with the same prompts as answer_qwen.py.

Usage: python answer_gpt.py --sample DIR --out RESULTS_DIR --latents-root DIR \
           --runs baseline clip:image vggt:image clip:image:wrong vggt:image:wrong \
           [--model gpt-6-luna] [--effort none] [--workers 8] [--limit N]

Only image formats (image, imagez, tilez) work over an API (padding needs the model's
embeddings). The key is read from ~/.config/openai_key and never printed.
Results use the same layout as answer_qwen.py, so compare_runs.py reads them,
plus input/output token counts per question and an estimated cost per run.
"""
import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

from answer_qwen import build_content, load_run_latent, run_stats, summarize, wrong_latent_donors
from common import load_sample, parse_letter

IMAGE_FORMATS = ("image", "imagez", "tilez")
PRICE_PER_M = {"gpt-6-luna": (0.10, 0.50)}  # USD per 1M input / output tokens, standard tier


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # lossless, so latent images keep every value
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_input(content):
    parts = []
    for c in content:
        if c["type"] == "image":
            parts.append({"type": "input_image", "image_url": to_data_url(c["image"])})
        else:
            parts.append({"type": "input_text", "text": c["text"]})
    return [{"role": "user", "content": parts}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latents-root")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="none", help="reasoning effort; none matches Qwen with thinking off")
    ap.add_argument("--max-output-tokens", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-photo", action="store_true", help="latent only; the baseline becomes question text only")
    args = ap.parse_args()

    key_file = Path("~/.config/openai_key").expanduser()
    key = os.environ.get("OPENAI_API_KEY") or (key_file.read_text().strip() if key_file.exists() else None)
    client = OpenAI(api_key=key, max_retries=6)
    items = load_sample(args.sample, args.limit)
    donors = wrong_latent_donors(items)
    price_in, price_out = PRICE_PER_M.get(args.model, (float("nan"), float("nan")))
    stats = {}

    for run in args.runs:
        kind, fmt, wrong = (run.split(":") + ["", ""])[:3]
        if kind != "baseline" and fmt not in IMAGE_FORMATS:
            raise SystemExit(f"{run}: only image formats ({', '.join(IMAGE_FORMATS)}) work over an API")
        st = run_stats(args.latents_root, kind, fmt, stats)
        name = "baseline" if kind == "baseline" else f"{kind}_{fmt}" + ("_wrong" if wrong else "")
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)

        def ask(it):
            src = donors[(it["bench"], it["id"])] if wrong else it["id"]
            latent = None if kind == "baseline" else load_run_latent(args.latents_root, kind, it["bench"], src)
            content, _, _ = build_content(it, latent, kind, fmt, None, st, photos=not args.no_photo)
            t0 = time.time()
            resp = client.responses.create(model=args.model, input=to_input(content), reasoning={"effort": args.effort},
                                           max_output_tokens=args.max_output_tokens)
            reply = (resp.output_text or "").strip()
            pred = parse_letter(reply, [k for k, _ in it["options"]])
            return {"bench": it["bench"], "id": it["id"], "type": it["type"], "options": it["options"],
                    "latent_from": None if kind == "baseline" else src, "answer": it["answer"], "reply": reply,
                    "prediction": pred, "correct": pred == it["answer"], "seconds": round(time.time() - t0, 2),
                    "prompt_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}

        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(ask, items))
        with open(out / "results.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = summarize(rows)
        tin, tout = sum(r["prompt_tokens"] for r in rows), sum(r["output_tokens"] for r in rows)
        metrics.update({"input_tokens": tin, "output_tokens": tout,
                        "estimated_cost_usd": round(tin / 1e6 * price_in + tout / 1e6 * price_out, 4)})
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        (out / "README.md").write_text(
            f"# {name}\n\nVLM: `{args.model}` via the Responses API, reasoning effort `{args.effort}`, "
            f"max {args.max_output_tokens} output tokens, images sent as PNG.\nSample: `{args.sample}`\n"
            f"Photos: {'not shown' if args.no_photo else 'shown'}\nLatent: {kind}"
            + (f", format `{fmt}`" if kind != "baseline" else "") + (" - **wrong latent** (see `latent_from`)" if wrong else "")
            + f"\n\nAccuracy: {metrics['accuracy']} on {metrics['n']} questions; "
              f"{tin} input + {tout} output tokens, about ${metrics['estimated_cost_usd']}\n")
        print(name, json.dumps({b: v["accuracy"] for b, v in metrics["by_benchmark"].items()}),
              f"tokens {tin}+{tout} ~${metrics['estimated_cost_usd']}", flush=True)


if __name__ == "__main__":
    main()
