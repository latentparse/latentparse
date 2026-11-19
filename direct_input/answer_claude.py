#!/usr/bin/env python3
"""Photo only and direct input with Claude (Haiku 5.5 in the paper) through the Anthropic Messages API.

Same prompts and image formats as answer_gpt.py (build_content), so results compare row for row.
Claude takes images only, so latents go in as image formats (image, imagez) and padded-token
formats are rejected. These models take no assistant prefill, so the prompt itself asks for the
letter only. Thinking is kept as low as each model allows, closest to the other readers' thinking-off
setting: Haiku disabled, Sonnet between_tools (no thinking before the answer), Opus effort low
(Opus 5.5 cannot turn thinking off).

The key is read from ~/.config/anthropic_key (never printed); ANTHROPIC_API_KEY also works.

Usage: python answer_claude.py --model claude-haiku-5-5 --sample DIR --latents-root DIR --out DIR \
           --runs baseline clip:imagez clip:imagez:wrong [--no-photo] [--limit 2]
"""
import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np

from answer_qwen import build_content, load_run_latent, run_stats, summarize, wrong_latent_donors
from common import load_sample, parse_letter

IMAGE_FORMATS = ("image", "imagez")
PRICE_PER_M = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}
THINKING = {
    "claude-haiku-5-5": {"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}, "max_tokens": 256},
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}, "output_config": {"effort": "high"}, "max_tokens": 256},
    "claude-opus-5-5": {"output_config": {"effort": "low"}, "max_tokens": 4096},
}
LETTER_ONLY = "\nReply with the letter of your answer only."


def image_block(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # lossless, so latent images keep every value
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode("utf-8")}}


def to_messages(content):
    parts = [image_block(c["image"]) if c["type"] == "image" else {"type": "text", "text": c["text"]} for c in content]
    parts[-1] = {"type": "text", "text": parts[-1]["text"] + LETTER_ONLY}
    return [{"role": "user", "content": parts}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latents-root")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", default="claude-haiku-5-5", choices=sorted(THINKING))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-photo", action="store_true", help="latent only; the baseline becomes question text only")
    args = ap.parse_args()

    key_file = Path("~/.config/anthropic_key").expanduser()
    key = os.environ.get("ANTHROPIC_API_KEY") or (key_file.read_text().strip() if key_file.exists() else None)
    client = anthropic.Anthropic(api_key=key, max_retries=6)
    items = load_sample(args.sample, args.limit)
    donors = wrong_latent_donors(items)
    price_in, price_out = PRICE_PER_M[args.model]
    params = THINKING[args.model]
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
            try:
                resp = client.messages.create(model=args.model, messages=to_messages(content), **params)
            except anthropic.BadRequestError as e:  # e.g. an image over the size limit: record, don't stop the run
                return {"bench": it["bench"], "id": it["id"], "type": it["type"], "options": it["options"],
                        "latent_from": None if kind == "baseline" else src, "answer": it["answer"], "reply": "",
                        "prediction": None, "correct": False, "error": str(e)[:300], "seconds": round(time.time() - t0, 2),
                        "prompt_tokens": 0, "output_tokens": 0}
            reply = "".join(b.text for b in resp.content if b.type == "text").strip()
            pred = parse_letter(reply, [k for k, _ in it["options"]])
            return {"bench": it["bench"], "id": it["id"], "type": it["type"], "options": it["options"],
                    "latent_from": None if kind == "baseline" else src, "answer": it["answer"], "reply": reply,
                    "prediction": pred, "correct": pred == it["answer"], "stop_reason": resp.stop_reason,
                    "seconds": round(time.time() - t0, 2), "prompt_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens}

        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(ask, items))
        with open(out / "results.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = summarize(rows)
        tin, tout = sum(r["prompt_tokens"] for r in rows), sum(r["output_tokens"] for r in rows)
        metrics.update({"input_tokens": tin, "output_tokens": tout, "refusals": sum(r.get("stop_reason") == "refusal" for r in rows),
                        "errors": sum("error" in r for r in rows),
                        "estimated_cost_usd": round(tin / 1e6 * price_in + tout / 1e6 * price_out, 4)})
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        (out / "README.md").write_text(
            f"# {name}\n\nVLM: `{args.model}` via the Messages API, {json.dumps(params)}, images sent as PNG, "
            f"prompt ends with a letter-only instruction (no prefill on this model).\nSample: `{args.sample}`\n"
            f"Photos: {'not shown' if args.no_photo else 'shown'}\nLatent: {kind}"
            + (f", format `{fmt}`" if kind != "baseline" else "") + (" - **wrong latent** (see `latent_from`)" if wrong else "")
            + f"\n\nAccuracy: {metrics['accuracy']} on {metrics['n']} questions; refusals {metrics['refusals']}, "
              f"errors {metrics['errors']}; {tin} input + {tout} output tokens, about ${metrics['estimated_cost_usd']}\n")
        print(name, json.dumps({b: v["accuracy"] for b, v in metrics["by_benchmark"].items()}),
              f"tokens {tin}+{tout} ~${metrics['estimated_cost_usd']}", flush=True)


if __name__ == "__main__":
    main()
