"""Shared helpers of the direct-input runs and LatentParse: load the sample, build the question text, parse answers."""
import json
import re
import sys
from pathlib import Path

# benchmark -> (top-level key in samples_<bench>_*.json, image field)
# 3dsr: 3DSRBench, used only to calibrate the tool thresholds; mmsi: MMSI-Bench (data/build_from_hf.py --full);
# mvrb: MV-RoboBench, its 861 spatial questions (data/build_mv_robobench.py); spar: the 550 multiple-choice questions of
# SPAR-Bench-Tiny (data/build_spar_tiny.py). MV-RoboBench and SPAR-Bench-Tiny each have a sample dir of their own.
BENCHES = {"3dsr": ("3dsrbench", "img"), "mmsi": ("mmsi", "imgs"), "mvrb": ("mvrb", "imgs"), "spar": ("spar", "imgs")}


def load_sample(sample_dir, limit=None):
    """Return every question in the sample as a flat dict, images as absolute paths.

    A benchmark with no samples_<bench>_*.json in the dir is skipped with a one-line note on stderr; a dir with none at
    all is an error. Skipping changes the items, and so the Wrong donors (answer_qwen.wrong_latent_donors): the MMSI
    donors of every run so far need the 3DSR file in the same dir."""
    sample_dir = Path(sample_dir)
    items, found, skipped = [], 0, []
    for bench, (key, field) in BENCHES.items():
        path = next(sample_dir.glob(f"samples_{bench}_*.json"), None)
        if path is None:
            skipped.append(bench)
            continue
        found += 1
        rows = json.loads(path.read_text())[key][:limit]
        for it in rows:
            imgs = it[field] if isinstance(it[field], list) else [it[field]]
            items.append({
                "bench": bench,
                "id": str(it["id"]),
                "type": it.get("cat") or it.get("type"),
                "question": it["q"],
                "options": it["opts"],
                "answer": it["ans"],
                "images": [str(sample_dir / p) for p in imgs],
            })
    if not found:
        raise FileNotFoundError(f"no samples_<{'|'.join(BENCHES)}>_*.json in {sample_dir}")
    if skipped:
        print(f"load_sample: no samples file for {', '.join(skipped)} in {sample_dir}, skipped", file=sys.stderr)
    return items


def question_text(item):
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"])
    return f"{item['question']}\n{opts}\nAnswer with the option's letter from the given choices directly."


def parse_letter(text, letters):
    """First standalone option letter in the reply, or None. Any capital is read, so MV-RoboBench's E counts; one that is
    not an option letter is skipped, so four-option questions parse exactly as with [A-D]."""
    for m in re.finditer(r"\b([A-Z])\b", text):
        if m.group(1) in letters:
            return m.group(1)
    return None
