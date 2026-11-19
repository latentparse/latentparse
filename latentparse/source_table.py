#!/usr/bin/env python3
"""The tables of the source controls: python source_table.py NAME=RUN_DIR [NAME=RUN_DIR ...] --out DIR

What does the gain of the Agent-Directed blend come from? Each RUN_DIR is the output of one agent.py run on the same
questions (scripts/source_runs.sh), and NAME says what the analyst's report was made from:
  latents   the blend itself, CLIP + VGGT + DINOv3 (the reference, when it was run again on the same GPUs)
  pixels    --pixels --latents clip vggt: the photos' pixels in place of the latents, no photo shown
  photos    --report-from photos: the question and the photo(s), no tools
  question  --report-from question: the question alone
The answer setting compared is agent_only (photo(s) + report), as in the paper. Photo only is the agent_boxes_nonotes
setting of a report run (question, else photos): no box is ever drawn there and no notes are given, so its prompt is
the photo-only baseline's, answered on the same GPUs and in the same batches as the runs it is compared with.
Every run is compared with photo only, and with the blend when it is there, on the questions both answered: how many
only one of the two gets right, and the two-sided exact sign test on those (compare_runs.sign_p).
Writes DIR/README.md (the tables), DIR/table.tex (the first table, for the paper) and DIR/summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "direct_input"))
from compare_runs import sign_p  # noqa: E402

ROWS = (("latents", "Latent features (the blend)"), ("pixels", "Pixels in place of the latent features"),
        ("photos", "Photo(s), no tools"), ("question", "Question alone"))
PHOTO = "Photo only (no report)"


def load(run_dir, setting):
    """{(bench, id): row} of one answer setting of a run; {} when the run does not have it."""
    f = Path(run_dir) / setting / "results.jsonl"
    return {(r["bench"], r["id"]): r for r in map(json.loads, f.open())} if f.exists() else {}


def versus(rows, other):
    """rows against other on the questions both answered: (right only in rows, right only in other, p)."""
    keys = [k for k in rows if k in other]
    won = sum(rows[k]["correct"] and not other[k]["correct"] for k in keys)
    lost = sum(other[k]["correct"] and not rows[k]["correct"] for k in keys)
    return {"questions": len(keys), "right_only": won, "wrong_only": lost, "p": round(sign_p(won, won + lost), 4)}


def summary(dirs):
    runs = {name: load(d, "agent_only") for name, d in dirs.items()}
    base_run = next((n for n in ("question", "photos") if n in dirs), None)
    photo = load(dirs[base_run], "agent_boxes_nonotes") if base_run else {}
    table = []
    for name, label, rows in [("photo_only", PHOTO, photo)] + [(n, lab, runs[n]) for n, lab in ROWS if n in runs]:
        if not rows:
            continue
        n, right = len(rows), sum(r["correct"] for r in rows.values())
        row = {"name": name, "label": label, "n": n, "correct": right, "accuracy": round(100 * right / n, 1)}
        if name != "photo_only" and photo:
            row["vs_photo_only"] = versus(rows, photo)
        if name not in ("photo_only", "latents") and runs.get("latents"):
            row["vs_latents"] = versus(rows, runs["latents"])
        types = sorted({r["type"] for r in rows.values()})
        row["by_type"] = {t: [sum(r["correct"] for r in rows.values() if r["type"] == t), sum(r["type"] == t for r in rows.values())]
                          for t in types}
        table.append(row)
    checks = {}
    for name, d in dirs.items():
        d = Path(d)
        stats = json.loads((d / "agent_stats.json").read_text()) if (d / "agent_stats.json").exists() else {}
        config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        settings = {p.parent.name: load(d, p.parent.name) for p in sorted(d.glob("*/results.jsonl"))}
        checks[name] = {"folder": str(d), "questions": stats.get("questions"), "errors": stats.get("errors"),
                        "missing_final": stats.get("missing_final"), "seconds_per_question": stats.get("mean_agent_seconds"),
                        "batch": config.get("batch", 1), "model": Path(str(config.get("model", ""))).name,
                        "settings": {s: [sum(r["correct"] for r in rows.values()), len(rows)] for s, rows in settings.items()}}
    return {"photo_only_from": base_run, "table": table, "runs": checks}


def wl(v):
    return f"{v['right_only']}W/{v['wrong_only']}L"


def markdown(s, title):
    lines = [f"# {title}", "",
             "What the analyst's report is made from, one run per row; the model that answers always gets the photo(s) and "
             "the report (`agent_only`). Photo only is the same model with no report (the `agent_boxes_nonotes` setting of "
             f"the `{s['photo_only_from']}` run, where no box is drawn). W/L: questions only this row / only the other one "
             "gets right, on the questions both answered; p: two-sided exact sign test.", "",
             "| Analyst's report made from | n | Correct | Accuracy | vs photo only | p | vs latent features | p |",
             "|---|---|---|---|---|---|---|---|"]
    for r in s["table"]:
        a, b = r.get("vs_photo_only"), r.get("vs_latents")
        lines.append(f"| {r['label']} | {r['n']} | {r['correct']} | {r['accuracy']:.1f} | {wl(a) if a else ''} | "
                     f"{a['p'] if a else ''} | {wl(b) if b else ''} | {b['p'] if b else ''} |")
    types = sorted({t for r in s["table"] for t in r["by_type"]})
    lines += ["", "## Accuracy by question type", "", "| Question type | n | " + " | ".join(r["label"] for r in s["table"]) + " |",
              "|---|---|" + "---|" * len(s["table"])]
    for t in types:
        cells = [f"{100 * r['by_type'][t][0] / r['by_type'][t][1]:.1f}" if t in r["by_type"] else "" for r in s["table"]]
        lines.append(f"| {t} | {max(r['by_type'].get(t, [0, 0])[1] for r in s['table'])} | " + " | ".join(cells) + " |")
    names = sorted({k for c in s["runs"].values() for k in c["settings"]})
    lines += ["", "## The runs", "", "| Run | Questions | Errors | Missing answers | s per question | Batch | Model | "
              + " | ".join(f"`{k}`" for k in names) + " |", "|---|---|---|---|---|---|---|" + "---|" * len(names)]
    for name, c in s["runs"].items():
        cells = [f"{c['settings'][k][0]}/{c['settings'][k][1]}" if k in c["settings"] else "" for k in names]
        lines.append(f"| `{name}` | {c['questions']} | {c['errors']} | {c['missing_final']} | {c['seconds_per_question']} | "
                     f"{c['batch']} | {c['model']} | " + " | ".join(cells) + " |")
    lines += ["", "Runs made with `--batch` K compare only with runs made with the same K. Seconds per question include "
              "the time a question waits for the others of its batch."]
    return "\n".join(lines) + "\n"


def latex(s):
    both = any("vs_latents" in r for r in s["table"])
    lines = ["% Source controls. Generated by latentparse/source_table.py; W/L and p as in README.md.",
             "\\begin{tabular}{lcc" + ("c" if both else "") + "}", "\\toprule",
             "Analyst's report made from & Acc. & vs.\\ photo only" + (" & vs.\\ latent features" if both else "") + " \\\\",
             "\\midrule"]
    for r in s["table"]:
        cell = lambda v: f"{wl(v)}, $p$={v['p']:.3f}" if v else "--"  # noqa: E731
        lines.append(f"{r['label']} & {r['accuracy']:.1f} & {cell(r.get('vs_photo_only'))}"
                     + (f" & {cell(r.get('vs_latents'))}" if both else "") + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", metavar="NAME=RUN_DIR", help="NAME: " + ", ".join(n for n, _ in ROWS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Source controls of the Agent-Directed blend")
    args = ap.parse_args()
    dirs = dict(a.split("=", 1) for a in args.runs)
    if bad := [n for n in dirs if n not in dict(ROWS)]:
        ap.error(f"unknown run names {bad}: " + ", ".join(n for n, _ in ROWS))
    s = summary(dirs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    (out / "README.md").write_text(markdown(s, args.title))
    (out / "table.tex").write_text(latex(s))
    print(markdown(s, args.title).split("\n## ")[0])


if __name__ == "__main__":
    main()
