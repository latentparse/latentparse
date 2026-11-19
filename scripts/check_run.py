"""Check the transcripts of a run: OK when every question has all its answer runs (6 in an Agent-Directed
Multi-Latent run, 8 in a Fixed-Sequence one, 4 in a one-encoder run), no error, a parsed answer in each run and findings the analyst produced. Prints the questions that
fail, the mean seconds per question and the share of reports with no findings.

    python check_run.py OUT_DIR [EXPECTED_QUESTIONS [ANSWER_RUNS]]
"""
import glob
import json
import sys

out = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else None
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ts = [json.load(open(p)) for p in glob.glob(f"{out}/transcripts/*.json")]
bad = [(t["id"], str(t.get("error", ""))[-300:] or (t.get("final_findings") or {}).get("summary"))
       for t in ts if "error" in t or len(t.get("final", {})) != runs
       or (t.get("final_findings") or {}).get("summary") == "The analyst produced no valid findings."
       or any(r.get("prediction") is None for r in t["final"].values())]
secs = [t["agent_seconds"] for t in ts if "agent_seconds" in t]
empty = sum(1 for t in ts if not (t.get("final_findings") or {}).get("findings"))
ok = not bad and (want is None or len(ts) == want)
print(f"{len(ts)} questions:", "OK" if ok else bad[:20] or f"expected {want}")
if ts:
    print(f"mean seconds per question {sum(secs) / max(len(secs), 1):.1f}, reports with no findings {empty}/{len(ts)}")
