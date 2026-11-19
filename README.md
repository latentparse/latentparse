# LatentParse

Code for **Parsing Visual Latents for Multimodal Spatial Reasoning**.

Vision encoders such as CLIP, DINOv3 and VGGT may already carry the 3D information that vision-language models (VLMs)
miss on spatial questions. We study whether a VLM can parse these latent features in context, without any training.
Given directly as an image next to the photo, the features do not help. **LatentParse** instead lets an analyst VLM
that never sees the photos write code in a sandbox to parse the patch features of the three encoders with a small set
of tools, and pass a short report to the VLM that answers from the photos.

The same code runs open VLMs from Hugging Face (Qwen3.8-27B, Gemma 4, InternVL3-38B) and closed VLMs over their APIs
(GPT-6 Luna, Claude Haiku) on MMSI-Bench, MV-RoboBench and SPAR-Bench-Tiny.

## Repository layout

```
data/          benchmark builders at pinned Hugging Face revisions, and the question id lists used in the paper
direct_input/  feature extraction (CLIP, VGGT, DINOv3), photo only and direct input, VGGT camera poses as text
latentparse/   the method: analyst, sandbox and tools, report, answering VLM, tool calibration, tests
scripts/       one-time data preparation and the run scripts behind every table
analysis/      paired exact sign tests, and the photo-only answers of the paired MMSI-Bench questions
```

## The method in the code

| Paper | Code |
| --- | --- |
| Latent features, no decoding | `direct_input/extract_latents.py`: CLIP ViT-L/14 patch tokens, the last aggregator layer of VGGT-1B, DINOv3 ViT-L/16 on the VGGT grid |
| Direct input, global min-max image | `direct_input/answer_qwen.py --runs <encoder>:image` (and `answer_gpt.py`, `answer_claude.py`) |
| Analyst that never sees the photos | `latentparse/agent.py`, `run_agent()` |
| Tools `locate`, `support`, `iou`, `motion`, `track`, `relate`, `layout` | `latentparse/sandbox.py`, class `Helpers` |
| Tool settings | the constants at the top of `latentparse/sandbox.py`, fitted once by `latentparse/calib_blend.py` |
| Analyst prompt | `blend_prompt()` in `latentparse/agent.py` |
| Agent-Directed mode | default: the analyst writes code, at most 4 steps |
| Fixed-Sequence mode | `--all-tools`: `blend_steps()` runs every tool once |
| Report | the analyst's `FINDINGS` JSON: findings with a reliability each, a summary and boxes |
| Answering VLM | `final_answers()` in `latentparse/agent.py`, in the prompt of the photo-only baseline. The answer run `agent_only` is LatentParse |
| Mismatched and Random Noise controls | `--wrong-latent` and `--noise-latent` (aliases `--mismatched`, `--random-noise`) |
| Source controls | `--report-from question`, `--report-from photos`, `--latents clip vggt --pixels` |
| Marks drawn into MV-RoboBench and SPAR photos | `--marks`, `latentparse/marks.py` |

## Setup

Python 3.11 and a CUDA GPU. The runs in the paper used these versions:

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce"
```

Closed VLMs read `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` from the environment.

| Model | Hugging Face repo | Revision |
| --- | --- | --- |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Gemma 4 | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| InternVL3-38B | `OpenGVLab/InternVL3-38B-hf` | `b2a05c0c325235f7530d8274c313a1d01082e069` |
| CLIP | `openai/clip-vit-large-patch14` | |
| DINOv3 | `timm/vit_large_patch16_dinov3.lvd1689m` | |
| VGGT | `facebook/VGGT-1B` | |

## Data and features

Each script downloads the benchmark at a pinned revision, builds the sample, checks it against our build by sha256 and
extracts the CLIP, DINOv3 and VGGT features into `WORK/lat/<benchmark>`.

```bash
# MMSI-Bench (and 3DSRBench, used only to calibrate the tool settings), and the VLM
ROOT=$PWD WORK=/path/to/work PY=$(which python) MODEL_REPO=Qwen/Qwen3.8-27B \
  MODEL_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 bash scripts/prep_full.sh
# MV-RoboBench (861 spatial questions) and SPAR-Bench-Tiny (550 multiple-choice questions)
ROOT=$PWD WORK=/path/to/work PY=$(which python) bash scripts/prep_bench.sh
```

## Running LatentParse

A run is split into shards that can run in parallel and resume after an interruption. All shards of a run write into
the same output folder.

```bash
W=/path/to/work
for s in 0 1 2 3; do
  ROOT=$PWD SAMPLE=$W/data/full LAT=$W/lat/full MODEL=$W/models/Qwen3.8-27B OUT=$W/runs/latentparse/mmsi \
  BENCH=mmsi MODE=agent-directed PY=$(which python) SHARD=$s N_SHARDS=4 bash scripts/blend_job.sh
done
```

On MV-RoboBench and SPAR-Bench-Tiny, use `SAMPLE=$W/data/mvrb LAT=$W/lat/mvrb BENCH=mvrb` (or `spar`) and add `MARKS=1`.
For a closed VLM, set `MODEL=gpt-6-luna` or `MODEL=claude-haiku-5-5`.

Each run writes `OUT/transcripts/<bench>_<id>.json` with every message, code step, tool output and the report, and one
`OUT/<answer run>/results.jsonl` and `metrics.json` per answer run. **`agent_only` is LatentParse**: the photos, the
report and the question. `python scripts/check_run.py OUT` checks that a run is complete.

## Reproducing the paper

| Result | Command |
| --- | --- |
| Photo only and direct input | `scripts/direct_input.sh`, or `direct_input/answer_gpt.py` and `direct_input/answer_claude.py` with the same `--runs` |
| LatentParse | `scripts/blend_job.sh MODE=agent-directed` |
| + CLIP, + DINOv3, + VGGT | `scripts/one_encoder_job.sh MODE=agent-directed LATENT=clip\|dinov3\|vggt` |
| Agent-Directed and Fixed-Sequence | `MODE=agent-directed` and `MODE=fixed-sequence` |
| Mismatched and Random Noise | `blend_job.sh CONTROL=mismatched\|noise`, and `answer_qwen.py --runs blend:image:wrong blend:image:noise` |
| Encoder layers | `extract_latents.py --layer 5 --ids-file data/layer300_ids.txt` into a separate features folder, then a run with `--layer vggt:5` or `--layer dinov3:5` |
| Source of the gain | `scripts/source_runs.sh`, tables with `latentparse/source_table.py` |
| Further baselines: PCA color images, self-consistency, VGGT camera poses as text | `scripts/further_baselines.sh pca\|sc\|camera` |
| Paired sign tests | `analysis/paired_stats.py` |
| Tool settings | `latentparse/calib_blend.py`; the fitted values are in `latentparse/sandbox.py` |

MV-RoboBench is scored on its 861 spatial questions, the union of `data/mvrb_box605_ids.txt` and
`data/mvrb_viewpoint256_ids.txt`.

Paired statistics compare two runs question by question with an exact two-sided sign test:

```bash
python analysis/paired_stats.py analysis/mmsi_photo_only_paired986.json $W/runs/latentparse/mmsi/agent_only/results.jsonl
```

`analysis/mmsi_photo_only_paired986.json` holds the photo-only answer, right or wrong, of the 986 MMSI-Bench questions
paired in the appendix tables.

## Tests

The tests run on CPU with synthetic features, a fake API model and no weights. test_agent and test_marks take seconds,
test_tools a few minutes:

```bash
cd latentparse
python test_agent.py   # every paper setting end to end; the requests must match those recorded with the paper's code
python test_tools.py   # the tools on synthetic scenes with planted objects and known camera motion
python test_marks.py   # the marks drawn into MV-RoboBench and SPAR photos
```

## Citation

```bibtex
@article{latentparse2026,
  title   = {Parsing Visual Latents for Multimodal Spatial Reasoning},
  author  = {},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
