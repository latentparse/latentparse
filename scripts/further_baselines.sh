#!/bin/bash
# The further baselines of the appendix on MMSI-Bench (Qwen3.8-27B), one per TASK:
#   pca       the features given directly as a color image: the first three principal components of each photo's patch
#             features as red, green and blue (answer_qwen.py format pcargb), for CLIP, VGGT, DINOv3 and all three
#   sc        self-consistency: five answers sampled from the photos at temperature 0.7, majority vote
#   camera    the camera poses decoded by VGGT's camera head, written as text (direct_input/cam_head_notes.py) and given to the
#             answering VLM in place of the report, as LatentParse gives its report
# W is the work dir of scripts/prep_full.sh.
# Usage:
#   ROOT=<repo> W=<work dir> [MODEL=$W/models/Qwen3.8-27B] PY=<python> bash scripts/further_baselines.sh pca|sc|camera
set -eu
: "${ROOT:?}" "${W:?}" "${1:?task: pca, sc or camera}"; PY=${PY:-python}; MODEL=${MODEL:-$W/models/Qwen3.8-27B}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IDS=$(cd "$ROOT/direct_input" && $PY -c "from common import load_sample
print(' '.join(str(it['id']) for it in load_sample('$W/data/full') if it['bench'] == 'mmsi'))")
cd "$ROOT/direct_input"
case "$1" in
  pca)
    $PY answer_qwen.py --sample "$W/data/full" --model "$MODEL" --latents-root "$W/lat/full" --out "$W/runs/direct_pca" \
      --runs clip:pcargb vggt:pcargb dinov3:pcargb blend:pcargb --ids $IDS ;;
  sc)
    $PY answer_qwen.py --sample "$W/data/full" --model "$MODEL" --out "$W/runs/self_consistency" --runs baseline \
      --samples 5 --temperature 0.7 --ids $IDS ;;
  camera)
    $PY cam_head_notes.py --sample "$W/data/full" --out "$W/runs/camera_head" --types all --bench mmsi
    cd "$ROOT/latentparse"
    DEVICE_MAP=cuda $PY agent.py --sample "$W/data/full" --latents-root "$W/lat/full" --model "$MODEL" \
      --out "$W/runs/camera_head" --blind --boxes --single-pass --index-chars 16000 --latents clip vggt dinov3 \
      --answer-formats pad --cond3-latents --bench mmsi --answers-only agent_only --notes-header camera_head ;;
  *) echo "task must be pca, sc or camera"; exit 1 ;;
esac
