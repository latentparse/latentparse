#!/bin/bash
# Photo only and direct input on one benchmark: the VLM answers from the photos alone (baseline), and from the photos
# with each encoder's features, or all three, given directly as a global min-max image next to each photo.
# Closed VLMs use direct_input/answer_gpt.py or direct_input/answer_claude.py with the same --runs.
# Usage:
#   ROOT=<repo> SAMPLE=<sample dir> LAT=<features dir> MODEL=<model dir> OUT=<output dir> PY=<python> bash scripts/direct_input.sh
set -eu
: "${ROOT:?}" "${SAMPLE:?}" "${LAT:?}" "${MODEL:?}" "${OUT:?}"; PY=${PY:-python}
soft=""; case "$MODEL" in *gemma*) soft="--max-soft-tokens 560" ;; esac
cd "$ROOT/direct_input"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY answer_qwen.py --sample "$SAMPLE" --model "$MODEL" $soft \
  --latents-root "$LAT" --out "$OUT" --runs baseline clip:image dinov3:image vggt:image blend:image
