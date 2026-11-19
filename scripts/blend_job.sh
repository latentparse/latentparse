#!/bin/bash
# One shard of a LatentParse run with the Multi-Latent features (CLIP, VGGT and DINOv3).
#   MODE=agent-directed   the analyst picks the tools and writes the code (the method)
#   MODE=fixed-sequence   every tool runs once with fixed arguments and the analyst writes the report from the output
# Rerunning resumes: finished questions are skipped, and all shards of one benchmark and mode write into the same OUT.
# Options, all off when unset:
#   MARKS=1             MV-RoboBench and SPAR-Bench-Tiny: the boxes or points drawn into the photos are found by color
#                       and listed for the analyst after the question
#   CONTROL=mismatched  every question reads the latent features of another question
#   CONTROL=noise       Random Noise with each feature map's per-channel mean and std
#   SOURCE=pixels       source control: the photos' pixels on the CLIP and VGGT grids in place of the features
#   SOURCE=question     source control: no features and no sandbox, the report is written from the question alone
#   SOURCE=photos       source control: the same, from the question and the photos
#   BATCH=<k>           k questions at a time on the GPU; compare only runs made with the same k
#   ARGS=<args>         more agent.py arguments, e.g. ARGS="--pilot 20" for a trial on 20 questions
# MODEL may also be gpt-6-luna or claude-haiku-5-5 (api_chat.py; OPENAI_API_KEY or ANTHROPIC_API_KEY).
# Usage:
#   ROOT=<repo> SAMPLE=<sample dir> LAT=<features dir> MODEL=<model dir or API model> OUT=<output dir> \
#   BENCH=mmsi|mvrb|spar MODE=agent-directed|fixed-sequence [options] PY=<python> SHARD=<0..n-1> N_SHARDS=<n> \
#   bash scripts/blend_job.sh
set -u
: "${ROOT:?}" "${SAMPLE:?}" "${LAT:?}" "${MODEL:?}" "${OUT:?}" "${BENCH:?}" "${SHARD:?}" "${MODE:?}"; PY=${PY:-python}; N=${N_SHARDS:-4}
COMMON="--blind --boxes --single-pass --index-chars 16000 --answer-formats pad --cond3-latents"
INPUT="--latents clip vggt dinov3"
case "$MODE" in
  agent-directed|free) EXTRA="" ;;
  fixed-sequence|fixed) EXTRA="--all-tools" ;;
  *) echo "MODE must be agent-directed or fixed-sequence"; exit 1 ;;
esac
case "${CONTROL:-}" in
  "") ;;
  mismatched|wrong) EXTRA="$EXTRA --wrong-latent" ;;
  noise) EXTRA="$EXTRA --noise-latent" ;;
  *) echo "CONTROL must be mismatched or noise"; exit 1 ;;
esac
case "${MARKS:-}" in
  "") ;;
  1) EXTRA="$EXTRA --marks" ;;
  *) echo "MARKS must be 1 or unset"; exit 1 ;;
esac
case "${SOURCE:-}" in
  "") ;;
  pixels) INPUT="--latents clip vggt --pixels" ;;
  question|photos) INPUT="--report-from $SOURCE" ;;
  *) echo "SOURCE must be pixels, question or photos"; exit 1 ;;
esac
[ -n "${BATCH:-}" ] && EXTRA="$EXTRA --batch $BATCH"
TAG="$BENCH $MODE${CONTROL:+ $CONTROL}${MARKS:+ marks}${SOURCE:+ source $SOURCE}${BATCH:+ batch $BATCH} shard $SHARD/$N"
mkdir -p "$OUT/logs"; cd "$ROOT/latentparse"
echo "[$(date "+%F %T")] $TAG on $(hostname)"
$PY agent.py --sample "$SAMPLE" --latents-root "$LAT" --model "$MODEL" --out "$OUT" $COMMON $INPUT $EXTRA ${ARGS:-} \
  --bench "$BENCH" --shard "$SHARD/$N" >> "$OUT/logs/shard$SHARD.log" 2>&1
rc=$?
echo "[$(date "+%F %T")] $TAG exit $rc"
