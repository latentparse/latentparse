#!/bin/bash
# One shard of a run in which the analyst reads one encoder alone with the basic tools (the + CLIP, + DINOv3 and
# + VGGT rows). LATENT=clip|dinov3|vggt (default vggt).
#   MODE=agent-directed   the analyst picks the tools and writes the code
#   MODE=fixed-sequence   every basic tool runs once with fixed arguments
# MARKS=1 lists the marks drawn into the photos (MV-RoboBench, SPAR-Bench-Tiny). A closed VLM gets the latent of its
# extra answer run as an image: ARGS="--answer-formats image". Rerunning resumes; all shards of one benchmark and mode
# write into the same OUT.
# Usage:
#   ROOT=<repo> SAMPLE=<sample dir> LAT=<features dir> MODEL=<model dir or API model> OUT=<output dir> \
#   BENCH=mmsi|mvrb|spar MODE=agent-directed|fixed-sequence [LATENT=clip|dinov3|vggt] [MARKS=1] [ARGS=<args>] \
#   PY=<python> SHARD=<0..n-1> N_SHARDS=<n> bash scripts/one_encoder_job.sh
set -u
: "${ROOT:?}" "${SAMPLE:?}" "${LAT:?}" "${MODEL:?}" "${OUT:?}" "${BENCH:?}" "${SHARD:?}" "${MODE:?}"; PY=${PY:-python}; N=${N_SHARDS:-4}
OPTS="--blind --boxes --single-pass --latent ${LATENT:-vggt} --index-chars 16000"
case "$MODE" in
  agent-directed|free) ;;
  fixed-sequence|fixed) OPTS="$OPTS --all-tools" ;;
  *) echo "MODE must be agent-directed or fixed-sequence"; exit 1 ;;
esac
case "${MARKS:-}" in
  "") ;;
  1) OPTS="$OPTS --marks" ;;
  *) echo "MARKS must be 1 or unset"; exit 1 ;;
esac
TAG="$BENCH ${LATENT:-vggt} $MODE${MARKS:+ marks} shard $SHARD/$N"
mkdir -p "$OUT/logs"; cd "$ROOT/latentparse"
echo "[$(date "+%F %T")] $TAG on $(hostname)"
$PY agent.py --sample "$SAMPLE" --latents-root "$LAT" --model "$MODEL" --out "$OUT" $OPTS ${ARGS:-} \
  --bench "$BENCH" --shard "$SHARD/$N" >> "$OUT/logs/shard$SHARD.log" 2>&1
rc=$?
echo "[$(date "+%F %T")] $TAG exit $rc"
