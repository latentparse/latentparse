#!/bin/bash
# The source controls of LatentParse (Agent-Directed, Multi-Latent) on one machine with several GPUs: what does the gain come from?
# Each run of RUNS is made in turn, split over the GPUs with one blend_job.sh per GPU (shard g on GPU g), then checked
# (check_run.py):
#   latents   LatentParse itself (CLIP + VGGT + DINOv3), the reference, on the same GPUs and with the same BATCH
#   pixels    the photos' pixels on the CLIP and VGGT grids in place of the features; the analyst still sees no photo
#   question  no features and no sandbox: the analyst writes its report in one reply, from the question alone
#   photos    the same, from the question and the photos
# BATCH=<k> runs k questions at a time on each GPU (agent.py --batch); such a run goes to a folder of its own (..._b<k>).
# ARGS=<more agent.py arguments>, e.g. ARGS="--pilot 20" with TAG=trial. Rerunning resumes. GPU use is logged every
# 30 s in <out>/logs/gpu.csv.
# Usage:
#   ROOT=<repo> W=<work dir of prep_full.sh> [RUNS="question photos pixels"] [BENCH=mmsi] [GPUS=<n>] [BATCH=<k>] \
#   [ARGS=<args>] [TAG=<name>] [MODEL=$W/models/Qwen3.8-27B] [PY=$W/venv/bin/python] bash scripts/source_runs.sh
set -u
: "${ROOT:?}" "${W:?}"
RUNS=${RUNS:-question photos pixels}; BENCH=${BENCH:-mmsi}; PY=${PY:-$W/venv/bin/python}; MODEL=${MODEL:-$W/models/Qwen3.8-27B}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}
# One process per GPU, with BATCH sandboxes each: every sandbox gets its share of the cores, at least one.
T=$(( $(nproc) / (GPUS * ${BATCH:-1}) )); [ "$T" -lt 1 ] && T=1
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export PYTORCH_ALLOC_CONF=expandable_segments:True   # less fragmentation on long multi-image prompts, as in env.sh
ALL=$($PY -c "import sys; sys.path.insert(0, '$ROOT/direct_input'); from common import load_sample
print(sum(it['bench'] == '$BENCH' for it in load_sample('$W/data/full')))" 2> /dev/null)
for RUN in $RUNS; do
  case "$RUN" in
    latents) SRC=""; ANSWERS=6 ;;
    pixels|question|photos) SRC=$RUN; ANSWERS=4 ;;
    *) echo "RUNS holds latents, pixels, question or photos, not $RUN"; exit 1 ;;
  esac
  OUT=$W/runs/source_$RUN${BATCH:+_b$BATCH}${TAG:+_$TAG}/$BENCH
  mkdir -p "$OUT/logs"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 >> "$OUT/logs/gpu.csv" &
  smi=$!
  pids=()
  for g in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$g SOURCE=$SRC SAMPLE=$W/data/full LAT=$W/lat/full MODEL=$MODEL OUT=$OUT BENCH=$BENCH MODE=agent-directed \
      SHARD=$g N_SHARDS=$GPUS PY=$PY bash "$ROOT/scripts/blend_job.sh" &
    pids+=($!)
  done
  wait "${pids[@]}"
  kill $smi
  # With ARGS the run may hold fewer questions than the benchmark: then only what was run is checked.
  [ -n "${ARGS:-}" ] && N=$(ls "$OUT/transcripts" 2> /dev/null | wc -l) || N=$ALL
  echo "== $RUN: $OUT"
  $PY "$ROOT/scripts/check_run.py" "$OUT" "$N" $ANSWERS
done
