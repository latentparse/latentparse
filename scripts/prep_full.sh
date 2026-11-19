#!/bin/bash
# One-time preparation of MMSI-Bench (and 3DSRBench, used only to calibrate the tool thresholds): the environment
# check, the VLM at a pinned revision, the benchmark files at pinned revisions with their checksums, the sample
# (data/build_from_hf.py --full, checked against the reference build) and its CLIP, VGGT and DINOv3 features
# (LATENT_MODELS="clip vggt" to extract only those). A model already in WORK/models is not downloaded again.
# Needs one GPU for about 30 minutes.
# Usage:
#   ROOT=<repo> WORK=<work dir> PY=<python> [LATENT_MODELS="clip vggt dinov3"] \
#   MODEL_REPO=Qwen/Qwen3.8-27B MODEL_REV=<revision> bash scripts/prep_full.sh
set -eu
: "${ROOT:?}" "${WORK:?}" "${MODEL_REPO:?}" "${MODEL_REV:?}"; PY=${PY:-python}
DSR_PARQUET=e304521774dacfa9ed61a07af859f9997cff9265409c9a8d03da5f89bb0a217c
DSR_ZIP=391673d4a38d915ffe7d86eb964343937cd3f5bef7dd36374da3f36780811f8b
MMSI_REV=ec7c92bfaf7728fcca1d61e3e224e190af309436
MMSI_PARQUET=72a9c94699ca88f3eba79f330ab28965ee537bccd7cbf563265982a6ff8b74b4
FULL_3DSR=2c2c33ab0f706b853fab04611cfb756953b2184dd13797f2c1a93d4dd786aee3  # data/full/samples_3dsr_full.json
FULL_MMSI=8a4bec69070a940d86fd0ebfea63f4c68b573a9298feb26cd059ba2e0932637b  # data/full/samples_mmsi_full.json
check() { [ "$(sha256sum "$1" | cut -d" " -f1)" = "$2" ] || { echo "sha256 mismatch: $1"; exit 1; }; echo "ok $(basename "$1")"; }

$PY -c "import torch,transformers,timm,numpy,PIL,vggt,pyarrow
from transformers import InternVLForConditionalGeneration
want={'torch':'2.13.0','transformers':'5.22.1','timm':'1.0.31','numpy':'2.4.6'}
got={'torch':torch.__version__.split('+')[0],'transformers':transformers.__version__,'timm':timm.__version__,'numpy':numpy.__version__}
bad={k:(got[k],v) for k,v in want.items() if got[k]!=v}
print('versions ok' if not bad else f'version mismatch (have, want): {bad}')"
HF="$(dirname "$PY")/hf"; [ -x "$HF" ] || HF=hf
M="$WORK/models/$(basename "$MODEL_REPO")"
[ -f "$M/model.safetensors.index.json" ] || $HF download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$M"
[ -f "$WORK/hf/3DSRBench/coco_images.zip" ] || \
  $HF download ccvl/3DSRBench 3dsrbench_v1-00000-of-00001.parquet coco_images.zip --repo-type dataset --local-dir "$WORK/hf/3DSRBench"
[ -f "$WORK/hf/MMSI-Bench/MMSI_Bench.parquet" ] || \
  $HF download RunsenXu/MMSI-Bench MMSI_Bench.parquet --repo-type dataset --revision $MMSI_REV --local-dir "$WORK/hf/MMSI-Bench"
check "$WORK/hf/3DSRBench/3dsrbench_v1-00000-of-00001.parquet" $DSR_PARQUET
check "$WORK/hf/3DSRBench/coco_images.zip" $DSR_ZIP
check "$WORK/hf/MMSI-Bench/MMSI_Bench.parquet" $MMSI_PARQUET
[ -f "$WORK/data/full/samples_mmsi_full.json" ] || $PY "$ROOT/data/build_from_hf.py" --full \
  --dsr-parquet "$WORK/hf/3DSRBench/3dsrbench_v1-00000-of-00001.parquet" --dsr-zip "$WORK/hf/3DSRBench/coco_images.zip" \
  --mmsi-parquet "$WORK/hf/MMSI-Bench/MMSI_Bench.parquet" --out "$WORK/data/full"
check "$WORK/data/full/samples_3dsr_full.json" $FULL_3DSR
check "$WORK/data/full/samples_mmsi_full.json" $FULL_MMSI
cd "$ROOT/direct_input"
for m in ${LATENT_MODELS:-clip vggt dinov3}; do
  [ -f "$WORK/lat/full/$m/meta.json" ] || $PY extract_latents.py --model $m --sample "$WORK/data/full" --out "$WORK/lat/full/$m"
done
echo "prep done: SAMPLE=$WORK/data/full LAT=$WORK/lat/full MODEL=$M"
