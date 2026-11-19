#!/bin/bash
# One-time preparation of MV-RoboBench (its 861 spatial questions, benchmark mvrb) and SPAR-Bench-Tiny (its 550
# multiple-choice questions, benchmark spar): each release at a pinned revision, the sample (data/build_mv_robobench.py,
# data/build_spar_tiny.py, checked against the reference build) and its CLIP, VGGT and DINOv3 features, then a check
# that every question has each feature map with one view per photo. Needs one GPU. Rerunning skips what is done; a build
# that stopped half way must be removed first (rm -r "$WORK/data/mvrb" or "$WORK/data/spar").
# Usage:
#   ROOT=<repo> WORK=<work dir> PY=<python> [BENCHES="mvrb spar"] bash scripts/prep_bench.sh
set -eu
: "${ROOT:?}" "${WORK:?}"; PY=${PY:-python}
declare -A REPO=([mvrb]=AaronFengZY24/MV_Robobench [spar]=jasonzhango/SPAR-Bench-Tiny)
declare -A REV=([mvrb]=869ea1113cf2f08498e81f57ced72597238455c0 [spar]=ae9bbc5297fd277123c42b0628d1f40bf72f89f1)
declare -A BUILD=([mvrb]=build_mv_robobench.py [spar]=build_spar_tiny.py)
declare -A SAMPLES=([mvrb]=886dc9fe25a55ff1bce860e603848c657673743425b58c9c1f6bdfbdd9865420   # samples_mvrb_full.json
                    [spar]=7753cd5c8f914bc1ab3f403345a8cd5b6987fe5fd75cc22fd7ff62c363c463a4)  # samples_spar_full.json
declare -A NQ=([mvrb]=861 [spar]=550)
check() { [ "$(sha256sum "$1" | cut -d" " -f1)" = "$2" ] || { echo "sha256 mismatch: $1"; exit 1; }; echo "ok $(basename "$1")"; }

HF="$(dirname "$PY")/hf"; [ -x "$HF" ] || HF=hf
for b in ${BENCHES:-mvrb spar}; do
  src="$WORK/hf/${REPO[$b]#*/}"
  $HF download "${REPO[$b]}" --repo-type dataset --revision "${REV[$b]}" --local-dir "$src" > /dev/null  # skips files it has
  [ -f "$WORK/data/$b/samples_${b}_full.json" ] || (cd /tmp && $PY -I "$ROOT/data/${BUILD[$b]}" "$src" "$WORK/data/$b")
  check "$WORK/data/$b/samples_${b}_full.json" "${SAMPLES[$b]}"
  for m in ${LATENT_MODELS:-clip vggt dinov3}; do
    [ -f "$WORK/lat/$b/$m/meta.json" ] || (cd "$ROOT/direct_input" && $PY extract_latents.py --model $m --sample "$WORK/data/$b" \
      --out "$WORK/lat/$b/$m")
  done
  (cd "$ROOT/direct_input" && $PY - "$WORK/data/$b" "$WORK/lat/$b" $b ${NQ[$b]} ${LATENT_MODELS:-clip vggt dinov3}) <<'PY'
import sys
from pathlib import Path
import numpy as np
from common import load_sample
sample, lat, bench, n, models = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5:]
items = load_sample(sample)
dims = {"clip": 1024, "vggt": 2048, "dinov3": 1024}
bad = []
for it in items:
    shapes = {}
    for m in models:
        p = lat / m / bench / f"{it['id']}.npy"
        if not p.exists():
            bad.append(f"{m}/{it['id']}: missing")
            continue
        shapes[m] = np.load(p, mmap_mode="r").shape
        if shapes[m][0] != len(it["images"]) or shapes[m][-1] != dims[m]:
            bad.append(f"{m}/{it['id']}: shape {shapes[m]}, {len(it['images'])} photos")
    if "vggt" in shapes and "dinov3" in shapes and shapes["vggt"][:3] != shapes["dinov3"][:3]:
        bad.append(f"{it['id']}: dinov3 grid {shapes['dinov3'][:3]} vs vggt {shapes['vggt'][:3]}")
print(f"{bench}: {len(items)} questions, latents {models}: {len(bad)} problems {bad[:5]}")
sys.exit(1 if bad or len(items) != n else 0)
PY
  echo "prep done: SAMPLE=$WORK/data/$b LAT=$WORK/lat/$b"
done
