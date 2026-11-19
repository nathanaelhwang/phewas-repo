#!/usr/bin/env bash
# Full pipeline. Stages 1-3 are run SEQUENTIALLY: each already parallelises its own
# random starts across ~20 workers, so running two domains at once just oversubscribes
# the 24 cores and makes both slower.
#
#   bash analysis/run_all.sh            full run
#   LCA_SMOKE=1 bash analysis/run_all.sh   K=1..3 with reduced starts, for a code check
#
# Stage 4 is the pause point: read results/MODEL_SELECTION.md, write
# results/K_CHOICES.json, then run 05_finalize.py.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-C:/Users/natha/anaconda3/envs/myenv2/python.exe}"

# One BLAS thread per worker: these matrices are far too small for threaded BLAS to
# pay for itself, and leaving it on turns 20 workers into 20x N threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

log() { echo; echo "=============== $* ==============="; date; }

log "00 prepare"
"$PY" -W ignore -u analysis/00_prepare.py

log "01 fit c1 (latent profile analysis)"
"$PY" -W ignore -u analysis/01_fit_c1.py

log "02 fit c2 (mixed-mode latent class cluster analysis)"
"$PY" -W ignore -u analysis/02_fit_c2.py

log "03 fit c3 (latent class analysis)"
"$PY" -W ignore -u analysis/03_fit_c3.py

log "04 report"
"$PY" -W ignore -u analysis/04_report.py

echo
echo "PAUSE: read results/MODEL_SELECTION.md, write results/K_CHOICES.json,"
echo "       then run: $PY analysis/05_finalize.py"
date
