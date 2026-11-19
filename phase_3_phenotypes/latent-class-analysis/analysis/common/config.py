"""Shared paths, seeds and the pre-declared class-enumeration rule.

Everything that a reader would need to reproduce the run lives here rather than
being scattered through the numbered scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(r"C:\Users\natha\Projects\work\phewas\latent-class-analysis")
DATA = ROOT / "data"
WORK = ROOT / "work"            # prepared model matrices (intermediate)
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
for _p in (WORK, RESULTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20260727
K_RANGE = range(1, 9)           # fit K = 1..8 per domain

# --- two-stage random starts (the Mplus "STARTS 200 20" idiom) ---------------
# Stage 1 runs many cheap short starts; the best few are then run to convergence.
# A solution is only trusted if its best log-likelihood is replicated (see
# fitting.replication_check), which is the standard guard against local maxima.
# The screening stage only has to *rank* starting points, so it runs on a random
# subsample; the surviving starts are then run to convergence on the full cohort.
N_SHORT_STARTS = 40
SHORT_ITER = 25
SHORT_SUBSAMPLE = 25_000
N_LONG_STARTS = 6
LONG_ITER = 500                 # EM converges well inside this at TOL (c1 K=6: 104 iters)
TOL = 1e-7

# Sensitivity arms exist to support an ARI comparison against the primary solution,
# not to establish a global optimum of their own, so they run on a smaller budget.
ARM_STARTS = dict(n_short=15, n_long=3)
# 24 cores. The per-start matrices are small enough that BLAS threading buys nothing
# and oversubscribes, so workers are pinned to one thread each (see run_all.sh) and
# the parallelism is spent on independent starts instead.
N_JOBS = int(os.environ.get("LCA_N_JOBS", 20))

# LCA_SMOKE=1 shrinks the sweep so a numbered script can be exercised end to end in
# minutes. It changes only the compute budget, never a modelling decision.
if os.environ.get("LCA_SMOKE"):
    K_RANGE = range(1, 4)
    N_SHORT_STARTS, SHORT_ITER = 6, 10
    N_LONG_STARTS, LONG_ITER = 2, 60
    SHORT_SUBSAMPLE = 8_000

# --- pre-declared selection rule (fixed BEFORE looking at any solution) ------
# At n = 74k-243k, BIC decreases essentially monotonically in K and the bootstrap
# LRT rejects K-1 at every step, so "lowest BIC" is not a usable stopping rule.
# We therefore select on the BIC *elbow* and require the supporting criteria below.
SELECTION = {
    "bic_elbow_threshold": 0.10,   # stop when a step buys <10% of the K=1->2 BIC drop
    "min_relative_entropy": 0.70,  # classification must be reasonably crisp
    "min_class_frac": 0.01,        # smallest class >= 1% of the cohort ...
    "min_class_n": 500,            # ... and >= 500 patients
    "min_split_half_ari": 0.70,    # solution must reproduce on an independent half
}

# --- external validation ------------------------------------------------------
# AHI is NEVER a model input. It enters only through 3-step estimation with the
# BCH correction, which relates class to a distal outcome without letting that
# outcome influence class formation.
AHI_COLS = ("index_ahi", "max_ahi")
AHI_SEVERITY_BINS = [0, 5, 15, 30, float("inf")]
AHI_SEVERITY_LABELS = ["none/<5", "mild 5-15", "moderate 15-30", "severe 30+"]

DOMAINS = ("c1_physiologic", "c2_symptomatic", "c3_comorbidities")
