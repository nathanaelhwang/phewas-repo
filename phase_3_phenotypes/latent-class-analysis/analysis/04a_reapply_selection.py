"""Recompute the selection columns on already-fitted sweeps.

The fit-index CSVs carry every raw quantity the selection rule reads (log-likelihood,
BIC, entropy, class sizes, component SDs, degeneracy, stability, replication), so a
change to the *rule* can be applied without refitting anything. Used when the elbow
search was corrected to exclude variance-collapsed fits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fitting  # noqa: E402
from common.config import RESULTS  # noqa: E402

RAW = ["K", "loglik", "n_params", "bic", "aic", "caic", "sabic", "rel_entropy",
       "min_class_frac", "min_class_n", "min_component_sd", "degenerate", "n_iter",
       "converged", "ll_replicated", "seconds"]

for path in sorted(RESULTS.glob("c*/c*_fit_indices*.csv")):
    fit = pd.read_csv(path)
    stab = fit[["K", "split_half_ari"]] if "split_half_ari" in fit.columns else None
    out = fitting.apply_selection(fit[[c for c in RAW if c in fit.columns]], stab)
    out.to_csv(path, index=False)
    rec = out.loc[out.recommended, "K"]
    print(f"{path.relative_to(RESULTS)}: elbow K={out.bic_elbow_K.iloc[0]} "
          f"(max non-degenerate K={out.max_non_degenerate_K.iloc[0]}), "
          f"selects K={int(rec.iloc[0]) if len(rec) else None}")
