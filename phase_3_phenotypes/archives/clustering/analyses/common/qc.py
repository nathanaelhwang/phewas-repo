"""Step-1 QC: per-feature z-score distribution + extreme-outlier report (no row-level values)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew


def zscore_outlier_report(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-feature summary on observed cells: coverage, spread, skew and extreme-z counts.

    z is computed on observed (non-NaN) values.  |z|>5 flags clinically implausible survivors
    that the enumerated range rules did not already turn into NaN.
    """
    rows = []
    for c in feats:
        a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        obs = a[~np.isnan(a)]
        if obs.size == 0:
            rows.append({"feature": c, "n_obs": 0, "pct_missing": 1.0}); continue
        mu, sd = obs.mean(), obs.std()
        z = (obs - mu) / sd if sd > 1e-9 else np.zeros_like(obs)
        rows.append({
            "feature": c, "n_obs": int(obs.size),
            "pct_missing": round(float(np.isnan(a).mean()), 4),
            "mean": round(float(mu), 4), "sd": round(float(sd), 4),
            "min": round(float(obs.min()), 4), "max": round(float(obs.max()), 4),
            "skew": round(float(skew(obs)), 3) if obs.size > 2 else None,
            "z_min": round(float(z.min()), 2), "z_max": round(float(z.max()), 2),
            "n_abs_z_gt4": int((np.abs(z) > 4).sum()),
            "n_abs_z_gt5": int((np.abs(z) > 5).sum()),
            "pct_zero": round(float((obs == 0).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("n_abs_z_gt5", ascending=False)
