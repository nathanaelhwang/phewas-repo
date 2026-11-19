"""Lossless plotting coordinates for the existing lifelines AJ estimator."""
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter


def event_curve(t, ev, horizon):
    """Keep every fitted CIF jump plus annual checkpoints and the display horizon.

    Fitting (including seeded tie jitter) matches steps 11 and 13. Flat censoring
    and competing-death coordinates can be omitted without changing the CIF.
    """
    t, ev = np.asarray(t, dtype=float), np.asarray(ev)
    ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
    ajf.fit(t, ev, event_of_interest=1)
    cif = ajf.cumulative_density_.iloc[:, 0]
    jumps = cif.index.to_numpy()[cif.diff().fillna(0).to_numpy() != 0]
    grid = np.unique(np.r_[np.arange(0, horizon + 1), jumps[(jumps > 0) & (jumps <= horizon)]])
    positions = np.searchsorted(cif.index.to_numpy(), grid, side="right") - 1
    values = np.where(positions >= 0, cif.to_numpy()[np.clip(positions, 0, None)], 0)
    return pd.DataFrame({
        "time_years": grid,
        "cif_pct": values * 100,
        "n_at_risk": len(t) - np.searchsorted(np.sort(t), grid, side="left"),
        "n_events_cum": np.searchsorted(np.sort(t[ev == 1]), grid, side="right"),
    })
