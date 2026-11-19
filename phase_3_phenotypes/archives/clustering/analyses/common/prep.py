"""Cleaning, feature-engineering and standardization shared across the clustering groups.

Cleaning turns only *enumerated corruption* into NaN (inf, INT32 sentinel, impossible
physiologic values) while preserving the large legitimate zero mass. Standardization is a
per-feature z-score fit on observed (non-NaN) values so every feature enters PPCA on a
comparable (correlation) scale.

c1 (physiologic) is the only group that needs clinical range-cleaning; c2 (ordinal/binary
questionnaire) and c3 (binary phecodes) are bounded by construction and use the Standardizer
directly. Pure functions, no I/O, no row-level printing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- c1 physiologic feature sets -------------------------------------------------------
# 11 physiologic features that go straight into the model (body position excluded: those
# metrics are ~100% missing on lab-PSG devices and previously formed a device artifact).
PHYSIO_11: list[str] = [
    "ODI4PCT", "SPOT_LT90", "T90%", "SPOT_LT80", "T80%", "AVERAGE_DESATURATION",
    "MEAN_SPO2", "MINIMUM_SPO2", "MEAN_PULSE_RATE", "MINIMUM_PULSE_RATE", "MAXIMUM_PULSE_RATE",
]
# raw columns consumed only to engineer ratio features / define the cohort (dropped from inputs).
RATIO_INPUTS: list[str] = ["AHI", "AHI_SUPINE", "AHI_NONSUPINE", "SUPINE_DURATION_PERCENT"]
ENGINEERED: list[str] = ["T90_over_AHI", "supine_nonsupine_AHI_ratio"]
C1_FEATS: list[str] = PHYSIO_11 + ENGINEERED                      # 13 model features

OBJECT_NUMERIC = ["MEAN_PULSE_RATE", "MAXIMUM_PULSE_RATE"]        # come in as object dtype
SPO2_COLS = ["MEAN_SPO2", "MINIMUM_SPO2"]
PULSE_COLS = ["MEAN_PULSE_RATE", "MINIMUM_PULSE_RATE", "MAXIMUM_PULSE_RATE"]
PCT_BOUNDED = ["T90%", "T80%", "SUPINE_DURATION_PERCENT"]        # 0..100 percentages
INT32_MIN = -2147483648

# skewed non-negative c1 features that benefit from a log1p before standardization.
C1_LOG1P = ["ODI4PCT", "SPOT_LT90", "T90%", "SPOT_LT80", "T80%",
            "T90_over_AHI", "supine_nonsupine_AHI_ratio"]


def clean_c1_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw physiologic + ratio-input columns to NaN-where-corrupt (in place on a copy).

    Enumerated rules only: inf, <=-1e9 / INT32 sentinel, impossible negatives, SpO2 outside
    (0,100], pulse outside (0,300], desaturation>100, percentages clipped at 100 (>100.5 -> NaN).
    Legitimate zeros in non-SpO2/pulse features are protected and asserted unchanged.
    """
    out = df.copy()
    for c in OBJECT_NUMERIC:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    feats = [c for c in PHYSIO_11 + RATIO_INPUTS if c in out.columns]
    X = out[feats].astype("float64")
    report: dict = {"per_rule": {}, "zero_protection": {}}
    raw_zero = {c: int((X[c] == 0).sum()) for c in feats}

    report["per_rule"]["inf_to_nan"] = int(np.isinf(X.to_numpy()).sum())
    X = X.replace([np.inf, -np.inf], np.nan)

    sentinel = (X <= -1e9) | (X == INT32_MIN)
    report["per_rule"]["sentinel_to_nan"] = int(sentinel.to_numpy().sum())
    X = X.mask(sentinel)

    neg = X < 0
    report["per_rule"]["negative_to_nan"] = int(neg.to_numpy().sum())
    X = X.mask(neg)

    n = 0
    for c in [c for c in SPO2_COLS if c in feats]:
        bad = (X[c] > 100) | (X[c] == 0); n += int(bad.sum()); X[c] = X[c].mask(bad)
    report["per_rule"]["spo2_impossible_to_nan"] = n

    if "AVERAGE_DESATURATION" in feats:
        bad = X["AVERAGE_DESATURATION"] > 100
        report["per_rule"]["avg_desat_gt100_to_nan"] = int(bad.sum())
        X["AVERAGE_DESATURATION"] = X["AVERAGE_DESATURATION"].mask(bad)

    n = 0                                          # heart rate physiologic floor/ceiling 20..300 bpm
    for c in [c for c in PULSE_COLS if c in feats]:
        bad = (X[c] > 300) | (X[c] < 20); n += int(bad.sum()); X[c] = X[c].mask(bad)
    report["per_rule"]["pulse_impossible_to_nan"] = n

    n_clip = n_nan = 0
    for c in [c for c in PCT_BOUNDED if c in feats]:
        overshoot = (X[c] > 100) & (X[c] <= 100.5); over = X[c] > 100.5
        n_clip += int(overshoot.sum()); n_nan += int(over.sum())
        X.loc[overshoot, c] = 100.0; X[c] = X[c].mask(over)
    report["per_rule"]["pct_overshoot_clip_to_100"] = n_clip
    report["per_rule"]["pct_gt100p5_to_nan"] = n_nan

    protected = [c for c in feats if c not in SPO2_COLS + PULSE_COLS]
    for c in protected:
        now = int((X[c] == 0).sum())
        report["zero_protection"][c] = {"raw": raw_zero[c], "post_clean": now, "ok": now == raw_zero[c]}
    bad_cols = [c for c, v in report["zero_protection"].items() if not v["ok"]]
    if bad_cols:
        raise AssertionError(f"Protected zeros changed during cleaning: {bad_cols}")

    out[feats] = X
    report["missing_after_clean"] = {c: int(X[c].isna().sum()) for c in feats}
    report["n_rows"] = len(out)
    return out, report


def engineer_c1(clean: pd.DataFrame) -> pd.DataFrame:
    """Add the two engineered ratio features to a *cleaned* c1 frame (returns a copy).

    T90_over_AHI                = T90% / AHI                 (NaN where AHI is missing or <= 0)
    supine_nonsupine_AHI_ratio  = AHI_SUPINE / AHI_NONSUPINE, computed ONLY where the supine
                                  fraction is in [25, 75]% and AHI_NONSUPINE > 0 (else NaN).
    """
    out = clean.copy()
    ahi = out["AHI"]
    out["T90_over_AHI"] = np.where(ahi > 0, out["T90%"] / ahi, np.nan)

    sup = out["AHI_SUPINE"]; non = out["AHI_NONSUPINE"]; pct = out["SUPINE_DURATION_PERCENT"]
    valid = pct.between(25, 75) & sup.notna() & non.notna() & (non > 0)
    out["supine_nonsupine_AHI_ratio"] = np.where(valid, sup / non, np.nan)
    return out


class Standardizer:
    """Per-feature z-score fit on observed (non-NaN) cells; NaNs pass through untouched.

    Degenerate features (sd ~ 0) are given sd = 1 so they map to a constant 0 column rather
    than exploding; callers should drop truly constant features before fitting.
    """

    def __init__(self, feats: list[str]):
        self.feats = feats
        self.mean_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "Standardizer":
        A = df[self.feats].to_numpy(dtype=float)
        self.mean_ = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
        self.sd_ = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        A = df[self.feats].to_numpy(dtype=float)
        return (A - self.mean_) / self.sd_

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
