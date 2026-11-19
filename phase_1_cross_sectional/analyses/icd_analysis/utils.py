"""Shared helpers for the ICD-analysis PheWAS pipeline.

Data-safety: nothing here prints row-level identifiers. Callers must only emit
de-identified aggregates (counts, distributions) to stdout/reports.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

# ---- canonical project locations -------------------------------------------
PROJECT = Path(r"C:\Users\natha\Projects\work\phewas")
DATA = Path(r"D:\phewas_data")
RESULTS = PROJECT / "results" / "icd_analysis_results"
COHORT_DIR = RESULTS / "cohort"
PHENO_DIR = RESULTS / "phenotype"
GROUP_DIR = RESULTS / "group"
PYPHEWAS_DIR = RESULTS / "pyphewas"
PLOTS_DIR = RESULTS / "plots"
EXPLORER_DIR = RESULTS / "explorer"
SUMMARY_DIR = RESULTS / "summary"

# ---- v2 analysis output tree (v1 preserved) --------------------------------
V2_RESULTS = PROJECT / "results" / "icd_analysis_v2_results"
V2_COHORT = V2_RESULTS / "cohort"
V2_PYPHEWAS = V2_RESULTS / "pyphewas"
V2_PLOTS = V2_RESULTS / "plots"
V2_SUMMARY = V2_RESULTS / "summary"


def normalize_mrn(s: pd.Series) -> pd.Series:
    """MRN -> zero-padded 12-char string (the universal join key).

    Strips whitespace and any trailing '.0' that sneaks in when a numeric MRN
    was parsed as float, then left-pads to 12. Empty/na become <NA>.
    """
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.where(~out.isin(["", "nan", "NaN", "<NA>", "None"]), other=pd.NA)
    return out.str.zfill(12)


def parse_sas_datetime(s: pd.Series) -> pd.Series:
    """Parse SAS 'DDMONYYYY:HH:MM:SS' (and bare 'DDMONYYYY') to datetime64."""
    raw = s.astype("string").str.strip()
    out = pd.to_datetime(raw, format="%d%b%Y:%H:%M:%S", errors="coerce")
    miss = out.isna() & raw.notna()
    if miss.any():  # fall back to date-only SAS form
        out.loc[miss] = pd.to_datetime(raw[miss], format="%d%b%Y", errors="coerce")
    miss = out.isna() & raw.notna()
    if miss.any():  # last resort: let pandas infer
        out.loc[miss] = pd.to_datetime(raw[miss], errors="coerce")
    return out


# AASM adult AHI severity cutoffs
AHI_BINS = [(-0.001, 5, "None", 0), (5, 15, "Mild", 1),
            (15, 30, "Moderate", 2), (30, 1e9, "Severe", 3)]


def ahi_to_severity(ahi: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (severity label, severity_ord 0-3) from AHI using AASM cutoffs."""
    label = pd.Series(pd.NA, index=ahi.index, dtype="object")
    order = pd.Series(pd.NA, index=ahi.index, dtype="Int64")
    for lo, hi, name, ordv in AHI_BINS:
        m = (ahi >= lo) & (ahi < hi)
        label[m] = name
        order[m] = ordv
    return label, order


def log_milestone(message: str) -> None:
    """Append a timestamped milestone line to logs/activity_log.md."""
    log_path = PROJECT / "logs" / "activity_log.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [icd_analysis] {message}\n")
