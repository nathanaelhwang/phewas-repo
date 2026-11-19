"""Shared helpers for the LabWAS lab-feature pipeline.

Reuses the ICD-analysis helpers (``normalize_mrn``, ``parse_sas_datetime``)
without modifying that package, and defines the labwas output tree plus a
``[labwas]``-tagged milestone logger.

Data-safety: nothing here prints row-level identifiers. Callers must emit only
de-identified aggregates (counts, distributions) to stdout/reports.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# ---- reuse the icd_analysis helpers without touching that package ----------
_ICD = Path(__file__).resolve().parent.parent / "icd_analysis"
sys.path.insert(0, str(_ICD))
import utils as _U  # noqa: E402

normalize_mrn = _U.normalize_mrn
parse_sas_datetime = _U.parse_sas_datetime

# ---- canonical project locations -------------------------------------------
PROJECT = _U.PROJECT
DATA = _U.DATA
ICD_RESULTS = _U.RESULTS  # to optionally align to the curated OSA cohort
RESULTS = PROJECT / "results" / "labwas_results"
COHORT_DIR = RESULTS / "cohort"
LABS_DIR = RESULTS / "labs"
CROSSWALK_DIR = RESULTS / "crosswalk"

# ---- source data (off-repo) ------------------------------------------------
POP_COHORT = "D:/phewas_data/PopulationCohort251103.csv"
LOINC_CSV = "D:/phewas_data/Loinc_2.82/LoincTable/Loinc.csv"

# SAS datetime format used across this EHR extract ("DDMONYYYY:HH:MM:SS").
SAS = "%d%b%Y:%H:%M:%S"

# DuckDB in-SQL MRN normalizer (mirrors normalize_mrn): zero-pad to 12 chars,
# dropping a trailing '.0' left by float parsing. Never CAST(MRN AS INT).
MRN_SQL = r"lpad(regexp_replace(CAST(MRN AS VARCHAR), '\.0$', ''), 12, '0')"

# Earliest plausible sleep-study year (pre-EHR dates are implausible).
MIN_STUDY_YEAR = 2010

# QualityLab-style dual inclusion threshold, applied per window on LOINC-level
# counts: an analyte is kept only if it has >= MIN_PATIENTS distinct patients
# AND >= MIN_OBS total observations in that window.
MIN_PATIENTS = 500
MIN_OBS = 1000


def duckdb_tmp(script_file: str) -> str:
    """Posix path to a per-run DuckDB spill dir next to the calling script."""
    p = Path(script_file).resolve().parent / "_duckdb_tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def ensure_dirs() -> None:
    for d in (COHORT_DIR, LABS_DIR, CROSSWALK_DIR):
        d.mkdir(parents=True, exist_ok=True)


def log_milestone(message: str) -> None:
    """Append a timestamped milestone to logs/activity_log.md with a [labwas] tag."""
    log_path = PROJECT / "logs" / "activity_log.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [labwas] {message}\n")
