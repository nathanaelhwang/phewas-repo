"""Shared helpers for the BehWAS behavioral-outcome pipeline.

BehWAS is the final prevalence stage of the OSA PheWAS. Unlike the other scans it
is NOT a phenome-wide sweep -- it is a small, curated forest plot of five behavioral
outcomes (medication adherence, no-show rate, flu shot, off-work orders, FMLA)
tested against OSA severity, reusing the exact same exposure / windows / adjustment
models as LabWAS / MedWAS / UtilWAS.

Reuses the ICD-analysis helpers (``normalize_mrn``, ``parse_sas_datetime``,
``AHI_BINS``, ``ahi_to_severity``) without modifying that package, and defines the
behwas output tree, source-data locations, a DuckDB configurator, and a
``[behwas]``-tagged milestone logger.

Data-safety: nothing here prints row-level identifiers. Callers must emit only
de-identified aggregates (counts, distributions) to stdout / reports. The source
files are real EHR/PHI on local disk -- fmla_380.csv in particular is read with an
explicit column allow-list so its free-text / identifier columns are never loaded.
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
AHI_BINS = _U.AHI_BINS
ahi_to_severity = _U.ahi_to_severity

# ---- canonical project locations -------------------------------------------
PROJECT = _U.PROJECT
DATA = _U.DATA
RESULTS = PROJECT / "results" / "behwas_results"
FEATURES_DIR = RESULTS / "features"
ANALYSIS = RESULTS / "analysis"
PLOTS = ANALYSIS / "plots"
TABLES = ANALYSIS / "tables"

# The v2 ICD cohort is the analytic spine (severity + covariates + index date).
COHORT = (PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()

# ---- source data (off-repo, PHI) ------------------------------------------
# NB: column names below come from results/data_preparation/data_dictionary.md;
# verify against the live header on first run.
POPCOHORT_CSV = "D:/phewas_data/PopulationCohort251103.csv"   # ~101 GB, cartesian-exploded
FMLA_CSV = DATA / "fmla_380.csv"                               # latin-1, WH-380 certifications
ADH_CSV = DATA / "medication_adherence.csv"                   # pre-computed all-class PDC
MED_CSV = "D:/phewas_data/medication.csv"                      # raw fills (step 07 comparison only)
# MedWAS's de-exploded distinct fills (has fill_date + rxsup) -- reused by step 07.
FILLS_PARQUET = (PROJECT / "results" / "medwas_results" / "fills" / "distinct_fills.parquet")

# SAS datetime format used across this EHR extract ("DDMONYYYY:HH:MM:SS"),
# plus the bare date variant.
SAS = "%d%b%Y:%H:%M:%S"
SAS_DATE = "%d%b%Y"

# In-SQL MRN normalizer (mirrors normalize_mrn): zero-pad to 12 chars, dropping a
# trailing '.0' left by float parsing. Never CAST(MRN AS INT).
MRN_SQL = r"lpad(regexp_replace(CAST(MRN AS VARCHAR), '\.0$', ''), 12, '0')"

# read_csv options for the giant / dirty PHI CSVs (RFC-4180 doubled-quote escaping;
# ALL_VARCHAR skips type sniffing; ignore_errors survives stray malformed lines).
CSV_READ = "header=true, quote='\"', escape='\"', all_varchar=true, ignore_errors=true"

# Two pre-index windows on the feature's own event date. [idx - N yr, idx).
WINDOWS = [("1yr", 1), ("5yr", 5)]

# Severity strata (reference group 'None'); display order.
SEVERITIES = ["None", "Mild", "Moderate", "Severe"]

# Adherence dichotomization threshold (standard PDC >= 0.80 = "adherent").
ADHERENT_PDC = 0.80

# PROB_COND_DUR_UNITS -> days multiplier (1=days, 2=weeks, 3=months; 30-day month).
DUR_UNIT_DAYS = {1: 1, 2: 7, 3: 30}


def duckdb_tmp(script_file: str) -> str:
    """Posix path to a per-run DuckDB spill dir next to the calling script."""
    p = Path(script_file).resolve().parent / "_duckdb_tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def configure_duckdb(con, script_file: str, memory_gb: int = 150, threads: int = 16) -> None:
    """Configure DuckDB to exploit large RAM (~192 GB host): stay in memory,
    spill to a local temp dir only as a fallback."""
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{duckdb_tmp(script_file)}'")


def ensure_dirs() -> None:
    for d in (RESULTS, FEATURES_DIR, ANALYSIS, PLOTS, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def log_milestone(message: str) -> None:
    """Append a timestamped milestone to logs/activity_log.md with a [behwas] tag."""
    log_path = PROJECT / "logs" / "activity_log.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [behwas] {message}\n")
