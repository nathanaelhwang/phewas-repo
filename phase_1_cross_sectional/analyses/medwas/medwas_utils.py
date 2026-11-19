"""Shared helpers for the MedWAS medication-class pipeline.

Reuses the ICD-analysis helpers (``normalize_mrn``, ``parse_sas_datetime``,
``AHI_BINS``, ``ahi_to_severity``) without modifying that package, and defines
the medwas output tree, GPI super-heading rollup, a DuckDB configurator that
exploits large RAM, and a ``[medwas]``-tagged milestone logger.

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
AHI_BINS = _U.AHI_BINS
ahi_to_severity = _U.ahi_to_severity

# ---- canonical project locations -------------------------------------------
PROJECT = _U.PROJECT
DATA = _U.DATA
RESULTS = PROJECT / "results" / "medwas_results"
FILLS_DIR = RESULTS / "fills"
CROSSWALK_DIR = RESULTS / "crosswalk"
ANALYSIS = RESULTS / "analysis"

# ---- source data (off-repo, PHI) ------------------------------------------
MED_CSV = "D:/phewas_data/medication.csv"
# The v2 ICD cohort is the analytic spine (severity + covariates + index date).
COHORT = (PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()

# GPI class-name reference (Medi-Span TCS Appendix A), text-extracted from the
# user-supplied PDF. Drug codes / class names only -- not PHI.
GPI_DOC = (Path(
    r"C:\Users\natha\AppData\Local\Temp\claude\C--Users-natha-Projects-work-phewas"
    r"\0ff26d9b-9f38-4623-9ece-67adfe73a5d8\scratchpad\gpi_codes.txt"
))

# SAS datetime format used across this EHR extract ("DDMONYYYY:HH:MM:SS"),
# plus the bare date variant seen on RFDISPDT/RFSOLDDT.
SAS = "%d%b%Y:%H:%M:%S"
SAS_DATE = "%d%b%Y"

# DuckDB in-SQL MRN normalizer (mirrors normalize_mrn): zero-pad to 12 chars,
# dropping a trailing '.0' left by float parsing. Never CAST(MRN AS INT).
MRN_SQL = r"lpad(regexp_replace(CAST(MRN AS VARCHAR), '\.0$', ''), 12, '0')"

# Earliest plausible study year (guards against pre-EHR junk dates).
MIN_STUDY_YEAR = 2010

# Prevalence floor: a GPI-4 class is tested only if >= MIN_PATIENTS distinct
# cohort patients filled it within the window (binary outcome -> no obs floor).
MIN_PATIENTS = 500

# Two pre-index windows, on fill date. Standardized labels (no 5yr/5yrs split).
WINDOWS = [("1yr", 1), ("5yr", 5)]

# The 17 Medi-Span TCS super-headings, as inclusive 2-digit Drug-Group ranges
# (from Appendix A). gpi2 -> super-heading is a fixed table, independent of the
# PDF text parse, so it is always populated.
SUPER_RANGES = [
    (1, 16, "ANTI-INFECTIVE AGENTS"), (17, 20, "BIOLOGICALS"),
    (21, 21, "ANTINEOPLASTIC AGENTS"), (22, 30, "ENDOCRINE & METABOLIC"),
    (31, 40, "CARDIOVASCULAR"), (41, 45, "RESPIRATORY"),
    (46, 52, "GASTROINTESTINAL"), (53, 56, "GENITOURINARY"),
    (57, 60, "CNS"), (61, 61, "STIMULANTS/ANTI-OBESITY"),
    (62, 63, "MISC PSYCH/NEURO"), (64, 71, "ANALGESICS & ANESTHETICS"),
    (72, 76, "NEUROMUSCULAR"), (77, 81, "NUTRITIONAL"),
    (82, 85, "HEMATOLOGICAL"), (86, 91, "TOPICAL"),
    (92, 99, "MISCELLANEOUS"),
]


def super_heading(gpi2: str | int | None) -> str:
    """Map a 2-digit GPI Drug Group to its TCS super-heading."""
    if gpi2 is None:
        return "unclassified"
    try:
        g = int(gpi2)
    except (TypeError, ValueError):
        return "unclassified"
    for lo, hi, name in SUPER_RANGES:
        if lo <= g <= hi:
            return name
    return "unclassified"


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
    for d in (RESULTS, FILLS_DIR, CROSSWALK_DIR, ANALYSIS):
        d.mkdir(parents=True, exist_ok=True)


def log_milestone(message: str) -> None:
    """Append a timestamped milestone to logs/activity_log.md with a [medwas] tag."""
    log_path = PROJECT / "logs" / "activity_log.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [medwas] {message}\n")
