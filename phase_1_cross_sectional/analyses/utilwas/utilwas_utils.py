"""Shared helpers for the UtilWAS healthcare-utilization pipeline.

Reuses the ICD-analysis helpers (``normalize_mrn``, ``parse_sas_datetime``,
``AHI_BINS``, ``ahi_to_severity``) without modifying that package, and defines
the utilwas output tree, the seven utilization-category filters, a DuckDB
configurator, and a ``[utilwas]``-tagged milestone logger.

Data-safety: nothing here prints row-level identifiers. Callers emit only
de-identified aggregates (counts, distributions) to stdout/reports. Patient-level
exports (the windowed CSVs and the phys/allied overlap file) are written to local
disk only -- never echoed to stdout or returned to a model.
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
RESULTS = PROJECT / "results" / "utilwas_results"
WINDOWED_DIR = RESULTS / "windowed"
ANALYSIS = RESULTS / "analysis"
TABLES = ANALYSIS / "tables"
OVERLAP_DIR = RESULTS / "overlap"

# ---- source data (off-repo, PHI) ------------------------------------------
UTIL_CSV = "D:/phewas_data/utilization.csv"
# The v2 ICD cohort is the analytic spine (severity + index date).
COHORT = (PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()

# Windowed store (diagnosis-line grain, cohort-only, <= 5yr pre-index, + in_1yr flag).
STORE_5YR = (WINDOWED_DIR / "utilization_5yr.parquet").as_posix()
# Requested literal windowed versions of utilization.csv (all original columns).
UTIL_1YR_CSV = (DATA / "utilization_1yr.csv").as_posix()
UTIL_5YR_CSV = (DATA / "utilization_5yr.csv").as_posix()

# SAS datetime format on ADATE/DDATE ("DDMONYYYY:HH:MM:SS").
SAS = "%d%b%Y:%H:%M:%S"

# read_csv options for utilization.csv: the file uses RFC-4180 doubled-quote escaping
# (e.g. '"""CYSTOCELE, MIDLINE"""'); DuckDB auto-detect wrongly guesses backslash escape and
# dies mid-file, so force quote/escape explicitly. ALL_VARCHAR skips type sniffing.
CSV_READ = "ALL_VARCHAR=TRUE, quote='\"', escape='\"'"

# In-SQL MRN normalizer (mirrors normalize_mrn): zero-pad to 12 chars, dropping a
# trailing '.0' left by float parsing. Never CAST(MRN AS INT).
MRN_SQL = r"lpad(regexp_replace(CAST(MRN AS VARCHAR), '\.0$', ''), 12, '0')"

# Two pre-index windows on encounter date (ADATE). [idx - N yr, idx).
WINDOWS = [("1yr", 1), ("5yr", 5)]

# Severity strata (reference group 'None'); display order.
SEVERITIES = ["None", "Mild", "Moderate", "Severe"]

# ---- the utilization-category value sets -----------------------------------
OUTPATIENT_CARE_TYPES = ["CLINIC", "VIRTUAL"]
OUTPATIENT_SUBTYPES = ["OFFICE VISIT", "TAV", "VIDEO"]
ICD_CODE_SUBTYPES = ["ICD9", "ICD10"]
MESSAGES_SUBTYPES = ["PATIENT MESSAGE", "PHONE"]   # NB: these live under CARE_TYPE='VIRTUAL'
ED_SUBTYPES = ["ED", "ED-IP", "ED-OB", "ED-OB-IP", "ED-OB-SU",
               "ED-OP", "ED-OP-IP", "ED-SU", "ED-SU-IP"]
INPATIENT_SUBTYPES = ["ED-IP", "ED-OB", "ED-OB-IP", "ED-OB-SU",
                      "ED-OP", "ED-OP-IP", "ED-SU", "ED-SU-IP", "IP"]
URGENT_CARE_SPECIALTY = "Urgent Care"
OTHER_CARE_TYPE = "OTHER"
SNF_SUBTYPE = "SNF"


def sql_in(vals) -> str:
    """Render a python list as a SQL IN-list body: ``'a','b',...`` (single-quote safe)."""
    return ",".join("'" + str(v).replace("'", "''") + "'" for v in vals)


def duckdb_tmp(script_file: str) -> str:
    """Posix path to a per-run DuckDB spill dir next to the calling script."""
    p = Path(script_file).resolve().parent / "_duckdb_tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def configure_duckdb(con, script_file: str, memory_gb: int = 96, threads: int = 12) -> None:
    """Configure DuckDB to exploit large RAM (~192 GB host): stay in memory,
    spill to a local temp dir only as a fallback."""
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{duckdb_tmp(script_file)}'")


def ensure_dirs() -> None:
    for d in (RESULTS, WINDOWED_DIR, ANALYSIS, TABLES, OVERLAP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def log_milestone(message: str) -> None:
    """Append a timestamped milestone to logs/activity_log.md with a [utilwas] tag."""
    log_path = PROJECT / "logs" / "activity_log.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [utilwas] {message}\n")
