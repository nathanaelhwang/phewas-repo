"""Shared helpers for the incwas (incidence PheWAS / "PheDAS incidence") pipeline.

A time-to-event PheWAS: for each phecode, Cox regression of OSA severity -> incident
onset among patients free of that phecode at index (STUDY_DATE), censoring at the
earliest of {event, death, end of insurance coverage, diagnosis cutoff 2023-05-31}.

Reuses the utilwas generic SQL/DuckDB helpers (which themselves reuse the icd_analysis
helpers) without modifying those packages, and defines the incwas output tree, source
paths, and analysis constants.

Data-safety: nothing here prints row-level identifiers. Callers emit only de-identified
aggregates (counts, person-time, rates, effect sizes). Patient-level parquet stores land
on local disk only -- never echoed to stdout or returned to a model.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys
from pathlib import Path

# ---- reuse the utilwas generic helpers (MRN_SQL, SAS, CSV_READ, DuckDB config, ...) --
_UW_DIR = Path(__file__).resolve().parent.parent / "utilwas"
sys.path.insert(0, str(_UW_DIR))
import utilwas_utils as _UW  # noqa: E402

normalize_mrn = _UW.normalize_mrn
parse_sas_datetime = _UW.parse_sas_datetime
AHI_BINS = _UW.AHI_BINS
ahi_to_severity = _UW.ahi_to_severity
MRN_SQL = _UW.MRN_SQL
SAS = _UW.SAS
CSV_READ = _UW.CSV_READ
sql_in = _UW.sql_in
duckdb_tmp = _UW.duckdb_tmp
configure_duckdb = _UW.configure_duckdb

PROJECT = _UW.PROJECT
DATA = _UW.DATA

# ---- incwas output tree ----------------------------------------------------
RESULTS = PROJECT / "results" / "incwas_results"
FOLLOWUP = RESULTS / "followup"
EVENT_STORE = RESULTS / "event_store"
ANALYSIS = RESULTS / "analysis"
SUMMARY = RESULTS / "summary"
PLOTS = RESULTS / "plots"
PILOT_PLOTS = PLOTS / "pilot"

# ---- source data (off-repo, PHI) ------------------------------------------
UTIL_CSV = "D:/phewas_data/utilization.csv"       # 104 GB dated phenome (ADATE <= 2023-05-31)
INS_SAS = "D:/sas_data/ins.sas7bdat"              # insurance coverage spans (MRN, STRT, STOP)
MEMBER = "D:/phewas_data/member.csv"              # DOD / DECEASED for death censoring
# v2 ICD cohort spine (index date + severity + all m1-m4 covariates); now under 'archived results'.
COHORT = (PROJECT / "results" / "archived results" / "icd_analysis_v2_results"
          / "cohort" / "cohort.parquet").as_posix()

# ---- derived outputs -------------------------------------------------------
INS_COVERAGE = (FOLLOWUP / "insurance_coverage.parquet").as_posix()
FOLLOWUP_FRAME = (FOLLOWUP / "followup_frame.parquet").as_posix()
EVENTS = (EVENT_STORE / "phecode_events.parquet").as_posix()
PHECODE_DIM = (EVENT_STORE / "phecode_dim.csv").as_posix()

# ---- analysis constants ----------------------------------------------------
ADATE_CUTOFF = "2023-05-31"      # administrative diagnosis-outcome cutoff (binding follow-up limit)
INS_SENTINEL = "4000-12-30"      # open-ended "still enrolled" STOP value in ins.sas7bdat
MIN_EVENTS = 20                  # keep-phecode floor: minimum incident events to fit
EPV = 10                         # events-per-variable floor (model back-off below this)
HORIZONS = [1, 2, 3]             # incidence-rate bucket years
SEVERITIES = ["None", "Mild", "Moderate", "Severe"]

# pyPheWAS v1.2 ICD->phecode maps (ship inside the installed package)
_res = os.path.join(os.path.dirname(importlib.util.find_spec("pyPheWAS").origin), "resources")
ICD9_MAP = os.path.join(_res, "phecode_map_v1_2_icd9.csv").replace("\\", "/")
ICD10_MAP = os.path.join(_res, "phecode_map_v1_2_icd10_beta.csv").replace("\\", "/")
ICD_CODE_SUBTYPES = ["ICD9", "ICD10"]

# Pilot: ~20 phecodes (v1.2) blending prevalence-PheWAS strength + a-priori OSA sequelae.
# Anchors that must replicate; plus direction-recovery tests (protective/null cross-sectionally).
# NOTE: post-index diagnoses are ICD-10 only, so incident signal lives on the ICD-10-active phecode
# (e.g. depression -> 296.22 not 296.2; AFib -> 427.2 not 427.21; stroke -> 433.2 not 433.21).
PILOT_PHECODES = [
    "401.1",   # essential hypertension
    "250.2",   # type 2 diabetes
    "278.1",   # obesity
    "272.1",   # hyperlipidemia
    "428.2",   # heart failure NOS
    "402",     # elevated BP w/o HTN dx
    "274.1",   # gout
    "401.22",  # hypertensive chronic kidney disease
    "401.21",  # hypertensive heart disease
    "585.3",   # chronic renal failure [CKD]
    "585.1",   # acute renal failure
    "250.23",  # type 2 diabetes with ophthalmic manifestations (ICD-10 diabetic retinopathy)
    "425.1",   # primary/intrinsic cardiomyopathy
    "427.2",   # atrial fibrillation/flutter   (direction-recovery)
    "296.22",  # major depressive disorder     (direction-recovery)
    "530.11",  # GERD                          (direction-recovery)
    "411.4",   # coronary atherosclerosis      (direction-recovery)
    "327.4",   # insomnia                      (direction-recovery)
    "415.21",  # primary pulmonary hypertension
    "433.2",   # occlusion of cerebral arteries [stroke]
]


def ensure_dirs() -> None:
    for d in (RESULTS, FOLLOWUP, EVENT_STORE, ANALYSIS, SUMMARY, PLOTS, PILOT_PLOTS):
        d.mkdir(parents=True, exist_ok=True)


def phecode_map_sql() -> str:
    """SQL that builds a TEMP TABLE `map(cst, code, phecode, phenotype, category)` from the
    pyPheWAS v1.2 icd9 + icd10_beta maps, deduped on (code-system, dotted code). Identical to
    the mapper reused across utilwas -- keep in one place so incwas stays consistent."""
    return f"""
        CREATE TEMP TABLE map AS
        SELECT cst, code, any_value(phecode) phecode, any_value(phenotype) phenotype,
               any_value(category) category
        FROM (
          SELECT 'ICD9' cst, upper(trim("ICD9")) code, CAST("PheCode" AS VARCHAR) phecode,
                 "Phenotype" phenotype, category_string category
          FROM read_csv_auto('{ICD9_MAP}', ALL_VARCHAR=TRUE) WHERE "ICD9" IS NOT NULL
          UNION ALL
          SELECT 'ICD10', upper(trim("ICD10")), CAST("PheCode" AS VARCHAR), "Phenotype", category_string
          FROM read_csv_auto('{ICD10_MAP}', ALL_VARCHAR=TRUE) WHERE "ICD10" IS NOT NULL
        ) GROUP BY cst, code
    """


def log_milestone(message: str) -> None:
    """Append a timestamped milestone to logs/activity_log.md with an [incwas] tag."""
    log_path = PROJECT / "logs" / "activity_log.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [incwas] {message}\n")
