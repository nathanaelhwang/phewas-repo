"""Shared helpers for the ProcWAS procedure-frequency association stage.

Outcome: per-patient procedures/yr for each 'Clear descriptor' (a Poisson rate WAS),
tested vs OSA severity over the 222,109 v2 cohort (non-performers = structural 0).
Reuses the utilwas rate engine (_was_engine.py) and the v2 cohort spine; the only
new piece is the m2/m3/m4 covariate ladder. Data is real EHR/PHI -> de-identified
aggregates only; the windowed store + target parquets land on local disk.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# reuse the utilwas rate engine + the icd_analysis canonical locations
_UTIL = Path(__file__).resolve().parent.parent / "utilwas"
sys.path.insert(0, str(_UTIL))
import utilwas_utils as _UU  # noqa: E402

PROJECT = _UU.PROJECT
DATA = _UU.DATA
COHORT = _UU.COHORT                      # results/icd_analysis_v2_results/cohort/cohort.parquet
WINDOWS = _UU.WINDOWS                    # [("1yr",1), ("5yr",5)]
SEVERITIES = _UU.SEVERITIES

PROC = (DATA / "proc.csv").as_posix()
XW = DATA / "procedures_classified_v4.xlsx"

RESULTS = PROJECT / "results" / "procwas_results"
WINDOWED_DIR = RESULTS / "windowed"
ANALYSIS = RESULTS / "analysis"
PLOTS = ANALYSIS / "plots"
TABLES = ANALYSIS / "tables"
STORE_5YR = (WINDOWED_DIR / "procedures_5yr.parquet").as_posix()

MIN_PATIENTS = 500                       # per-window patient floor for a testable descriptor

# Sleep-medicine / OSA-pathway procedures EXCLUDED from the WAS + totals: they are circular with
# the OSA exposure (the diagnostic study, titration, PAP therapy). Matches PSG, home sleep tests,
# MSLT, CPAP — but NOT 'Pap smear' or a plain awake+asleep EEG (a neurology seizure-workup test).
SLEEP_TERMS = "polysomn|sleep study|sleep apnea test|home sleep test|multiple sleep latency|cpap"
SLEEP_SQL = f"COALESCE(regexp_matches(lower(clear_descriptor), '{SLEEP_TERMS}'), FALSE)"

# proc.csv read settings + the b'...'/b"..." strip macro (see 01_build_procwas_data.py).
CSV_READ = _UU.CSV_READ                  # ALL_VARCHAR=TRUE, quote='"', escape='"'
STRIP_B = r"""CREATE MACRO strip_b(x) AS regexp_replace(x, '^b[''"](.*)[''"]$', '\1')"""
# canonical 12-char MRN join key straight off the b'-wrapped proc MRN.
PROC_ID_SQL = "lpad(strip_b(MRN), 12, '0')"

configure_duckdb = _UU.configure_duckdb


def ensure_dirs() -> None:
    for d in (RESULTS, WINDOWED_DIR, ANALYSIS, PLOTS, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def crosswalk() -> "any":
    """De-duplicated procedure crosswalk keyed on UPPER('CPT primary').

    One row per key = the dominant procedure (max 'N performed'); carries the
    display name (Clear descriptor) + both broader categories used for plots.
    """
    import pandas as pd
    xw = pd.read_excel(XW, sheet_name="Procedures")
    xw["key_u"] = xw["CPT primary"].astype("string").str.strip().str.upper()
    xw = xw[xw["key_u"].notna() & (xw["key_u"] != "")]
    xw = xw.sort_values("N performed", ascending=False).drop_duplicates("key_u", keep="first")
    return pd.DataFrame({
        "key_u": xw["key_u"].astype(str).to_numpy(),
        "clear_descriptor": xw["Clear descriptor"].astype("string").to_numpy(),
        "procedure_type": xw["Procedure type"].astype("string").to_numpy(),
        "body_system": xw["Body system"].astype("string").to_numpy(),
    })


def log_milestone(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROJECT / "logs" / "activity_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [procwas] {message}\n")
