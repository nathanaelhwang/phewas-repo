"""Shared helpers for MeasWAS - the cardiopulmonary MEASUREMENT arm of the ProcWAS stage.

ProcWAS proper asks how OFTEN a procedure happens (Poisson procedures/yr). This arm asks
what the procedure MEASURED: echocardiogram parameters (echo.csv + ef.csv) and spirometry
values (pft.csv) vs OSA severity, over the same 222,109 v2 cohort spine and the same
5-year pre-index window. Continuous outcomes -> rank-inverse-normal + OLS, mirroring
analyses/labwas/10_run_labwas.py.

EXPLORATORY: PFT covers only ~3.6% of the cohort in-window (pft.csv ends 2019-03 while
index dates run to 2025-03). Echo reaches ~14%, ef.csv ~21%. See SUMMARY.md.

Data is real EHR/PHI -> de-identified aggregates only; intermediate stores stay on local disk.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import sys
from pathlib import Path

# reuse the utilwas rate engine + the icd_analysis canonical locations (same as procwas_utils)
_PROCWAS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROCWAS))
import procwas_utils as _PU  # noqa: E402

PROJECT = _PU.PROJECT
DATA = _PU.DATA
COHORT = _PU.COHORT                      # results/icd_analysis_v2_results/cohort/cohort.parquet
SEVERITIES = _PU.SEVERITIES

ECHO_SRC = (DATA / "echo.csv").as_posix()
EF_SRC = (DATA / "ef.csv").as_posix()
PFT_SRC = (DATA / "pft.csv").as_posix()

RESULTS = PROJECT / "results" / "procwas_results" / "measwas"
STORE = RESULTS / "store"
ANALYSIS = RESULTS / "analysis"
PLOTS = ANALYSIS / "plots"
TABLES = ANALYSIS / "tables"

ECHO_CLEAN = (STORE / "echo_clean.parquet").as_posix()     # 00: repaired echo, PHI dropped
PFT_FEAT = (STORE / "pft_5yr.parquet").as_posix()          # 01: per-patient spirometry
ECHO_FEAT = (STORE / "echo_5yr.parquet").as_posix()        # 02: per-patient echo + EF

WINDOW_YEARS = 5                         # single window; the 1yr window is not viable (PFT 1.1%)
WINDOW_LABEL = "5yr"
MIN_PATIENTS = 500                       # same testability floor as ProcWAS

configure_duckdb = _PU.configure_duckdb


def ensure_dirs() -> None:
    for d in (RESULTS, STORE, ANALYSIS, PLOTS, TABLES):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# echo.csv reader
# --------------------------------------------------------------------------------------
# echo.csv is NOT corrupted (the 2026-06-27 data-dictionary verdict is wrong) but it is not
# RFC 4180 either: the exporter quotes a field only when it contains a comma or a quote, NOT
# when it contains a newline. So the RECORD terminator is a bare LF while newlines embedded in
# free-text report bodies are CRLF (2.4M of them) or a lone CR (74). Split records on bare LF,
# re-join the 28 segments that do not start with the MRN,procedure_id signature, neutralise the
# embedded newlines, and the DEFAULT csv dialect parses all 284,031 records at 228 fields each.
#
# Do NOT use pandas.read_csv on the raw file: it silently returns 926,702 NaN-padded junk rows
# with no warning under every engine/on_bad_lines setting. Do NOT use duckdb ignore_errors=true:
# the 61.6% it keeps is confounded by facility (46-80% retention) and year.
ECHO_N_FIELDS = 228
_BARE_LF = re.compile(rb"(?<!\r)\n")
_RECORD_START = re.compile(rb"^\d{12},\d+,")

# Direct identifiers + free-text report bodies. Dropped at read time; never materialised.
ECHO_PHI_COLS = frozenset({
    "accession_number", "patient_medical_id_ci_txt", "Procedure_Summary", "patient_id",
    "patient_tcs_id", "patient_name", "unique_id_str", "patient_first_name", "patient_last_name",
    "patient_DOB", "DOB", "referring_person_name", "performing_person_name",
    "admitting_person_name", "procedure_step_description", "dicom_study_instance_uid",
    "reserved_text1", "reserved_text2", "reserved_text3", "procedure_recommendations",
    "procedure_addendum",
})

csv.field_size_limit(10 ** 9)


def echo_records(path: str = ECHO_SRC, chunk: int = 1 << 24):
    """Yield one raw echo record (bytes) at a time, at constant memory."""
    buf = b""
    pending = None
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            buf += block
            segs = _BARE_LF.split(buf)
            buf = segs.pop()                                  # incomplete tail
            for s in segs:
                if _RECORD_START.match(s) or pending is None:
                    if pending is not None:
                        yield pending
                    pending = s
                else:
                    pending += b"\r\n" + s                     # bare LF inside free text
    if buf:
        if _RECORD_START.match(buf) or pending is None:
            if pending is not None:
                yield pending
            pending = buf
        else:
            pending += b"\r\n" + buf
    if pending is not None:
        yield pending


def echo_fields(seg: bytes) -> list[str]:
    """Neutralise embedded newlines so they cannot terminate the record, then csv-parse."""
    line = seg.replace(b"\r\n", b"\x1f").replace(b"\r", b"\x1e").decode("latin-1")
    row = next(csv.reader(io.StringIO(line, newline="")))
    return [f.replace("\x1f", "\n").replace("\x1e", "\n") for f in row]


# --------------------------------------------------------------------------------------
# pft.csv column contract (user spec, 2026-07-30)
# --------------------------------------------------------------------------------------
# Rows with an empty TEST_DATE are dropped (4,008 predicted-value-only shells with zero
# spirometry). Reference values are renamed *_PREDICTED -> *_REF; the *_PREDICTED name is
# reused for the ENGINEERED percent-of-reference ratios below.
#
# NOTE (flagged to user): the engineered ratios are fractions, not percents. The file's own
# PreFEV1pred/PostFEV1pred are 100x these. Multiply by 100 here if the percent scale is wanted.
PFT_RENAME = {"PEF_PREDICTED": "PEF_REF", "FEV1_PREDICTED": "FEV1_REF"}

PFT_DERIVED = {                                    # new name -> (numerator, reference)
    "PEF_PRE_PREDICTED": ("PEF_PRE", "PEF_REF"),
    "PEF_POST_PREDICTED": ("PEF_POST", "PEF_REF"),
    "FEV1_PRE_PREDICTED": ("FEV1_PRE", "FEV1_REF"),
    "FEV1_POST_PREDICTED": ("FEV1_POST", "FEV1_REF"),
}

# The tested phenome - exactly the user's list.
PFT_FEATURES = [
    "FEV1_FVC_pre", "FEV1_FVC_post",
    "FEV1_PRE", "FEV1_POST",
    "FVC_PRE", "FVC_POST",
    "PEF_POST", "PEF_REF",
    "PEF_PRE_PREDICTED", "PEF_POST_PREDICTED",
    "FEV1_PRE_PREDICTED", "FEV1_POST_PREDICTED",
    "FEF25_75_PRE", "FEF25_75_POST",
]

# Retained in the built table as derivation inputs / QC, but NOT tested unless the user says so.
PFT_INPUTS_ONLY = ["PEF_PRE", "FEV1_REF"]

# Physiologic gates applied before aggregation. Values outside -> NaN (not row drops).
# PEF_PRE/PIF_PRE carry a handful of L/min readings in an L/s column (max 327); the POST
# columns are clean. FET100 is seconds, not a volume - it is not in the phenome.
PFT_GATES = {
    "FVC_PRE": (0.1, 10.0), "FVC_POST": (0.1, 10.0),
    "FEV1_PRE": (0.1, 8.0), "FEV1_POST": (0.1, 8.0),
    "FEF25_75_PRE": (0.01, 15.0), "FEF25_75_POST": (0.01, 15.0),
    "PEF_PRE": (0.1, 20.0), "PEF_POST": (0.1, 20.0),
    "PEF_REF": (0.1, 20.0), "FEV1_REF": (0.1, 8.0),
    "FEV1_FVC_pre": (10.0, 100.0), "FEV1_FVC_post": (10.0, 100.0),
}

SAS_MS = "%d%b%Y:%H:%M:%S.%f"            # pft VISITDATE/TEST_DATE + every echo SAS timestamp
US_DATE = "%m/%d/%Y"                     # echo performed_date, ef ECHO_DATE


# --------------------------------------------------------------------------------------
# echo phenome (spec, 2026-07-30) - declared ONCE here; steps 02/03/04 import it
# --------------------------------------------------------------------------------------
# Deterministic x100 unit repair, applied PER VALUE before any gating:
#   v_repaired = v*100 if v < lo   else   v/100 if v >= hi   else   v
# The thresholds sit in empirically empty valleys (valley mass G <= 0.00025 for all four),
# so the rule is deterministic, not probabilistic. Only these four contaminated columns are
# used by the phenome; the other eight contaminated columns are excluded outright.
ECHO_UNIT_REPAIR = {
    "tr_velocity": (10.0, 1000.0),                  # 3.30% m/s + 2.92% cm/s x100 -> cm/s
    "av_peak_velocity": (10.0, 1000.0),             # 4.17% m/s + 1.62% x100     -> cm/s
    "mv_peak_e_wave": (10.0, 1000.0),               # 5.32% m/s                  -> cm/s
    "mv_annulus_septal_e_prime": (1.0, 100.0),      # 8.56% m/s (tissue Doppler) -> cm/s
}

# Inputs gated BEFORE the cube/ratio derivations, because the cube amplifies input error.
ECHO_INPUT_GATES = {
    "lv_diastolic_dimension": (2.5, 8.5),           # LVIDd, cm
    "lv_post_wall_thick": (0.4, 3.0),               # PWTd, cm
    "ivs_thick": (0.4, 3.0),                        # IVSd, cm
}

# Study-level plausibility filter (raw age_performed min is -7). The cohort is adults-only AT
# INDEX, but the window reaches 5 years back, so an echo at 14-17 on a patient who is 18-22 at
# index is legitimate data - an [18,105] gate would discard 127 real studies. Only the negative
# ages are unambiguously bad; 13 is the floor implied by "18+ at index, 5yr look-back".
ECHO_AGE_GATE = (13.0, 105.0)

# Derived features: formula of record. Implemented in 02_build_echo.py; the derived value is
# gated AFTER derivation using the feature's own gate in ECHO_FEATURES.
ECHO_DERIVED = {
    "LVEF_pct": {
        "inputs": ["lv_ef_calculated", "lv_ef_estimated"],
        "formula": "coalesce(lv_ef_calculated, lv_ef_estimated) per study, each gated [10,85]",
    },
    "LV_mass_ASE_g": {
        "inputs": ["lv_diastolic_dimension", "lv_post_wall_thick", "ivs_thick"],
        "formula": ("0.8*1.04*((LVIDd + PWTd + IVSd)**3 - LVIDd**3) + 0.6  [cm -> g; "
                    "ASE/Devereux cube]. The file's lv_mass_penn == 1.04*cube - 13.6 on 99.99% "
                    "of 220,885 rows, i.e. an exact affine map -> lv_mass_penn is dropped."),
    },
    "RWT": {
        "inputs": ["lv_post_wall_thick", "lv_diastolic_dimension"],
        "formula": "2*PWTd/LVIDd  [dimensionless]. No BSA needed -> no era confound.",
    },
    "TR_grad_mmHg": {
        "inputs": ["tr_velocity"],
        "formula": ("4*(repair(tr_velocity)/100)**2  [simplified Bernoulli, cm/s -> mmHg]. "
                    "Do NOT use the stored tr_grad: the exporter always assumes cm/s, so on the "
                    "5,771 m/s rows it has median 0.00223 mmHg."),
    },
    "e_prime_septal_cms": {
        "inputs": ["mv_annulus_septal_e_prime"],
        "formula": "repair(mv_annulus_septal_e_prime)  [cm/s]",
    },
    "E_over_e_prime_septal": {
        "inputs": ["mv_peak_e_wave", "mv_annulus_septal_e_prime"],
        "formula": ("repair(mv_peak_e_wave) / repair(mv_annulus_septal_e_prime). BOTH must be "
                    "repaired: the contamination is not row-aligned (4,569 rows e'-only m/s, 66 "
                    "E-only), so the raw ratio has p99 = 1433 vs 36.7 after repair."),
    },
    "AV_Vmax_cms": {
        "inputs": ["av_peak_velocity"],
        "formula": ("repair(av_peak_velocity)  [cm/s]. NOT av_peak_gradient - it is exactly 0 on "
                    "97.9% of the m/s rows."),
    },
}

# The tested phenome. kind: 'raw' = the gated column as-is, 'derived' = see ECHO_DERIVED,
# 'ef' = from ef.csv (a SEPARATE source from the echo LVEF, deliberately not pooled).
# 'era' records availability at 2010 / 2015 / 2020 as % of that year's echoes: because a
# patient's in-window echoes are drawn from [index-5yr, index), availability is mechanically
# coupled to index year, so study_year adjustment is MANDATORY in every model.
ECHO_FEATURES = [
    {"name": "LVEF_pct", "concept": "Systolic function", "kind": "derived",
     "sources": ["lv_ef_calculated", "lv_ef_estimated"], "gate": (10.0, 85.0),
     "era": "lv_ef_estimated collapses 26.3->4.3; calculated carries it",
     "note": "Echo-side LVEF only. ef.csv is tested separately as ef_value."},
    {"name": "ef_value", "concept": "Systolic function", "kind": "ef",
     "sources": ["ef.csv:EF_VALUE"], "gate": (10.0, 85.0),
     "era": "rising - ef.csv grows 1,418 readings (2000) -> 53,631 (2024)",
     "note": "Independent LVEF source, 211,503 patients 1995-2025. Best-powered feature."},
    {"name": "lv_diastolic_dimension", "concept": "LV chamber size", "kind": "raw",
     "sources": ["lv_diastolic_dimension"], "gate": (2.5, 8.5),
     "era": "STABLE 81.4/78.0/81.8", "note": "LVIDd, cm. ASE normal 3.9-5.9."},
    {"name": "lv_systolic_dimension", "concept": "LV chamber size (systolic)", "kind": "raw",
     "sources": ["lv_systolic_dimension"], "gate": (1.0, 8.0),
     "era": "STABLE 81.6/76.5/79.2", "note": "LVIDs, cm. ASE normal 2.0-3.5."},
    {"name": "LV_mass_ASE_g", "concept": "LV hypertrophy (mass)", "kind": "derived",
     "sources": ["ivs_thick", "lv_post_wall_thick", "lv_diastolic_dimension"],
     "gate": (40.0, 600.0), "era": "STABLE 79.9/75.9/79.5",
     "note": "Central OSA cardiac-remodelling hypothesis. ASE normal 67-224 g."},
    {"name": "RWT", "concept": "LV concentric remodelling", "kind": "derived",
     "sources": ["lv_post_wall_thick", "lv_diastolic_dimension"], "gate": (0.15, 1.2),
     "era": "STABLE 81.0/77.2/80.8", "note": "ASE concentric cut-off 0.42."},
    {"name": "lv_edv_index", "concept": "LV volume (BSA-indexed)", "kind": "raw",
     "sources": ["lv_edv_index"], "gate": (15.0, 250.0),
     "era": "DRIFT rising 0.0/52.3/72.7", "note": "LVEDVi mL/m2. ASE normal 35-75."},
    {"name": "la_systolic_dimension", "concept": "LA size (linear)", "kind": "raw",
     "sources": ["la_systolic_dimension"], "gate": (1.5, 8.0),
     "era": "DRIFT declining 79.0/72.8/57.5", "note": "LAd cm. Normal 2.7-4.0."},
    {"name": "la_volume_index", "concept": "LA size (volumetric, BSA-indexed)", "kind": "raw",
     "sources": ["la_volume_index"], "gate": (5.0, 150.0),
     "era": "DRIFT SEVEREST rising 0.0/25.6/46.7",
     "note": "LAVi mL/m2, ASE normal <=34. Spearman only 0.406 with LAd - not a duplicate."},
    {"name": "TR_grad_mmHg", "concept": "Pulmonary hypertension", "kind": "derived",
     "sources": ["tr_velocity"], "gate": (4.0, 120.0), "era": "mild decline 72.4/60.9/62.4",
     "note": "Headline OSA feature. 4 mmHg = 1.0 m/s jet; 120 = 5.5 m/s, suprasystemic."},
    {"name": "estimated_rap", "concept": "Right atrial pressure / volume status", "kind": "raw",
     "sources": ["estimated_rap"], "gate": (0.0, 25.0),
     "era": "DRIFT declining 62.2/30.9/26.7",
     "note": "QUASI-ORDINAL: 5 values hold 96% of the mass -> RINT will be heavily tied."},
    {"name": "rv_diastolic_dimension", "concept": "RV size / cor pulmonale", "kind": "raw",
     "sources": ["rv_diastolic_dimension"], "gate": (0.8, 6.0),
     "era": "DRIFT declining 38.3/30.5/25.9", "note": "Basal RV normal <=4.1 cm."},
    {"name": "mv_ea_ratio", "concept": "Diastolic function (filling pattern)", "kind": "raw",
     "sources": ["mv_ea_ratio"], "gate": (0.2, 4.0), "era": "STABLE 68.1/66.3/69.6",
     "note": "Use the STORED column: unit contamination cancels within the row (verified "
             "== E/A on 99.96% of 191,964 pairs); do NOT recompute from the dirty E and A."},
    {"name": "e_prime_septal_cms", "concept": "Diastolic function (myocardial relaxation)",
     "kind": "derived", "sources": ["mv_annulus_septal_e_prime"], "gate": (1.5, 25.0),
     "era": "DRIFT rising 43.7/60.5/62.2", "note": "ASE impaired relaxation e' < 7 cm/s."},
    {"name": "E_over_e_prime_septal", "concept": "Diastolic function (LV filling pressure)",
     "kind": "derived", "sources": ["mv_peak_e_wave", "mv_annulus_septal_e_prime"],
     "gate": (2.0, 40.0), "era": "DRIFT rising 42.9/58.5/60.8",
     "note": "ASE elevated-filling-pressure cut-off 15."},
    {"name": "mv_dec_time", "concept": "Diastolic function (deceleration)", "kind": "raw",
     "sources": ["mv_dec_time"], "gate": (60.0, 500.0), "era": "mild rise 57.4/61.1/64.8",
     "note": "ms. <150 restrictive, >240 impaired relaxation."},
    {"name": "ivrt", "concept": "Diastolic function (isovolumic relaxation)", "kind": "raw",
     "sources": ["ivrt"], "gate": (20.0, 250.0), "era": "DRIFT SEVERE rising 1.0/6.3/8.2",
     "note": "ms, normal 50-100. Lowest-powered feature; the designated first cut."},
    {"name": "ao_root", "concept": "Aortic dimension", "kind": "raw",
     "sources": ["ao_root"], "gate": (1.5, 6.0), "era": "mild decline 79.3/75.0/70.4",
     "note": "Sinus of Valsalva, cm. Normal 2.8-3.6."},
    {"name": "ao_ascending", "concept": "Aortic dimension", "kind": "raw",
     "sources": ["ao_ascending"], "gate": (1.5, 7.0), "era": "DRIFT SEVERE rising 0.2/8.2/17.6",
     "note": "cm. Spearman only 0.599 with ao_root - a different segment."},
    {"name": "AV_Vmax_cms", "concept": "Aortic valve (stenosis / sclerosis)", "kind": "derived",
     "sources": ["av_peak_velocity"], "gate": (50.0, 700.0), "era": "DRIFT rising 46.0/70.3/75.9",
     "note": "cm/s. Normal <=170, sclerosis 170-250, severe AS >=400."},
    {"name": "heart_rate_rest", "concept": "Autonomic / haemodynamic state", "kind": "raw",
     "sources": ["heart_rate_rest"], "gate": (30.0, 200.0), "era": "50.4/32.3/38.2",
     "note": "bpm. Also the interpretive covariate for the rate-dependent diastolic indices."},
    {"name": "syst_bp_rest_supine", "concept": "POSITIVE CONTROL - systemic blood pressure",
     "kind": "raw", "sources": ["syst_bp_rest_supine"], "gate": (70.0, 240.0),
     "era": "declining 46.1/29.3/30.9",
     "note": "POSITIVE CONTROL: the OSA-hypertension link is textbook, so a null here indicts "
             "the pipeline. It sits ON the causal path from OSA to every remodelling feature - "
             "it must NEVER enter the covariate ladder for the other features."},
]

ECHO_FEATURE_NAMES = [f["name"] for f in ECHO_FEATURES]
ECHO_GATES = {f["name"]: f["gate"] for f in ECHO_FEATURES}


def echo_repair_units(s, col):
    """Deterministic x100 unit repair for a contaminated echo column (pandas Series in/out).

    v*100 if v < lo, v/100 if v >= hi, else v. NaN compares False on both, so it passes through.
    """
    lo, hi = ECHO_UNIT_REPAIR[col]
    out = s.copy()
    out[s < lo] = s[s < lo] * 100.0
    out[s >= hi] = s[s >= hi] / 100.0
    return out


def log_milestone(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROJECT / "logs" / "activity_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [measwas] {message}\n")
