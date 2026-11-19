"""Shared helpers for the QWAS sleep-questionnaire association stage.

QWAS is the prevalence scan over the cleaned sleep questionnaire: each questionnaire
item (and four derived instrument scores) is tested against OSA severity using the
exact same exposure, contrasts and adjustment ladder as LabWAS / MedWAS / UtilWAS /
ProcWAS / BehWAS. It reuses the BehWAS engine (_behwas_engine.py) because that engine
already carries both outcome kinds QWAS needs -- logistic OR for binary items and
OLS-on-rank-inverse-normal beta for ordinal/continuous items.

Unlike the other stages the analytic sample is NOT the full 222,109 v2 cohort: only
patients with a questionnaire on or before their sleep-study index date are testable
(~74k). The v2 cohort still supplies the exposure and every covariate.

Data-safety: the questionnaire source is real EHR/PHI. The per-patient target matrix
stays on local disk keyed on the 12-char MRN (`id`); everything printed or written to
analysis/ is a de-identified aggregate.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# reuse the icd_analysis canonical locations (same pattern as behwas/procwas utils)
_ICD = Path(__file__).resolve().parent.parent / "icd_analysis"
sys.path.insert(0, str(_ICD))
import utils as _U  # noqa: E402

PROJECT = _U.PROJECT
DATA = _U.DATA

# The v2 ICD cohort is the exposure/covariate spine (severity + index date + covariates).
COHORT = (PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()
# Deliverable of the questionnaire-cleaning pipeline (analyses/questionnaire/01-08).
QUESTIONNAIRE = PROJECT / "results" / "questionnaire_results" / "questionnaire_wide.parquet"

RESULTS = PROJECT / "results" / "qwas_results"
TARGETS_DIR = RESULTS / "targets"
ANALYSIS = RESULTS / "analysis"
PLOTS = ANALYSIS / "plots"
TABLES = ANALYSIS / "tables"

SEVERITIES = ["None", "Mild", "Moderate", "Severe"]

# ---- sample-size floors ----------------------------------------------------
MIN_PATIENTS = 500      # answered-item floor for a feature to enter the phenome
MIN_EVENTS = 100        # minority-class floor for the binary form of a feature

# ---- covariate ladder ------------------------------------------------------
# m1 crude / m2 demographics+index-year / m3 (+smoking+income+obs_window, == the m3
# every other stage uses) / m4 (+BMI). obs_window is a covariate, not an offset --
# QWAS has no rate outcome.
_M2 = "age_z + sex_M + C(RACE_ETH) + C(study_year)"
_M3 = _M2 + " + C(smoking_status) + C(income_gp) + obs_window_z"
COV = {"m1": "", "m2": f" + {_M2}", "m3": f" + {_M3}", "m4": f" + {_M3} + bmi_z"}

# ---- item scaling ----------------------------------------------------------
# Ordinal families get BOTH forms: the continuous form keeps the full ordinal
# information (rank-inverse-normal -> OLS beta), the binary form gives a prevalence
# (logistic -> OR). The cut is the uniform "moderate or worse" midpoint of each
# scale; FOSQ runs the other way (higher = better function) so its cut is inverted.
BINARY_CUT = {
    "ess": (">=", 2),        # moderate/high chance of dozing
    "freq": (">=", 2),       # at least 1-2 times per week
    "isi_sev": (">=", 2),    # moderate or worse
    "isi_sat": (">=", 2),    # moderately satisfied or worse (higher = more dissatisfied)
    "isi_imp": (">=", 2),    # somewhat or more
    "phq": (">=", 2),        # more than half the days
    "fosq": ("<=", 2),       # moderate or extreme difficulty (reverse-scored)
}
# Families that are already 0/1 -> binary form only.
BINARY_ONLY = {"yesno", "cc_binary", "timebin_onehot"}
# Open-ended numeric families -> continuous form only (no defensible clinical cut).
CONT_ONLY = {"count", "hours", "latency", "time"}

# FOSQ value 0 = "difficulty for other reasons", which standard FOSQ scoring treats
# as NOT APPLICABLE rather than as the worst response. Recoded to <NA> so the item
# scale is a clean 1-4 (4 = no difficulty); 4.3% of answered FOSQ cells.
FOSQ_NA_VALUE = 0
FOSQ_MIN_ITEMS = 5       # >=5 of 10 items answered to score FOSQ_MEAN

# ---- excluded columns ------------------------------------------------------
# STOP5/6/8   never populated (EHR-derived by design)
# STOP7       22.9% fill; RLS4 1.6%; NAPDURATION 16.3% (Somnoware-only) -- too sparse
# CC1         raw free text (kept only as the CC_* one-hots)
# *_CONT      clock minutes wrap at midnight (23:00=1380, 00:30=30); the 2-hour bin
#             one-hots carry the same information without the discontinuity
EXCLUDED = {"STOP5", "STOP6", "STOP8", "STOP7", "RLS4", "NAPDURATION", "CC1",
            "BEDTIME_CONT", "WAKETIME_CONT", "BEDTIME", "WAKETIME"}

# ---- derived instrument scores --------------------------------------------
# (name, source items, all-items-required?, continuous range, clinical binary cut)
COMPOSITES = [
    ("ESS_TOTAL", [f"ESS{i}" for i in range(1, 9)], True, "0-24", (">=", 11)),
    ("ISI_TOTAL", [f"ISI{i}" for i in range(1, 8)], True, "0-28", (">=", 15)),
    ("PHQ2_TOTAL", ["PHQ1", "PHQ2"], True, "0-6", (">=", 3)),
    ("FOSQ_MEAN", [f"FOSQ{i}" for i in range(1, 11)], False, "1-4", None),
]

# ---- domains (Manhattan grouping / plot colours) ---------------------------
DOMAIN = {
    **{f"ESS{i}": "Sleepiness" for i in range(1, 9)},
    "ESS_TOTAL": "Sleepiness", "SLEEPY": "Sleepiness", "DROWSYDRIVING": "Sleepiness",
    "SLEEPATTACK": "Sleepiness", "SVS2": "Sleepiness",
    **{f"ISI{i}": "Insomnia" for i in range(1, 8)},
    "ISI_TOTAL": "Insomnia", "LATENCY": "Insomnia", "AWAKENINGS": "Insomnia",
    "SVS1": "Insomnia", "MEDS": "Insomnia",
    "PHQ1": "Mood", "PHQ2": "Mood", "PHQ2_TOTAL": "Mood", "PTSD": "Mood",
    **{f"FOSQ{i}": "Function" for i in range(1, 11)},
    "FOSQ_MEAN": "Function",
    "STOP1": "OSA symptoms", "STOP2": "OSA symptoms", "STOP3": "OSA symptoms",
    "STOP4": "OSA symptoms", "GASP": "OSA symptoms",
    "PARASOMNIA": "Parasomnia", "SLEEPHALLUCINATION": "Parasomnia",
    "SLEEPPARALYSIS": "Parasomnia", "CATAPLEXY": "Parasomnia",
    "RLS1": "Restless legs", "RLS2": "Restless legs", "RLS3": "Restless legs",
    "SLEEPTIME": "Sleep habits", "NAPS": "Sleep habits", "CAFFEINE": "Sleep habits",
    "SHIFT": "Sleep habits", "PARTNER": "Sleep habits", "CLAUST": "Sleep habits",
    "HEADACHE": "Somatic",
}
DOMAIN_ORDER = ["Sleepiness", "Insomnia", "Mood", "Function", "OSA symptoms",
                "Parasomnia", "Restless legs", "Sleep habits", "Somatic",
                "Chief complaint"]

# Airway-specific items that are themselves the reason patients get referred for a
# sleep study -> their association with severity is partly ascertainment, not
# biology. Kept in the phenome (user decision) but flagged in every output.
REFERRAL_ITEMS = {"STOP1", "STOP3", "GASP", "CC_SNORING_APNEA"}


def domain_of(feature: str) -> str:
    if feature.startswith("CC_"):
        return "Chief complaint"
    if feature.startswith(("BEDTIME_", "WAKETIME_")):
        return "Sleep habits"
    return DOMAIN.get(feature, "Other")


def ensure_dirs() -> None:
    for d in (RESULTS, TARGETS_DIR, ANALYSIS, PLOTS, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def log_milestone(message: str) -> None:
    """Append a timestamped milestone to logs/activity_log.md with a [qwas] tag."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROJECT / "logs" / "activity_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [qwas] {message}\n")
