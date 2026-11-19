# %% [markdown]
# Shared helpers for the questionnaire-cleaning pipeline.
#
# Reuses the icd_analysis helpers (`normalize_mrn`, `parse_sas_datetime`) without
# modifying that package (same pattern as `labwas_utils.py`), and defines the
# questionnaire output tree, the 54-column target schema, the per-item decoders,
# and loaders that read the crosswalk maps straight out of the MASTER workbook so
# the encoding has a single source of truth.
#
# The answer-encoding maps below are EXACT and were verified against the
# distinct KPHC answer-option sets profiled in 02_profile_sources (PROFILE.md).
#
# Data-safety: nothing here prints row-level identifiers. Callers emit only
# de-identified aggregates (counts, distinct option sets) to stdout/reports.
# %%
from __future__ import annotations

import csv
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

# ---- reuse the icd_analysis helpers without touching that package ----------
_ICD = Path(__file__).resolve().parent.parent / "icd_analysis"
sys.path.insert(0, str(_ICD))
import utils as _U  # noqa: E402

normalize_mrn = _U.normalize_mrn
parse_sas_datetime = _U.parse_sas_datetime
PROJECT = _U.PROJECT
DATA = _U.DATA

csv.field_size_limit(10_000_000)
_NA = pd.NA

# ---- canonical locations ---------------------------------------------------
RESULTS = PROJECT / "results" / "questionnaire_results"
PROFILE_DIR = RESULTS / "profile"

# ---- source data (in-scope only; see plan, decisions 5-7) ------------------
QDIR = DATA / "questionnaire"
KPHC_CSV = QDIR / "sleep_history_quest.csv"          # KPHC/Epic, comma, has STUDY_DATE
SOMNO_CSV = QDIR / "kaiser_questionnaire.csv"         # Somnoware, ^||^-delimited
MASTER_XLSX = QDIR / "MASTER sample_questionnairesXL-forclaude.xlsx"
TARGET_XLSX = QDIR / "target_datastructure.xlsx"

SAS = "%d%b%Y:%H:%M:%S"
# DuckDB in-SQL MRN normalizer (mirrors normalize_mrn): zero-pad to 12 chars,
# dropping a trailing '.0' left by float parsing. Never CAST(MRN AS INT).
MRN_SQL = r"lpad(regexp_replace(CAST(MRN AS VARCHAR), '\.0$', ''), 12, '0')"

# ---- target schema (verbatim header of target_datastructure.xlsx) ----------
KEY_COLS = ["MRN", "STUDY_DATE", "CONTACT_DATE"]
TARGET_COLUMNS = [
    "MRN", "STUDY_DATE", "CONTACT_DATE",
    "ESS1", "ESS2", "ESS3", "ESS4", "ESS5", "ESS6", "ESS7", "ESS8",
    "GASP", "SLEEPY", "DROWSYDRIVING", "CLAUST", "MEDS", "PARTNER",
    "PARASOMNIA", "SLEEPATTACK", "SLEEPHALLUCINATION", "CATAPLEXY",
    "ISI1", "ISI2", "ISI3", "ISI4", "ISI5", "ISI6", "ISI7",
    "PHQ1", "PHQ2",
    "FOSQ1", "FOSQ2", "FOSQ3", "FOSQ4", "FOSQ5", "FOSQ6", "FOSQ7", "FOSQ8", "FOSQ9", "FOSQ10",
    "PTSD", "SHIFT", "BEDTIME", "WAKETIME", "LATENCY",
    "RLS1", "RLS2", "RLS3", "SVS1", "SVS2",
    "NAPS", "HEADACHE", "AWAKENINGS", "CAFFEINE",
    "STOP1", "STOP2", "STOP3", "STOP4", "STOP5", "STOP6", "STOP7", "STOP8",
    "CC1", "NAPDURATION", "SLEEPTIME",
    "SLEEPPARALYSIS",              # off-target extra column, appended per user
    "RLS4",                        # RLSHowOften (Scanned) frequency 0-5
    "BEDTIME_CONT", "WAKETIME_CONT",  # continuous minutes-since-midnight (0-1439)
]
TARGET_CODES = [c for c in TARGET_COLUMNS if c not in KEY_COLS]  # item codes
# STOP5/STOP6/STOP8 (BMI/Age/Gender) are intentionally NOT sourced here -> the
# columns exist but stay <NA>; the user derives them from EHR.
EMPTY_CODES = {"STOP5", "STOP6", "STOP8"}
# BEDTIME/WAKETIME (binned labels) are DERIVED in unify from *_CONT (not extracted).
DERIVED_CODES = {"BEDTIME", "WAKETIME"}
EXTRACT_CODES = [c for c in TARGET_CODES if c not in DERIVED_CODES]

# CC1 chief-complaint -> one-hot binary columns CC_<cat> (added in unify). Each
# option (from any source) is keyword-matched to a canonical complaint category.
CC_CATEGORIES = {
    "SNORING_APNEA": ["snor", "apnea", "gasp", "paus", "stop breathing", "breathing"],
    "INSOMNIA": ["insomnia", "falling asleep", "staying asleep", "waking too early",
                 "not enough sleep", "low sleep", "init", "frag", "fall asleep",
                 "stay asleep", "problems falling", "problems staying"],
    "SLEEPINESS": ["sleepiness", "excessive sleep", "hard to wake", "sleepy", "drowsy"],
    "FATIGUE": ["fatigue", "tired"],
    "RESTLESS_LEGS": ["restless leg"],
    "PARASOMNIA": ["unusual", "behavior", "walking", "punching", "grind", "teeth",
                   "act out", "kick", "sleepwalk"],
    "PAP_THERAPY": ["apnea therapy", "pap therapy", "cpap", "with my sleep apnea"],
    "HEADACHE": ["headache"],
    "DRY_MOUTH": ["dry mouth"],
    "NOCTURIA": ["bathroom", "restroom", "urinat"],
    "REFERRAL": ["sent me", "clearance", "commercial driver"],
    "OTHER": ["other", "none", "no complaint"],
}
CC_COLUMNS = ["CC_" + k for k in CC_CATEGORIES]
FULL_COLUMNS = TARGET_COLUMNS + CC_COLUMNS       # final output schema (unify/verify)


def cc_onehot(cc_value) -> dict:
    """Map a chief-complaint string to {CC_<cat>: 0/1}. All-NA if the source value
    is missing (unknown). Keywords are matched against the whole string (sources
    use different separators and some option labels contain commas). CC_OTHER = 1
    when the text says other/none OR nothing else matched."""
    if cc_value is None or pd.isna(cc_value) or not str(cc_value).strip():
        return {c: pd.NA for c in CC_COLUMNS}
    s = str(cc_value).lower()
    out = {c: 0 for c in CC_COLUMNS}
    matched = False
    for cat, kws in CC_CATEGORIES.items():
        if cat == "OTHER":
            continue
        if any(k in s for k in kws):
            out["CC_" + cat] = 1
            matched = True
    if not matched or any(k in s for k in CC_CATEGORIES["OTHER"]):
        out["CC_OTHER"] = 1
    return out

# Target spelling `SLEEPHALLUCINATION` is a typo-fixed alias for the KPHC col-I
# code `SLEEPHALLCUINATION`; map raw->target when reading the workbook.
COL_I_ALIASES = {"SLEEPHALLCUINATION": "SLEEPHALLUCINATION",
                 "BEDTIME": "BEDTIME_CONT", "WAKETIME": "WAKETIME_CONT"}

# Target codes NOT present verbatim in col-I -> mapped to a KPHC QUEST_ID by hand:
#   CC1   <- 45190834 (chief complaint; col-I row is uncoded)
#   STOP7 <- 45190838 (collar size / BANG row; reused as the neck criterion)
MANUAL_KPHC = {
    "45190834": ("CC1", "Which of the following would you say is your chief complaint?"),
    "45190838": ("STOP7", "Collar size (neck) -> STOP-BANG neck criterion"),
}


# ---- answer encodings (EXACT maps, verified against 02 PROFILE.md) ----------
def norm(value) -> str:
    """Normalize an answer for lookup: strip, collapse whitespace, lowercase."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().lower().split())


_ESS = {                                            # Epworth doze-likelihood 0..3
    "there is no chance i would ever doze or sleep": 0,
    "there is a slight chance i would doze or sleep": 1,
    "there is a moderate chance i would doze or sleep": 2,
    "there is a high chance i would doze or sleep": 3,
}
_FREQ = {                                           # "how often" 0..4 (word labels
    "never or very rarely": 0,                      # only -- NO digit fallback, the
    "1-2 times per month": 1,                       # leading digit is misleading)
    "1-2 times per week": 2,
    "3-4 times per week": 3,
    "almost every day": 4, "almost everyday": 4,     # KPHC vs Somnoware spelling
}
_YESNO = {"y": 1, "yes": 1, "n": 0, "no": 0,
          "1": 1, "0": 0, "true": 1, "false": 0}
_ISI_SEV = {"none": 0, "mild": 1, "moderate": 2, "severe": 3, "very severe": 4}
_ISI_SAT = {                                        # ISI4 satisfaction (reverse)
    "very satisfied": 0, "satisfied": 1, "moderately satisfied": 2,
    "dissatisfied": 3, "very dissatisfied": 4,
}
_ISI_IMP = {                                        # ISI5-7 impact 0..4
    "not at all noticeable": 0, "not at all worried": 0, "not at all interfering": 0,
    "a little": 1, "somewhat": 2, "much": 3,
    "very much noticeable": 4, "very much worried": 4, "very much interfering": 4,
}
_PHQ = {"not at all": 0, "several days": 1, "more than half the days": 2,
        "nearly every day": 3}
_FOSQ = {                       # STANDARD FOSQ 0..4 (higher = better function)
    "difficulty for other reasons": 0, "difficullty for other reasons": 0,
    "affected for other reasons": 0,
    "yes, extreme": 1, "yes, moderate": 2, "yes, a little": 3, "no": 4,
}
_LATENCY = {                  # categorical bins -> midpoint minutes (JUDGMENT CALL)
    "< 15 minutes": 8, "15- 30 minutes": 23, "15-30 minutes": 23, "30-60 minutes": 45,
    "1-2 hours": 90, "2-3 hours": 150, "3-4 hours": 210, "> 4 hours": 270,
}
_DURATION = {                 # NAPDURATION bins -> midpoint minutes (confirm in 02)
    "< 15 minutes": 8, "15- 30 minutes": 23, "15-30 minutes": 23, "30-60 minutes": 45,
    "30-90 minutes": 60, "1-2 hours": 90, "2-3 hours": 150, "3-4 hours": 210,
    "> 4 hours": 270,
}
_RLSFREQ = {                  # RLS4 (Scanned RLSHowOften) ordered frequency 0..5
    "< 1 time per year": 0, "< 1 time per month": 1, "1-4 times per month": 2,
    "2-3 times per week": 3, "4-5 times per week": 4, "6-7 times per week": 5,
}
_MAPS = {
    "ess": _ESS, "freq": _FREQ, "yesno": _YESNO, "isi_sev": _ISI_SEV,
    "isi_sat": _ISI_SAT, "isi_imp": _ISI_IMP, "phq": _PHQ, "fosq": _FOSQ,
    "latency": _LATENCY, "duration": _DURATION, "rlsfreq": _RLSFREQ,
}

# ---- code -> family (decoder selection) ------------------------------------
FAMILY: dict[str, str] = {
    **{f"ESS{i}": "ess" for i in range(1, 9)},
    **{c: "freq" for c in ["GASP", "SLEEPY", "DROWSYDRIVING", "CLAUST", "MEDS",
                           "PARTNER", "PARASOMNIA", "SLEEPATTACK", "SLEEPHALLUCINATION"]},
    **{c: "yesno" for c in ["CATAPLEXY", "PTSD", "SHIFT", "RLS1", "RLS2", "RLS3"]},
    "ISI1": "isi_sev", "ISI2": "isi_sev", "ISI3": "isi_sev",
    "ISI4": "isi_sat",
    "ISI5": "isi_imp", "ISI6": "isi_imp", "ISI7": "isi_imp",
    "PHQ1": "phq", "PHQ2": "phq",
    **{f"FOSQ{i}": "fosq" for i in range(1, 11)},
    "BEDTIME": "timebin", "WAKETIME": "timebin", "LATENCY": "latency",
    **{c: "count" for c in ["NAPS", "SVS1", "SVS2", "HEADACHE", "AWAKENINGS", "CAFFEINE"]},
    # --- REV2 additions ---
    "STOP1": "yesno", "STOP2": "yesno", "STOP3": "yesno", "STOP4": "yesno",
    "STOP7": "neck",
    "STOP5": "empty", "STOP6": "empty", "STOP8": "empty",
    "CC1": "text", "NAPDURATION": "duration", "SLEEPTIME": "hours",
    "SLEEPPARALYSIS": "freq",     # KPHC "how often" section; confirm in 02 re-profile
    "RLS4": "rlsfreq",            # Scanned RLSHowOften 0-5
    "BEDTIME_CONT": "time", "WAKETIME_CONT": "time",  # continuous minutes
}

# valid decoded range per family (for verify-time bounds checks)
FAMILY_RANGE = {
    "ess": (0, 3), "freq": (0, 4), "yesno": (0, 1), "isi_sev": (0, 4),
    "isi_sat": (0, 4), "isi_imp": (0, 4), "phq": (0, 3), "fosq": (0, 4),
    "latency": (0, 1440), "time": (0, 1439), "count": (0, 9),
    "neck": (0, 1), "duration": (0, 1440), "hours": (0, 24), "rlsfreq": (0, 5),
    "empty": (0, 0), "text": (0, 0), "timebin": (0, 0),
}
# families that carry text / label / no numeric value (skip range validation)
NONNUMERIC_FAMILIES = {"text", "empty", "timebin"}

_TIME_AMPM = re.compile(r"^(\d{1,2})[:.](\d{2})\s*([ap])\.?m?\.?$")
_TIME_HONLY = re.compile(r"^(\d{1,2})\s*([ap])\.?m?\.?$")
_TIME_24 = re.compile(r"^(\d{1,2})[:.](\d{2})$")


def decode_time_minutes(value):
    """Clock time -> minutes since midnight. KPHC stores 24h 'HH:MM'; also
    tolerates '10:30 pm' / '7 am'. NOTE: late-night bedtimes wrap (23:00=1380,
    00:30=30) -- a modeling caveat flagged in the crosswalk."""
    s = norm(value)
    if not s:
        return _NA
    ap = ""
    if (m := _TIME_AMPM.match(s)):
        hh, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    elif (m := _TIME_HONLY.match(s)):
        hh, mm, ap = int(m.group(1)), 0, m.group(2)
    elif (m := _TIME_24.match(s)):
        hh, mm = int(m.group(1)), int(m.group(2))
    else:
        return _NA
    if ap == "p" and hh != 12:
        hh += 12
    elif ap == "a" and hh == 12:
        hh = 0
    if not (0 <= hh < 24 and 0 <= mm < 60):
        return _NA
    return hh * 60 + mm


def decode_count(value):
    """Per-period count: 'Never'->0, '>8'->9, else leading integer (0-8; 9-15->9;
    >15 = data-entry junk -> <NA>)."""
    s = norm(value)
    if not s:
        return _NA
    if "never" in s:
        return 0
    if s.startswith(">") or ">8" in s or s.startswith("8+"):
        return 9
    m = re.search(r"(\d+)", s)
    if not m:
        return _NA
    n = int(m.group(1))
    if n > 15:
        return _NA          # implausible count (manual-entry error)
    return min(n, 9)        # cap at 9 (== ">8")


def decode_hours(value):
    """Hours of sleep per night -> integer 0..24 ('Never'->0)."""
    s = norm(value)
    if not s:
        return _NA
    if "never" in s:
        return 0
    m = re.search(r"(\d+)", s)
    if not m:
        return _NA
    h = int(m.group(1))
    return h if 0 <= h <= 24 else _NA


_NECK_FRAC = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75}


def decode_neck(value):
    """STOP-BANG neck: parse collar size (inches) then binarize > 15.75 in
    (40 cm) -> 1. Handles '16', '16.5', '16 1/2', '17 inches'. Out-of-range
    (<10 or >26 in) or unparseable -> <NA>."""
    s = norm(value)
    if not s or "unknown" in s or "not sure" in s:
        return _NA
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return _NA
    inches = float(m.group(1))
    for frac, add in _NECK_FRAC.items():           # "16 1/2" -> 16.5
        if frac in s:
            inches += add
            break
    if not (10 <= inches <= 26):                    # implausible collar -> NA
        return _NA
    return 1 if inches >= 16 else 0                  # STOP-BANG neck: >=16 in (both sexes)


def decode_text(value):
    """Raw category passthrough (chief complaint): trimmed original string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return _NA
    s = str(value).strip()
    return s if s else _NA


# 2-hour clock bins spanning the full 24h; BEDTIME/WAKETIME hold the label string
# (one-hot-encoded downstream). Every 2h block is its own category (no wraparound).
_TIME_BINS = ["00:00-02:00", "02:00-04:00", "04:00-06:00", "06:00-08:00",
              "08:00-10:00", "10:00-12:00", "12:00-14:00", "14:00-16:00",
              "16:00-18:00", "18:00-20:00", "20:00-22:00", "22:00-00:00"]


def bin_from_minutes(m):
    """Minutes-since-midnight -> 2-hour bin label; <NA> if missing."""
    if pd.isna(m):
        return _NA
    return _TIME_BINS[int(m) // 120]


def decode_timebin(value):
    """Clock time -> 2-hour bin label (e.g. '22:00-00:00'); unparseable -> <NA>."""
    return bin_from_minutes(decode_time_minutes(value))


# Families whose Somnoware encoding is digit-prefixed ("0 (NEVER doze)",
# "4 - Very Much Interfering"): after the exact word-label map misses, fall back
# to the leading digit (in range). EXCLUDES freq (digit lies) and fosq (digit
# meaning is source-specific -> handled per-source in 05/06).
_DIGIT_FALLBACK = {"ess", "isi_sev", "isi_sat", "isi_imp", "phq"}
_LEAD1 = re.compile(r"^(\d)\b")


def decode(code: str, value):
    """Decode one raw answer for a target `code`. Empty or unmapped -> <NA>.

    Handles both source encodings: KPHC word labels (exact map) and Somnoware
    digit-prefixed labels (leading-digit fallback for the families that use it).
    """
    fam = FAMILY[code]
    if fam == "empty":
        return _NA
    if fam == "timebin":
        return decode_timebin(value)
    if fam == "time":
        return decode_time_minutes(value)
    if fam == "count":
        return decode_count(value)
    if fam == "hours":
        return decode_hours(value)
    if fam == "neck":
        return decode_neck(value)
    if fam == "text":
        return decode_text(value)
    key = norm(value)
    if key == "":
        return _NA
    v = _MAPS[fam].get(key, None)        # None => unmapped; _NA => mapped-to-NA
    if v is not None:
        return v
    if fam in _DIGIT_FALLBACK and (m := _LEAD1.match(key)):
        d, (lo, hi) = int(m.group(1)), FAMILY_RANGE[fam]
        if lo <= d <= hi:
            return d
    # freq: only a BARE single digit is trustworthy (Scanned encodes 0-4 as digits);
    # a leading digit like "1-2 times per week" would lie, so require full-match.
    if fam == "freq" and len(key) == 1 and key.isdigit():
        d, (lo, hi) = int(key), FAMILY_RANGE[fam]
        if lo <= d <= hi:
            return d
    return _NA


def known_options(code: str) -> set:
    """Normalized answer strings this code's map recognizes (unmapped audits).
    Returns empty for parsed families (time/count) which are not enumerated."""
    fam = FAMILY[code]
    return set(_MAPS[fam]) if fam in _MAPS else set()


def code_dtype(code: str) -> str:
    """Nullable pandas dtype for a code column: string for text/timebin, else Int64."""
    return "string" if FAMILY[code] in ("text", "timebin") else "Int64"


# per-family value -> label descriptions for the column data dictionary
FAMILY_VALUE_LABELS = {
    "ess": "0=No chance; 1=Slight; 2=Moderate; 3=High chance of dozing",
    "freq": "0=Never/very rarely; 1=1-2 per month; 2=1-2 per week; 3=3-4 per week; 4=Almost every day",
    "yesno": "0=No; 1=Yes",
    "isi_sev": "0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Very severe",
    "isi_sat": "0=Very satisfied; 1=Satisfied; 2=Moderately satisfied; 3=Dissatisfied; 4=Very dissatisfied",
    "isi_imp": "0=Not at all; 1=A little; 2=Somewhat; 3=Much; 4=Very much",
    "phq": "0=Not at all; 1=Several days; 2=More than half the days; 3=Nearly every day",
    "fosq": "0=Difficulty/Affected for other reasons; 1=Yes, extreme; 2=Yes, moderate; 3=Yes, a little; 4=No (no difficulty)",
    "neck": "0=collar <16 in; 1=collar >=16 in (>=40cm)",
    "hours": "hours of sleep per night (integer 0-24)",
    "count": "count per period (integer); Never=0, >8=9",
    "latency": "sleep-latency bin midpoint minutes: <15->8, 15-30->23, 30-60->45, 1-2h->90, 2-3h->150, 3-4h->210, >4h->270",
    "duration": "nap-duration bin midpoint minutes (same bins as latency)",
    "rlsfreq": "0=<1 per year; 1=<1 per month; 2=1-4 per month; 3=2-3 per week; 4=4-5 per week; 5=6-7 per week",
    "time": "minutes since midnight (0-1439)",
    "timebin": "2-hour clock bin label; one of: " + ", ".join(_TIME_BINS),
    "text": "raw chief-complaint text ('; '-joined multi-select)",
    "empty": "left blank -> derive from EHR (STOP5=BMI>35, STOP6=Age>50, STOP8=Male)",
}


def value_labels(code: str) -> str:
    return FAMILY_VALUE_LABELS.get(FAMILY.get(code, ""), "")


# ---- crosswalk loaders (single source of truth = the MASTER workbook) ------
def load_kphc_quest_map() -> pd.DataFrame:
    """KPHC sheet -> DataFrame[quest_id, code, question] for all coded rows.

    Column letters on the KPHC sheet: H(8)=QUEST_ID, I(9)=code, K(11)=QUESTION.
    Applies the SLEEPHALLCUINATION->SLEEPHALLUCINATION typo fix.
    """
    import openpyxl
    wb = openpyxl.load_workbook(MASTER_XLSX, read_only=True, data_only=True)
    ws = wb["KPHC"]
    recs = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        qid, code, ques = r[7], r[8], r[10]
        if code in (None, ""):
            continue
        code = COL_I_ALIASES.get(str(code).strip(), str(code).strip())
        recs.append({"quest_id": str(qid).strip() if qid is not None else None,
                     "code": code, "question": str(ques).strip() if ques else None})
    wb.close()
    for qid, (code, ques) in MANUAL_KPHC.items():       # hand-mapped target codes
        recs.append({"quest_id": qid, "code": code, "question": ques})
    return pd.DataFrame(recs)


def load_scanned_seed() -> pd.DataFrame:
    """Scanned sheet -> DataFrame[quest_id, friendly, code].

    Row 2 holds KPHC QUEST_IDs, row 3 holds friendly column names; join to the
    KPHC map to attach `code`. Only columns that carry a QUEST_ID are kept.
    """
    import openpyxl
    wb = openpyxl.load_workbook(MASTER_XLSX, read_only=True, data_only=True)
    ws = wb["Scanned"]
    rows = list(ws.iter_rows(min_row=1, max_row=3, values_only=True))
    wb.close()
    qids, friendly = rows[1], rows[2]
    qmap = load_kphc_quest_map().set_index("quest_id")["code"].to_dict()
    recs = []
    for i in range(len(friendly)):
        qid = qids[i] if i < len(qids) else None
        if qid in (None, ""):
            continue
        qid = str(int(qid)) if isinstance(qid, float) else str(qid).strip()
        recs.append({"quest_id": qid,
                     "friendly": str(friendly[i]).strip() if friendly[i] else None,
                     "code": qmap.get(qid)})
    return pd.DataFrame(recs)


# ---- Somnoware ^||^ parsing (caret-wrapped fields, 4-char delimiter) --------
_SPLIT = re.compile(r"\^\|\|\^")


def somno_split(line: str) -> list[str]:
    parts = _SPLIT.split(line.rstrip("\r\n"))
    if parts:
        parts[0] = parts[0].lstrip("^")
        parts[-1] = parts[-1].rstrip("^")
    return parts


# Somnoware forms that hold no target items (admin / PAP-followup / consent),
# plus the 4 forms the user asked to drop (pediatric x2, Espanol, Pre Sleep).
SOMNO_FORM_DENYLIST = {
    "upenn triage tool", "upenn consent", "kaiser permanente los angeles",
    "equipment loan agreement", "circul consent", "psg morning questionnaire",
    "attended pap morning questionnaire", "cpap follow-up questionnaire",
    "pediatric sleep questionnaire (0-4 years)",
    "pediatric  sleep questionnaire (5-17)", "pediatric sleep questionnaire (5-17)",
    "espanol questionnaire", "pre sleep",
}

# KP Sleep Questionnaire (1) is ~the same form as KP SLEEP QUESTIONNAIRE; fold it
# so the KP SLEEP QUESTIONNAIRE FEEDBACK notes apply to both.
SOMNO_FORM_ALIASES = {"kp sleep questionnaire (1)": "kp sleep questionnaire"}


def canon_form(form: str) -> str:
    f = " ".join(str(form).strip().lower().split())
    return SOMNO_FORM_ALIASES.get(f, f)

# ---- Somnoware question-text anchors -> target code ------------------------
# Substring patterns (lowercased) matched against `question` text. Seeded from
# OSA-pred ess_extract anchors + the 02 Somnoware question inventory. Used to
# PROPOSE the Somnoware crosswalk (03) and to extract labeled items (05).
# Designed to be specific enough that each question hits exactly one code; the
# matcher flags any multi-hit as a conflict for review.
SOMNOWARE_ANCHORS: dict[str, list[str]] = {
    "ESS1": ["sitting and reading"],
    "ESS2": ["watching t"],                                  # tv / television
    "ESS3": ["in a public place", "inactive in a public"],
    "ESS4": ["passenger in a car", "passenger in a car for an hour"],
    "ESS5": ["lying down"],
    "ESS6": ["talking to someone"],
    "ESS7": ["after a lunch", "after lunch", "after a meal", "quietly after a"],
    "ESS8": ["stopped for a few minutes", "stopped in traffic", "when stopped", "at a light"],
    "ISI1": ["difficulty falling asleep"],
    "ISI2": ["difficulty staying asleep"],
    "ISI3": ["waking up too early", "problems waking up too early"],
    "ISI4": ["satisfied/dissatisfied", "satisfied with your current sleep",
             "satisfied are you with your current"],
    "ISI5": ["noticeable to others"],
    "ISI6": ["worried/distressed", "worried are you about your current"],
    "ISI7": ["interfere with daily", "interfere with your daily"],
    "PHQ1": ["little interest or pleasure"],
    "PHQ2": ["feeling down, depressed", "down, depressed or hopeless"],
    "FOSQ1": ["concentrating on the things"],
    "FOSQ2": ["remembering things"],
    "FOSQ3": ["short distances", "for short distances"],
    "FOSQ4": ["long distances", "for long distances"],
    "FOSQ5": ["visiting your family"],
    "FOSQ6": ["watching a movie"],
    "FOSQ7": ["active as you want to be in the morning"],
    "FOSQ8": ["active as you want to be in the evening"],
    "FOSQ9": ["relationship with family"],
    "FOSQ10": ["with your mood", "your mood"],
    "GASP": ["gasping for air", "breath or gasping", "shortness of breath or gasping"],
    "SLEEPY": ["sleepy or tired during your waking"],
    "DROWSYDRIVING": ["asleep while driving", "off or fall asleep while driving"],
    "CLAUST": ["claustrophobia"],
    "MEDS": ["medication to help you sleep"],
    "PARTNER": ["same room as another person"],
    "PARASOMNIA": ["unusual movements during sleep", "kicking, punching or sleepwalking"],
    "SLEEPATTACK": ["sleep attacks", "falling asleep suddenly"],
    "SLEEPHALLUCINATION": ["sleep hallucinations", "things that aren"],
    "CATAPLEXY": ["cataplexy", "muscle weakness when you laugh"],
    "PTSD": ["post-traumatic stress", "(ptsd)"],
    "SHIFT": ["schedule shift", "rotating or variable schedule", "split, or variable"],
    "BEDTIME_CONT": ["what time do you get into bed", "time do you go to bed"],
    "WAKETIME_CONT": ["usual wake time", "your morning begin"],
    "LATENCY": ["how long does it take you to fall asleep", "how long do you take to fall asleep"],
    "RLS1": ["urge or need to move", "urge to move your legs", "need to move them"],
    "RLS2": ["rest and improve with movement", "occur mainly at rest"],
    "RLS3": ["worse in the evening"],
    "NAPS": ["do you take naps", "week do you take naps", "days ... take naps"],
    "HEADACHE": ["wake up with a headache"],
    "AWAKENINGS": ["times per night do you wake up", "how many times per night",
                   "times do you wake up in the night"],
    "CAFFEINE": ["caffeinated drink"],
    "SVS1": ["not satisfied with your sleep"],
    "SVS2": ["sleepiness a problem"],
    # --- REV2 additions ---
    "STOP1": ["snore loudly", "do you snore"],
    "STOP2": ["tired, fatigued, or sleepy during"],
    "STOP3": ["stop breathing during your sleep", "seen you stop breathing"],
    "STOP4": ["high blood pressure"],
    "STOP7": ["what is your collar size", "your collar size"],
    "CC1": ["would you say is your chief complaint"],
    "NAPDURATION": ["how long are your naps"],
    "SLEEPTIME": ["how much sleep do you get on average", "hours on average do you sleep",
                  "hours of sleep do you get"],
    "SLEEPPARALYSIS": ["sleep paralysis", "feel paralyzed"],
}


def match_question(qtext: str) -> list[str]:
    """Return target codes whose anchors match `qtext` (>1 => conflict to review)."""
    ql = " ".join(str(qtext).strip().lower().split())
    hits = []
    for code, anchors in SOMNOWARE_ANCHORS.items():
        if any(a in ql for a in anchors):
            hits.append(code)
    return hits


# ---- user FEEDBACK ingestion (somnoware_unmatched_questions.csv) ------------
# The reviewer tagged each unmatched question: 'x' = ignore; a code = map to it;
# a note containing 'matrix' = collapsed matrix (positional reconstruction).
FEEDBACK_ALIASES = {                 # reviewer shorthand / typo -> canonical code
    "STOP5": "STOP3", "STOPBANG6": "STOP3",   # stop-breathing -> STOP3 (per user)
    "SSLEEPY": "SLEEPY",
    "BEDTIME": "BEDTIME_CONT", "WAKETIME": "WAKETIME_CONT",   # -> continuous minutes
}
# Superseded reviewer tags to DROP (user clarified in follow-up): the "current use
# during sleep" question is TREATMENT, not CC1 -> not mapped this round.
FEEDBACK_DROP_PATTERNS = ["currently use during sleep"]


def load_feedback() -> dict:
    """Parse the reviewer's FEEDBACK column into
    {'q2code': {norm(question)->code}, 'matrix': [(form, question, note)],
     'unresolved': [...], 'dropped': [...]}."""
    import pandas as pd
    path = RESULTS / "somnoware_unmatched_questions.csv"
    out = {"q2code": {}, "matrix": [], "unresolved": [], "dropped": []}
    if not path.exists():
        return out
    df = pd.read_csv(path, dtype=str)
    col = next((c for c in df.columns if c.strip().lower() == "feedback"), None)
    if col is None:
        return out
    allcodes = set(TARGET_CODES)
    for _, r in df.iterrows():
        raw = r.get(col)
        note = "" if pd.isna(raw) else str(raw).strip()
        if not note or note.lower() in ("x", "nan"):
            continue
        form, q = str(r.get("source_form") or ""), str(r.get("question_text") or "")
        qn = " ".join(q.strip().lower().split())
        if "matrix" in note.lower():
            out["matrix"].append((form, q, note))
            continue
        if any(p in qn for p in FEEDBACK_DROP_PATTERNS):
            out["dropped"].append((form, q, note))
            continue
        token = re.split(r"[\s\-–:]", note, 1)[0].strip().upper()
        token = FEEDBACK_ALIASES.get(token, token)
        if token in allcodes:
            out["q2code"][qn] = token
        else:
            out["unresolved"].append((form, q, note))
    return out


# ---- collapsed-matrix specs (positional reconstruction in step 05) ----------
# item order per matrix; scale = # response options per item. Item order/scale
# marked None is resolved from MASTER labeled sheets + block-size profiling (02).
_ESS8 = [f"ESS{i}" for i in range(1, 9)]
# real FOSQ matrix item order (from MASTER "Sleep History Questionnaire-1 P" sheet)
_FOSQ_ORDER = ["FOSQ1", "FOSQ2", "FOSQ3", "FOSQ4", "FOSQ5",
               "FOSQ9", "FOSQ6", "FOSQ8", "FOSQ7", "FOSQ10"]
# Sleep History "how often" 19-item order; only freq target codes mapped, rest None
_HOWOFTEN19 = [None, "GASP", None, None, None, "SLEEPY", None, "DROWSYDRIVING",
               None, None, None, None, None, None, "SLEEPATTACK", None,
               "SLEEPPARALYSIS", "SLEEPHALLUCINATION", "PARASOMNIA"]


def _ident(d):
    return d


def _fosq_val(d):        # matrix digit 0=No..4=other -> standard 4=No..0=other
    return 4 - d


# mode 'exact' => question must equal stem (avoids "for each situation" collisions);
# 'contains' => substring. codes=None => deferred. digitmap: raw digit -> value.
MATRIX_SPECS = [
    {"form": "sleep history questionnaire", "stem": "situation", "mode": "exact",
     "codes": _ESS8, "scale": 4, "digitmap": _ident, "note": "ESS 8x4"},
    {"form": "sleep history questionnaire", "stem": "epworth sleep scale",
     "mode": "contains", "codes": _ESS8, "scale": 4, "digitmap": _ident,
     "note": "ESS variant stem"},
    {"form": "sleep history questionnaire", "stem": "use the following scale",
     "mode": "contains", "codes": _FOSQ_ORDER, "scale": 5, "digitmap": _fosq_val,
     "note": "FOSQ 10x5, standard dir, real item order"},
    {"form": "sleep history questionnaire", "stem": "how often do you",
     "mode": "contains", "codes": _HOWOFTEN19, "scale": 5, "digitmap": _ident,
     "note": "19-item freq screen; 7 target codes mapped"},
    {"form": "kp sleep questionnaire", "stem": "how likely are you to doze",
     "mode": "contains", "codes": None, "scale": None, "digitmap": _ident,
     "note": "abbreviated KP ESS (7 items); deferred"},
    {"form": "kp sleep questionnaire", "stem": "use the following scale",
     "mode": "contains", "codes": None, "scale": None, "digitmap": _ident,
     "note": "abbreviated KP FOSQ; deferred"},
    {"form": "patient intake form", "stem": "how often do you", "mode": "contains",
     "codes": ["SLEEPY", "DROWSYDRIVING", "CLAUST", "PARTNER", "PARASOMNIA"],
     "scale": 5, "digitmap": _ident, "note": "5x5 screen (MASTER Patient Intake Form-1pt)"},
]

# ---- misc ------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in (RESULTS, PROFILE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    """to_csv that tolerates the file being open in Excel (Windows lock)."""
    try:
        df.to_csv(path, index=False, **kw)
    except PermissionError:
        alt = path.with_name(path.stem + "_NEW" + path.suffix)
        df.to_csv(alt, index=False, **kw)
        print(f"  [locked] {path.name} open elsewhere -> wrote {alt.name}")


def log_milestone(message: str) -> None:
    """Append a timestamped milestone to logs/activity_log.md with a [questionnaire] tag."""
    log_path = PROJECT / "logs" / "activity_log.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {stamp} — [questionnaire] {message}\n")
