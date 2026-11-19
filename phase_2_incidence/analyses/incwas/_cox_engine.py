"""Shared incwas Cox engine — time-to-event association of OSA severity with incident phecodes.

For each phecode: at-risk = follow-up-spine patients WITHOUT that phecode at index (rule-of-1
washout); event = 2nd distinct post-index code (rule-of-2), event date = that 2nd code; cause-
specific censoring at min(event, death, coverage_end, 2023-05-31) is already baked into the
follow-up frame. Fits Cox PH (ridge-penalized for separation stability) across the nested ladder
m1 (crude) / m2 (+demo) / m3 (+SES+smoking+obs_window+baseline-utilization) / m4 (+BMI) and the
seven OSA-severity contrasts (omnibus 4-group Wald, ordinal trend, mild/moderate/severe-vs-none,
AHI>=5, AHI>=15). Design columns are built explicitly with fixed reference levels matching the v2
PheWAS (race=most frequent, study_year=earliest, smoking=Never, income=Mid).

CPAP is deliberately NOT adjusted (it is a mediator on the OSA->treatment->outcome path).
DE-IDENTIFIED aggregates only (effect sizes, counts) — no patient rows.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import incwas_utils as U

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

PENALIZER = 0.1           # small ridge: finite estimates under sparse/near-separated phecodes
SMK_REF = "Never"
INC_REF = "Mid"

# contrast -> (exposure design columns, kind)
PAIRS = [("mild_vs_none", "sev_Mild"), ("moderate_vs_none", "sev_Moderate"), ("severe_vs_none", "sev_Severe")]
SEV_DUMMIES = ["sev_Mild", "sev_Moderate", "sev_Severe"]

_PREP = None      # (followup design frame, model/exposure column blocks)
_ID2IX = None     # {id -> positional index in the followup frame}
_PREVIX = None    # {phecode -> np.int32 array of followup indices with a baseline occurrence}
_EXCL = None      # {phecode -> set of family phecodes} for the harmonized (control-exclusion) washout


def _dummies(s: pd.Series, prefix: str, ref) -> tuple[pd.DataFrame, list[str]]:
    """One-hot `s`, drop the reference level; returns (frame, kept-column-names)."""
    d = pd.get_dummies(s.astype("string").fillna("~NA~"), prefix=prefix, dtype=float)
    refcol = f"{prefix}_{ref}"
    if refcol in d.columns:
        d = d.drop(columns=[refcol])
    return d, list(d.columns)


def prep_followup() -> tuple[pd.DataFrame, dict]:
    """Load the follow-up spine and build the full design matrix ONCE (cached per process).

    Returns (frame, blocks) where blocks maps model name -> covariate column list and also holds
    the exposure column lists. z-scores/dummies are computed on the whole spine so references and
    scaling are constant across every phecode's at-risk subset.
    """
    global _PREP, _ID2IX
    if _PREP is not None:
        return _PREP

    f = pd.read_parquet(U.FOLLOWUP_FRAME).reset_index(drop=True)
    f["idx"] = pd.to_datetime(f["idx"])
    f["censor_date"] = pd.to_datetime(f["censor_date"])

    # continuous (z-scored on the spine); enc rate is log-transformed first (right-skewed)
    f["age_z"] = _z(f["age"])
    f["bmi_z"] = _z(f["bmi"])
    f["obs_window_z"] = _z(f["obs_window"])
    f["enc_rate_z"] = _z(np.log1p(f["enc_per_year"]))            # baseline utilization / detection proxy
    f["sex_M"] = f["sex_M"].astype(float)

    # exposure encodings
    f["severity_ord"] = f["severity_ord"].astype(float)
    for lvl in ("Mild", "Moderate", "Severe"):
        f[f"sev_{lvl}"] = (f["severity"] == lvl).astype(float)
    f["ahi_ge5"] = (f["AHI"] >= 5).astype(float)
    f["ahi_ge15"] = (f["AHI"] >= 15).astype(float)

    # categorical dummy blocks with fixed references (v2 convention)
    race_ref = f["RACE_ETH"].astype("string").fillna("~NA~").value_counts().idxmax()
    year_ref = int(f["study_year"].min())
    race_d, race_cols = _dummies(f["RACE_ETH"], "race", race_ref)
    year_d, year_cols = _dummies(f["study_year"].astype(int).astype(str), "yr", str(year_ref))
    smk_d, smk_cols = _dummies(f["smoking_status"], "smk", SMK_REF)
    inc_d, inc_cols = _dummies(f["income_gp"], "inc", INC_REF)
    f = pd.concat([f, race_d, year_d, smk_d, inc_d], axis=1)

    demo = ["age_z", "sex_M"] + race_cols + year_cols
    m3 = demo + smk_cols + inc_cols + ["obs_window_z", "enc_rate_z"]
    blocks = {
        "models": {"m1": [], "m2": demo, "m3": m3, "m4": m3 + ["bmi_z"]},
        "exposures": {
            "categorical": SEV_DUMMIES,
            "trend": ["severity_ord"],
            "ahi_ge5": ["ahi_ge5"],
            "ahi_ge15": ["ahi_ge15"],
        },
    }
    _ID2IX = {v: i for i, v in enumerate(f["id"].to_numpy())}
    _PREP = (f, blocks)
    return _PREP


def prev_index() -> dict:
    """{phecode -> np.int32 array of followup-frame indices with a BASELINE (has_pre) occurrence}.

    Built once per process from the event store; ~34 MB. Used with the exclusion family to define the
    harmonized at-risk set. Ids outside the follow-up spine are ignored (they are not analyzed)."""
    global _PREVIX
    if _PREVIX is None:
        prep_followup()  # ensures _ID2IX
        ev = pd.read_parquet(U.EVENTS, columns=["id", "phecode", "has_pre"])
        ev = ev[ev["has_pre"]].copy()
        ev["ix"] = ev["id"].map(_ID2IX)
        ev = ev.dropna(subset=["ix"])
        ev["ix"] = ev["ix"].astype("int32")
        ev["phecode"] = ev["phecode"].astype(str)
        _PREVIX = {p: g["ix"].to_numpy() for p, g in ev.groupby("phecode")}
    return _PREVIX


def exclude_map() -> dict:
    """{phecode -> set of family phecodes} from the pyPheWAS control-exclusion table (undirected).

    Testing incident phecode P, we exclude from the at-risk set anyone with a baseline occurrence of P
    or any phecode in its exclusion family -- the PheWAS-standard control exclusion, which also
    harmonizes the ICD-9/ICD-10 phecode split (e.g. baseline 296.2 [ICD-9 depression] excludes from
    incident 296.22 [ICD-10])."""
    global _EXCL
    if _EXCL is None:
        import importlib.util
        import os
        res = os.path.join(os.path.dirname(importlib.util.find_spec("pyPheWAS").origin), "resources")
        ex = pd.read_csv(os.path.join(res, "phecode_exclude.csv"), dtype=str).dropna()
        d: dict = {}
        for x, y in zip(ex["phewas_code"], ex["exclusion_criteria"]):
            d.setdefault(x, set()).add(y)
            d.setdefault(y, set()).add(x)   # symmetric family
        _EXCL = d
    return _EXCL


# ---- landmark (grace-period) + CPAP adherence helpers -----------------------
_GRACEIX = {}   # {grace_days -> {phecode -> int32 positional indices with first_post_date <= idx+grace}}

ADH_REF = "4+ hr/night"                                          # reference = adherent CPAP users
ADH_NONREF = ["0-2 hr/night", "2-4 hr/night", "never-started", "unknown"]
ADH_LEVELS = [ADH_REF] + ADH_NONREF


def grace_index(grace_days: int = 90) -> dict:
    """{phecode -> np.int32 positional indices of patients whose FIRST post-index occurrence of the
    phecode falls within (index, index+grace_days]} -- the grace-window analogue of prev_index(), used
    to reclassify grace-window onsets as prevalent in a landmark build. Reads only first_post_date from
    the existing event store (no 104GB rescan). Cached per grace_days."""
    if grace_days not in _GRACEIX:
        f, _ = prep_followup()                         # ensures _ID2IX
        idx_by_pos = pd.to_datetime(f["idx"]).to_numpy()
        ev = pd.read_parquet(U.EVENTS, columns=["id", "phecode", "first_post_date"])
        ev = ev.dropna(subset=["first_post_date"]).copy()
        ev["ix"] = ev["id"].map(_ID2IX)
        ev = ev.dropna(subset=["ix"])
        ev["ix"] = ev["ix"].astype("int32")
        fpd = pd.to_datetime(ev["first_post_date"]).to_numpy()
        landmark = idx_by_pos[ev["ix"].to_numpy()] + np.timedelta64(grace_days, "D")
        ev = ev[fpd <= landmark]
        ev["phecode"] = ev["phecode"].astype(str)
        _GRACEIX[grace_days] = {p: g["ix"].to_numpy() for p, g in ev.groupby("phecode")}
    return _GRACEIX[grace_days]


def adherence_series(mean_pap_usage, setup_date) -> pd.Series:
    """[Therapy-anchored, DEPRECATED] 5-level CPAP adherence from the whole-record first-90-day
    compliance summary (uses info regardless of when it occurred relative to the landmark). Kept for
    reference; the landmark analyses use adherence_landmark() instead."""
    u = pd.to_numeric(mean_pap_usage, errors="coerce")
    out = pd.Series("never-started", index=u.index, dtype=object)
    out[setup_date.notna()] = "unknown"
    out[u.notna() & (u < 120)] = "0-2 hr/night"
    out[u.notna() & (u >= 120) & (u < 240)] = "2-4 hr/night"
    out[u.notna() & (u >= 240)] = "4+ hr/night"
    return out


def adherence_landmark(mean_pap_usage, setup_date, idx, grace_days) -> pd.Series:
    """5-level CPAP adherence derived STRICTLY from the [index, index+grace_days] window.

    The data has only a therapy-anchored first-90-day compliance summary (mean_pap_usage, MINUTES) +
    setup_date -- no per-night usage to re-window. So a usage LEVEL (4+/2-4/0-2 hr/night) is assigned
    only when that 90-day compliance window falls ENTIRELY within [index, index+grace]:
        index <= setup_date <= index + grace_days - 90   (so [setup, setup+90] subset of [index, land]).
    Device set up by the landmark but its compliance window not within-window (started too late for a
    full window by grace-end, or set up pre-index) -> 'unknown'. No device by the landmark -> 'never-
    started'. No information after index+grace_days is used. (For grace_days < ~110 almost no one has a
    complete within-window compliance measure, so levels collapse -- that is the honest data limit.)"""
    u = pd.to_numeric(mean_pap_usage, errors="coerce")
    idx = pd.to_datetime(idx)
    s = pd.to_datetime(setup_date)
    land = idx + pd.to_timedelta(grace_days, unit="D")
    dev_by_land = s.notna() & (s <= land)
    out = pd.Series("never-started", index=u.index, dtype=object)
    out[dev_by_land] = "unknown"
    within = dev_by_land & (s >= idx) & (s <= land - pd.Timedelta(days=90)) & u.notna()
    out[within & (u < 120)] = "0-2 hr/night"
    out[within & (u >= 120) & (u < 240)] = "2-4 hr/night"
    out[within & (u >= 240)] = "4+ hr/night"
    return out


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / s.std()


def row(phecode, contrast, model, n_atrisk, n_events, effect, se, lo, hi, p, ph_p=np.nan, note=""):
    return dict(phecode=phecode, contrast=contrast, model=model,
                n_atrisk=int(n_atrisk), n_events=int(n_events), effect_type="HR",
                effect=effect, se=se, ci_low=lo, ci_high=hi, p=p,
                neglog10p=(-np.log10(p) if (p and p > 0 and np.isfinite(p)) else np.nan),
                ph_p=ph_p, unstable=note)


def build_survival(phecode, ev_slice, foll, prev_by_phe, excl, grace_days=0, grace_by_phe=None):
    """At-risk survival frame for one phecode, with HARMONIZED washout: exclude any patient with a
    baseline occurrence of this phecode OR any phecode in its exclusion family (fixes the ICD-9/ICD-10
    prevalent-as-incident leakage). Event = 2nd distinct post-index code for the exact phecode.

    LANDMARK mode (grace_days > 0): time origin moves to index + grace_days; anyone with a FIRST
    occurrence of the phecode (or family) within (index, index+grace] is reclassified prevalent (via
    grace_by_phe from grace_index()); and only patients still event-free/alive/enrolled past the
    landmark (censor_date > origin) enter. This removes immortal-time bias for a landmark-assigned
    exposure. grace_days=0 (default) reproduces the original index-anchored build exactly."""
    grace = pd.Timedelta(days=grace_days)
    prevalent = np.zeros(len(foll), dtype=bool)
    srcs = (prev_by_phe,) if grace_by_phe is None else (prev_by_phe, grace_by_phe)
    for q in (excl.get(phecode, set()) | {phecode}):
        for src in srcs:
            a = src.get(q)
            if a is not None and len(a):
                prevalent[a] = True
    at = foll[~prevalent].merge(ev_slice[["id", "second_post_date"]], on="id", how="left")
    at["origin"] = at["idx"] + grace
    if grace_days:
        at = at[at["censor_date"] > at["origin"]].copy()   # landmark entry: survived event-free to origin
    ed = pd.to_datetime(at["second_post_date"])
    is_ev = ed.notna() & (ed <= at["censor_date"])
    at["event"] = is_ev.astype(int)
    end = at["censor_date"].where(~is_ev, ed)
    at["time"] = (end - at["origin"]).dt.days / 365.25
    return at[at["time"] > 0].copy()


def _fit(d: pd.DataFrame, cols: list[str]):
    """Fit ridge Cox on time/event + cols (dropping zero-variance cols); return (cph, kept)."""
    keep = [c for c in cols if d[c].nunique() > 1]
    sub = d[["time", "event"] + keep]
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(sub, duration_col="time", event_col="event", batch_mode=True)
    return cph, keep


def _eff(cph, name):
    s = cph.summary.loc[name]
    return (float(s["exp(coef)"]), float(s["se(coef)"]),
            float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"]))


def _wald_omnibus(cph, terms):
    """Joint Wald chi-square that all `terms` coefficients are 0 (omnibus 4-group test)."""
    terms = [t for t in terms if t in cph.params_.index]
    if len(terms) < 2:
        return np.nan, np.nan, len(terms)
    b = cph.params_.loc[terms].to_numpy()
    V = cph.variance_matrix_.loc[terms, terms].to_numpy()
    try:
        stat = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, len(terms)
    df = len(terms)
    return stat, float(stats.chi2.sf(stat, df)), df


PH_SAMPLE = 8000


def _ph_sub(d, cols, terms):
    """Proportional-hazards (Schoenfeld) p on a stratified subsample (all events kept, controls
    sampled to ~PH_SAMPLE). The full lifelines test scales badly (~5 min at 90k rows), so we screen
    on a subsample -- flag, don't gate. Returns the min p over the exposure `terms`."""
    ev1 = d[d["event"] == 1]
    if len(ev1) > PH_SAMPLE:
        ev1 = ev1.sample(PH_SAMPLE, random_state=0)
    ev0 = d[d["event"] == 0]
    ntake = min(len(ev0), max(PH_SAMPLE - len(ev1), 2000))
    sub = pd.concat([ev1, ev0.sample(ntake, random_state=0)]) if ntake else ev1
    keep = [c for c in cols if sub[c].nunique() > 1]
    terms = [t for t in ([terms] if isinstance(terms, str) else terms) if t in keep]
    if not terms:
        return np.nan
    try:
        s = sub[["time", "event"] + keep]
        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(s, "time", "event", batch_mode=True)
        r = proportional_hazard_test(cph, s, time_transform="rank")
        ps = [float(r.summary.loc[t, "p"]) for t in terms if t in r.summary.index]
        return float(np.nanmin(ps)) if ps else np.nan
    except Exception:
        return np.nan


def run_phecode(phecode, ev_slice, foll, blocks, prev_by_phe, excl) -> list[dict]:
    """All 7 contrasts x m1-m4 for one phecode. Returns [] if below the incident-event floor."""
    d = build_survival(phecode, ev_slice, foll, prev_by_phe, excl)
    n_at = len(d)
    n_ev = int(d["event"].sum())
    if n_ev < U.MIN_EVENTS:
        return []

    out = []
    models = blocks["models"]
    exps = blocks["exposures"]
    for model, cov in models.items():
        # ---- categorical severity: omnibus (Wald) + mild/moderate/severe-vs-none ----
        npar = len([c for c in SEV_DUMMIES if d[c].nunique() > 1]) + len(cov)
        epv_note = "epv<10" if n_ev < U.EPV * max(npar, 1) else ""
        try:
            cph, kept = _fit(d, SEV_DUMMIES + cov)
            stat, p_om, dfo = _wald_omnibus(cph, SEV_DUMMIES)
            ph_sev = _ph_sub(d, SEV_DUMMIES + cov, SEV_DUMMIES) if model == "m4" else np.nan
            out.append(row(phecode, "omnibus", model, n_at, n_ev, stat, np.nan, np.nan, np.nan, p_om,
                           ph_sev, epv_note))
            for cname, col in PAIRS:
                if col in kept:
                    e, se, lo, hi, p = _eff(cph, col)
                    note = epv_note + ("|unstable" if (abs(np.log(e)) > 10 or se > 10) else "")
                    out.append(row(phecode, cname, model, n_at, n_ev, e, se, lo, hi, p, ph_sev, note))
        except Exception as ex:
            out.append(row(phecode, "omnibus", model, n_at, n_ev, np.nan, np.nan, np.nan, np.nan, np.nan,
                           note=str(ex)[:40]))

        # ---- trend + AHI dichotomies ----
        for cname, ecols in (("trend", exps["trend"]), ("ahi_ge5", exps["ahi_ge5"]),
                             ("ahi_ge15", exps["ahi_ge15"])):
            col = ecols[0]
            if d[col].nunique() < 2:
                continue
            npar = 1 + len(cov)
            epv_note = "epv<10" if n_ev < U.EPV * npar else ""
            try:
                cph, kept = _fit(d, ecols + cov)
                if col not in kept:
                    continue
                e, se, lo, hi, p = _eff(cph, col)
                php = _ph_sub(d, ecols + cov, col) if (model == "m4" and cname == "trend") else np.nan
                note = epv_note + ("|unstable" if (abs(np.log(e)) > 10 or se > 10) else "")
                out.append(row(phecode, cname, model, n_at, n_ev, e, se, lo, hi, p, php, note))
            except Exception as ex:
                out.append(row(phecode, cname, model, n_at, n_ev, np.nan, np.nan, np.nan, np.nan, np.nan,
                               note=str(ex)[:40]))
    return out


def add_fdr_bonferroni(res: pd.DataFrame, group_cols) -> pd.DataFrame:
    """BH-FDR + Bonferroni significance flags within each group (matches the project convention)."""
    res = res.copy()
    res["sig_bon"] = False
    res["sig_fdr"] = False
    valid = res[res["p"].notna() & np.isfinite(res["p"])]
    for _, idx in valid.groupby(group_cols).groups.items():
        sub = res.loc[idx]
        n = len(sub)
        res.loc[idx, "sig_bon"] = sub["p"] < 0.05 / n
        order = sub["p"].sort_values()
        crit = 0.05 * np.arange(1, n + 1) / n
        passed = order.to_numpy() <= crit
        thr = order.to_numpy()[passed].max() if passed.any() else -1
        res.loc[idx, "sig_fdr"] = sub["p"] <= thr
    return res
