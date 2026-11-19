"""Per-domain construction of the model matrices.

c1  raw physiologic features -> Yeo-Johnson -> z-score (kept continuous for LPA).
c2  90 questionnaire columns -> ~30 instrument scale scores + standalone items,
    with the one-hot bedtime/waketime blocks collapsed back to single nominal
    chronotype variables.
c3  656 binary phecodes -> rolled up to integer phecodes -> chronic-disease filter.

No file I/O and no identifiers here; the numbered scripts do the reading and writing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

# =============================================================== c1 physiologic
C1_PHYSIO_11 = [
    "ODI4PCT", "SPOT_LT90", "T90%", "SPOT_LT80", "T80%", "AVERAGE_DESATURATION",
    "MEAN_SPO2", "MINIMUM_SPO2", "MEAN_PULSE_RATE", "MINIMUM_PULSE_RATE",
    "MAXIMUM_PULSE_RATE",
]
C1_ENGINEERED = ["T90_over_AHI", "supine_nonsupine_AHI_ratio"]

# --- device-level data-quality exclusions (verified against the file, not assumed) --
# Alice G3 (n=460): T90% and T80% are byte-exact copies of SPOT_LT90/SPOT_LT80 in
#   MINUTES (all 460 rows, max 100) while every other device reports them as a
#   fraction bounded by 1.0. These 460 rows alone are what drive the pooled
#   SPOT_LT90 ~ T90% correlation down to 0.15; within NOX T3 and WatchPAT it is
#   0.96/0.97 (Spearman 0.99/0.996) with an implied recording-time divisor of
#   450 and 399 minutes -- i.e. there was never a scale inconsistency, only bad rows.
# Sandman (n=2,889): T80% is a byte-exact duplicate of T90%, 0 discordant rows
#   (other devices disagree on 3-23% of rows).
# Alice G3, Polysmith, Sandman: 100% missing AVERAGE_DESATURATION, MEAN_SPO2 and
#   MINIMUM_SPO2. FIML on a row missing the whole oximetry block assigns it on pulse
#   rate alone, which is precisely how the prior clustering run manufactured its
#   "missing-oximetry device artifact" class.
C1_EXCLUDE_DEVICES = ["Alice G3", "Polysmith", "Sandman"]

# --- features excluded from the model, with the reason each was ruled out ----------
# SPOT_LT90 / SPOT_LT80 are the same physiologic quantities as T90% / T80% before
#   division by recording time (within-device Spearman 0.991 and 0.9997). Keeping
#   both double-weights hypoxemia and injects device wear time as if it were
#   physiology; the duration-normalised member of each pair is the one to keep.
# T80% carries a 56% exact-zero atom. A Gaussian mixture cannot represent a point
#   mass, so it spends a component on it: piloted at K=2,3,4 the model returned a
#   component pinned at the variance floor containing 0.00% of patients with
#   T80% > 0 -- the "phenotype" was the indicator T80% == 0. No monotone transform
#   removes a point mass, so the feature has to go. Severe desaturation is still
#   represented through MINIMUM_SPO2 and AVERAGE_DESATURATION.
C1_DROPPED_REASONS = {
    "SPOT_LT90": "duration-unnormalised duplicate of T90% (within-device Spearman 0.991)",
    "SPOT_LT80": "duration-unnormalised duplicate of T80% (within-device Spearman 0.9997)",
    "T80%": "56% exact-zero atom; produced a variance-floor degenerate component at K=2,3,4",
}
C1_FEATURES = [f for f in C1_PHYSIO_11 if f not in C1_DROPPED_REASONS] + C1_ENGINEERED

# ODI4PCT (the 4% oxygen desaturation index) correlates 0.961 with index AHI
# (log-log 0.928; within-device 0.922/0.937). Under the study's rule that AHI is
# never an input, it is AHI in all but name, and leaving it in makes "c1 classes
# track AHI severity" close to circular. Rather than pick, both are fitted as
# co-primary models and the AHI separation is reported for each.
C1_AHI_SURROGATE = "ODI4PCT"
C1_FEATURES_NO_ODI = [f for f in C1_FEATURES if f != C1_AHI_SURROGATE]


def transform_c1(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict]:
    """Yeo-Johnson + z-score, NaN-preserving.

    log1p (used by the prior clustering run) leaves |skew| up to 5.9 on T80%/T90%,
    which a Gaussian mixture answers by spending classes on the tail. Yeo-Johnson
    brings the worst |skew| to ~1.9 while staying monotone and invertible, so class
    means can be back-transformed to clinical units for the profile tables --
    unlike a rank-based inverse-normal, which normalises better but destroys that
    interpretation.
    """
    A = df[features].to_numpy(float)
    pt = PowerTransformer("yeo-johnson", standardize=True).fit(A)
    Z = pt.transform(A)
    report = {
        "features": features,
        "lambdas": dict(zip(features, np.round(pt.lambdas_, 4).tolist())),
        "skew_before": dict(zip(features, np.round(pd.DataFrame(A).skew().to_numpy(), 3).tolist())),
        "skew_after": dict(zip(features, np.round(pd.DataFrame(Z).skew().to_numpy(), 3).tolist())),
        "pct_missing": dict(zip(features, np.round(np.isnan(A).mean(0), 5).tolist())),
    }
    return Z, report


def inverse_c1(Z: np.ndarray, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Back-transform standardized values to the original clinical scale."""
    pt = PowerTransformer("yeo-johnson", standardize=True).fit(df[features].to_numpy(float))
    return pt.inverse_transform(Z)


# =============================================================== c2 symptomatic
ESS = [f"ESS{i}" for i in range(1, 9)]           # 0-3 each, total 0-24
ISI = [f"ISI{i}" for i in range(1, 8)]           # 0-4 each, total 0-28
FOSQ = [f"FOSQ{i}" for i in range(1, 11)]        # 0-4 each, HIGH = BETTER function
PHQ = ["PHQ1", "PHQ2"]                           # 0-3 each, total 0-6
STOP = ["STOP1", "STOP2", "STOP3"]               # binary
RLS = ["RLS1", "RLS2", "RLS3"]                   # binary
SVS = ["SVS1", "SVS2"]                           # 0-7

# single items kept as-is in the scale-score model (each a distinct construct;
# no candidate pair exceeds |r| = 0.58, so none is redundant with a scale total)
C2_SINGLE_ORDINAL = [
    "SLEEPY", "GASP", "DROWSYDRIVING", "CLAUST", "MEDS", "PARTNER", "PARASOMNIA",
    "SLEEPATTACK", "SLEEPHALLUCINATION", "SLEEPPARALYSIS", "HEADACHE",
    "NAPS", "AWAKENINGS", "CAFFEINE", "SLEEPTIME", "LATENCY",
]
C2_SINGLE_BINARY = ["CATAPLEXY", "PTSD", "SHIFT"]
C2_CC = [
    "CC_SNORING_APNEA", "CC_INSOMNIA", "CC_SLEEPINESS", "CC_FATIGUE",
    "CC_RESTLESS_LEGS", "CC_PARASOMNIA", "CC_PAP_THERAPY", "CC_HEADACHE",
    "CC_DRY_MOUTH", "CC_NOCTURIA", "CC_REFERRAL", "CC_OTHER",
]

# The 12 BEDTIME_*/WAKETIME_* dummies are ONE nominal variable each (row sums are
# exactly 1). Feeding them in as 12 mutually-exclusive Bernoullis would be a
# deterministic local-dependence violation. Collapse to interpretable chronotype
# bands, keeping every level >= ~5% so multinoulli cell counts stay estimable.
BEDTIME_BANDS = {
    "early_le_2200": ["BEDTIME_1800_2000", "BEDTIME_2000_2200"],
    "normal_2200_0000": ["BEDTIME_2200_0000"],
    "late_0000_0200": ["BEDTIME_0000_0200"],
    "atypical_daytime": ["BEDTIME_0200_0400", "BEDTIME_0400_0600", "BEDTIME_0600_0800",
                         "BEDTIME_0800_1000", "BEDTIME_1000_1200", "BEDTIME_1200_1400",
                         "BEDTIME_1400_1600", "BEDTIME_1600_1800"],
}
WAKETIME_BANDS = {
    "very_early_lt_0400": ["WAKETIME_0000_0200", "WAKETIME_0200_0400"],
    "early_0400_0600": ["WAKETIME_0400_0600"],
    "normal_0600_0800": ["WAKETIME_0600_0800"],
    "late_ge_0800": ["WAKETIME_0800_1000", "WAKETIME_1000_1200", "WAKETIME_1200_1400",
                     "WAKETIME_1400_1600", "WAKETIME_1600_1800", "WAKETIME_1800_2000",
                     "WAKETIME_2000_2200", "WAKETIME_2200_0000"],
}


# Count-style items where a top code of 9 is a "unknown / refused" sentinel rather
# than a real count. The signature is an empty or near-empty 8 immediately below a
# large 9: CAFFEINE has no 8 at all against n=1,833 nines, AWAKENINGS has 15 eights
# against 1,698 nines. Modelled as counts these manufacture a spurious high tail.
C2_SENTINEL_9 = ["CAFFEINE", "AWAKENINGS", "NAPS"]
C2_SENTINEL_RATIO = 5.0          # count(9) > 5 x count(8) => 9 is a sentinel
C2_SLEEPTIME_PLAUSIBLE = (2, 12)  # hours; outside this is a data-entry artifact


def clean_c2_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn enumerated sentinel/implausible codes into NaN (never into a value).

    Same philosophy as the prior project's c1 cleaning: only *enumerated* corruption
    becomes missing, and the rule is checked against the data rather than assumed.
    """
    out = df.copy()
    report: dict = {"sentinel_9_to_nan": {}, "implausible_to_nan": {}}
    for c in C2_SENTINEL_9:
        if c not in out.columns:
            continue
        n8 = int((out[c] == 8).sum())
        n9 = int((out[c] == 9).sum())
        if n9 > C2_SENTINEL_RATIO * max(n8, 1) or (n8 == 0 and n9 > 0):
            out.loc[out[c] == 9, c] = np.nan
            report["sentinel_9_to_nan"][c] = {"n_8": n8, "n_9_nulled": n9}
    if "SLEEPTIME" in out.columns:
        lo, hi = C2_SLEEPTIME_PLAUSIBLE
        bad = out.SLEEPTIME.notna() & ~out.SLEEPTIME.between(lo, hi)
        report["implausible_to_nan"]["SLEEPTIME"] = {
            "range": [lo, hi], "n_nulled": int(bad.sum())}
        out.loc[bad, "SLEEPTIME"] = np.nan
    return out, report


# --- measurement-model assignment rule (pre-declared, applied to the data) -----
# A Gaussian component can collapse onto a point mass and drive the likelihood to
# infinity; that is not a hypothetical here, since several c2 items carry 75-90% of
# their mass on a single response. The rule below keeps Gaussians only for
# indicators with enough distinct support that a spike class cannot form, sends
# heavily one-sided items to Bernoulli (where the informative contrast really is
# presence vs absence), and gives everything in between an exact categorical model.
C2_GAUSS_MIN_DISTINCT = 8
C2_GAUSS_MAX_MODAL = 0.45
C2_BERNOULLI_MIN_MODAL = 0.65


def assign_measurement_models(df: pd.DataFrame, cols: list[str],
                              force_binary: list[str] | None = None) -> tuple[dict, pd.DataFrame]:
    """Classify each indicator as gaussian / multinoulli / bernoulli from its support."""
    force_binary = set(force_binary or [])
    rows = []
    for c in cols:
        s = df[c].dropna()
        nd = int(s.nunique())
        modal = float(s.value_counts(normalize=True).max()) if len(s) else 1.0
        if c in force_binary or nd <= 2:
            model = "bernoulli"
        elif nd >= C2_GAUSS_MIN_DISTINCT and modal <= C2_GAUSS_MAX_MODAL:
            model = "gaussian"
        elif modal >= C2_BERNOULLI_MIN_MODAL:
            model = "bernoulli"
        else:
            model = "multinoulli"
        rows.append({"indicator": c, "n_distinct": nd, "modal_share": round(modal, 4),
                     "model": model})
    meta = pd.DataFrame(rows)
    groups = {m: meta.loc[meta.model == m, "indicator"].tolist()
              for m in ("gaussian", "bernoulli", "multinoulli")}
    return groups, meta


def encode_for_stepmix(df: pd.DataFrame, groups: dict) -> tuple[pd.DataFrame, dict]:
    """Dichotomise the Bernoulli block and recode categoricals to contiguous 0..L-1.

    StepMix's multinoulli expects integer codes starting at 0, so items like LATENCY
    (banded midpoints 8/23/45/90/150/210/270) must be re-coded rather than passed
    through as their band values.
    """
    out = df.copy()
    levels: dict[str, list] = {}
    for c in groups.get("bernoulli", []):
        vals = out[c].dropna().unique()
        if not set(vals) <= {0.0, 1.0}:
            out[c] = (out[c] > 0).astype(float).where(out[c].notna())
    for c in groups.get("multinoulli", []):
        cats = sorted(out[c].dropna().unique())
        levels[c] = [float(v) for v in cats]
        out[c] = out[c].map({v: i for i, v in enumerate(cats)}).astype(float)
    return out, levels


def _prorate(df: pd.DataFrame, cols: list[str], min_frac: float = 0.75) -> pd.Series:
    """Scale total, prorated for item non-response when >= min_frac items answered."""
    sub = df[cols]
    total = sub.mean(axis=1) * len(cols)
    return total.where(sub.notna().sum(axis=1) >= np.ceil(min_frac * len(cols)))


def _collapse_onehot(df: pd.DataFrame, bands: dict[str, list[str]]) -> pd.Series:
    """Fold a one-hot block back into a single integer-coded nominal variable."""
    order = list(bands)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for code, (_, cols) in enumerate(bands.items()):
        hit = df[cols].sum(axis=1, min_count=1) > 0
        out = out.mask(hit.fillna(False), float(code))
    all_missing = df[[c for cols in bands.values() for c in cols]].isna().all(axis=1)
    return out.mask(all_missing), order


def build_c2_scales(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Instrument scale scores + standalone items + collapsed chronotype nominals.

    FOSQ is reverse-scored to an *impairment* score so that, for every continuous
    indicator in the model, higher = worse. That makes the class profile plot
    readable in one direction instead of requiring the reader to remember which
    of 30 indicators runs backwards.
    """
    out = pd.DataFrame(index=df.index)
    out["ESS_total"] = _prorate(df, ESS)                       # 0-24 sleepiness
    out["ISI_total"] = _prorate(df, ISI)                       # 0-28 insomnia
    out["FOSQ_impair"] = 40.0 - _prorate(df, FOSQ)             # 0-40, reversed
    out["PHQ2_total"] = _prorate(df, PHQ)                      # 0-6 depressed mood
    out["STOP_total"] = _prorate(df, STOP)                     # 0-3
    out["RLS_count"] = _prorate(df, RLS)                       # 0-3
    out["SVS_mean"] = df[SVS].mean(axis=1)                     # 0-7 snoring volume
    for c in C2_SINGLE_ORDINAL:
        out[c] = df[c]
    for c in C2_SINGLE_BINARY:
        out[c] = df[c]
    for c in C2_CC:
        out[c] = df[c]
    bed, bed_order = _collapse_onehot(df, BEDTIME_BANDS)
    wake, wake_order = _collapse_onehot(df, WAKETIME_BANDS)
    out["BEDTIME_band"] = bed
    out["WAKETIME_band"] = wake

    # measurement model per indicator, decided from its own support rather than
    # from its nominal item type; the chronotype bands are nominal by construction
    groups, meta = assign_measurement_models(
        out, [c for c in out.columns],
        force_binary=C2_SINGLE_BINARY + C2_CC)
    for c in ("BEDTIME_band", "WAKETIME_band"):
        for g in groups.values():
            if c in g:
                g.remove(c)
        meta.loc[meta.indicator == c, "model"] = "multinoulli"
    groups["multinoulli"] += ["BEDTIME_band", "WAKETIME_band"]

    report = {
        "n_indicators": int(out.shape[1]),
        "groups": {k: len(v) for k, v in groups.items()},
        "bedtime_levels": bed_order,
        "waketime_levels": wake_order,
        "measurement_models": meta.to_dict("records"),
        "coverage": {c: round(float(out[c].notna().mean()), 4) for c in out.columns},
        "note": "FOSQ reverse-scored to impairment (40 - total) so higher = worse "
                "on every continuous indicator. Measurement model per indicator is "
                f"gaussian if n_distinct>={C2_GAUSS_MIN_DISTINCT} and modal share "
                f"<={C2_GAUSS_MAX_MODAL}, bernoulli if modal share "
                f">={C2_BERNOULLI_MIN_MODAL}, else multinoulli -- a Gaussian on an "
                "item with 75-90% of its mass on one response collapses onto that "
                "spike and makes the likelihood (and therefore BIC) meaningless.",
    }
    return out, groups, report


C2_INSTRUMENT_BLOCKS = {
    "ESS": ESS, "ISI": ISI, "FOSQ": FOSQ, "PHQ": PHQ, "STOP": STOP, "RLS": RLS,
    "SVS": SVS, "CC": C2_CC,
    "BEDTIME": [c for cols in BEDTIME_BANDS.values() for c in cols],
    "WAKETIME": [c for cols in WAKETIME_BANDS.values() for c in cols],
    "NARC": ["CATAPLEXY", "SLEEPATTACK", "SLEEPHALLUCINATION", "SLEEPPARALYSIS"],
    "MISC": ["GASP", "SLEEPY", "DROWSYDRIVING", "CLAUST", "MEDS", "PARTNER",
             "PARASOMNIA", "PTSD", "SHIFT", "LATENCY", "NAPS", "HEADACHE",
             "AWAKENINGS", "CAFFEINE", "SLEEPTIME"],
}


def c2_form_pattern(df: pd.DataFrame) -> pd.Series:
    """Which questionnaire blocks a patient has -- the intake-form-version signature.

    Missingness in c2 is block-structured by instrument availability (5 patterns
    cover 92.6% of the cohort). If the fitted classes line up with this signature,
    the "symptom phenotypes" are really form-version cohorts; script 04 reports the
    Cramer's V of class x form-pattern as an explicit artifact check.
    """
    avail = pd.DataFrame({
        b: (df[cols].notna().mean(axis=1) > 0.5).astype(int)
        for b, cols in C2_INSTRUMENT_BLOCKS.items()
    })
    return avail.apply(lambda r: "".join(map(str, r)), axis=1)


# ============================================================= c3 comorbidities
# Categories dropped from the chronic-disease indicator set. These are either not
# diseases (symptoms, administrative "other" codes), not chronic (acute injuries),
# sex/episode-specific (pregnancy), or dominated by routine screening rather than
# morbidity (refraction/vision under "sense organs").
C3_DROP_CATEGORIES = {
    "symptoms", "injuries & poisonings", "pregnancy complications", "other",
    "sense organs", "congenital anomalies",
}
C3_MIN_PREVALENCE = 0.02        # >= 2% of the cohort after roll-up

# Phecode roots that are CIRCULAR with the exposure or with another domain. The 327
# root absorbs 327.3 "Sleep apnea" (24%), 327.32 "Obstructive sleep apnea" (9%),
# 327.4/327.41 insomnia and 327.71 restless legs into one 43%-prevalence indicator.
# Every patient in this cohort has OSA by construction, so that indicator measures
# whether the diagnosis was billed, not whether the patient has it -- and its
# insomnia/RLS content would manufacture a c2-c3 association out of nothing.
C3_EXCLUDE_ROOTS = {"327"}

# --- sex-specific indicators ---------------------------------------------------
# The K=4 solution on the full indicator set split its two middle classes almost
# perfectly by patient sex (48.8% male-specific-coded vs 90.6% female-specific-coded
# at similar burden), because sex-specific billing codes encode sex even though sex
# is not a model input and is absent from data/. Removing them lets the model spend
# its classes on disease structure instead of recovering a demographic.
#
# The line drawn here is ANATOMY, not association:
#   dropped  - phecodes the phecode system itself restricts to one sex, anything
#              pregnancy-related, and the breast organ codes (610-613). The breast
#              codes are labelled "Both" upstream because males do get them, but
#              empirically they run 13-14x more common in female patients here.
#   retained - diseases that merely differ in incidence between sexes (gout 3.5x
#              male, migraine 3.5x female, UTI, hypothyroidism, iron-deficiency
#              anaemia, coronary atherosclerosis). These are real comorbidity signal;
#              dropping them would remove the structure the analysis is looking for.
PYPHEWAS_CODES = (r"C:\Users\natha\anaconda3\envs\myenv2\Lib\site-packages"
                  r"\pyPheWAS\resources\codes.csv")
C3_SEX_ORGAN_ROOTS = (
    [str(r) for r in range(600, 609)]      # male genital
    + [str(r) for r in range(610, 614)]    # breast
    + [str(r) for r in range(614, 630)]    # female genital
    + ["174", "185", "796"]                # breast / prostate cancer, PSA
)


def sex_specific_phecodes(codes_csv: str = PYPHEWAS_CODES) -> set[str]:
    """Phecodes the phecode system restricts to one sex, or flags as pregnancy-related."""
    cd = pd.read_csv(codes_csv, dtype=str)
    cd["phewas_code"] = cd.phewas_code.astype(str)
    by_sex = cd.groupby("phewas_code")["sex"].agg(
        lambda s: {v for v in s.dropna().unique()})
    preg = cd.groupby("phewas_code")["pregnancy_related"].max()
    out = {k for k, v in by_sex.items() if v - {"Both"}}
    out |= {k for k, v in preg.items() if str(v) == "1"}
    return out

# Process-of-care markers rather than diseases: they record that something was
# looked for or managed, which is the healthcare-contact signal this domain is
# already most at risk of measuring.
C3_EXCLUDE_LABEL_RE = (
    r"encounter|screen|personal history|family history|abnormal finding|"
    r"nonspecific (abnormal|finding)|other tests|examination|counsel")


def rollup_phecodes(X: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    """Collapse decimal phecodes onto their integer parent by logical OR.

    The pyPheWAS build maps each ICD to exactly one phecode with no hierarchy
    roll-up, so parent and child codes here are largely *disjoint*: empirically
    P(parent=1 | child=1) is only 0.27. That means a single clinical condition is
    fragmented across sibling codes (401 vs 401.1 vs 401.2) rather than
    double-counted, and the fix is to take the union at the integer level.
    """
    roots: dict[str, list[str]] = {}
    for c in feats:
        roots.setdefault(c.split(".")[0], []).append(c)
    out = pd.DataFrame(
        {r: X[cols].max(axis=1).astype(np.int8) for r, cols in sorted(roots.items())},
        index=X.index,
    )
    return out, {"n_before": len(feats), "n_after": out.shape[1],
                 "members": {r: cols for r, cols in sorted(roots.items())}}


def select_c3_indicators(rolled: pd.DataFrame, labels: pd.DataFrame,
                         drop_categories=C3_DROP_CATEGORIES,
                         min_prev=C3_MIN_PREVALENCE,
                         drop_sex_specific: bool = True) -> tuple[list[str], pd.DataFrame]:
    """Apply the chronic-disease filter + prevalence floor to the rolled-up codes.

    A root inherits its category from its own phecode entry when one exists,
    otherwise from its most prevalent child.
    """
    cat = dict(zip(labels.phecode, labels.category))
    name = dict(zip(labels.phecode, labels.phenotype))
    prev_child = dict(zip(labels.phecode, labels.prevalence))

    rows = []
    for root in rolled.columns:
        kids = [p for p in labels.phecode if str(p).split(".")[0] == root]
        best = max(kids, key=lambda p: prev_child.get(p, 0)) if kids else None
        rows.append({
            "root": root,
            "category": cat.get(root) or (cat.get(best) if best else None),
            "label": name.get(root) or (f"{name.get(best)} (group)" if best else root),
            "prevalence": float(rolled[root].mean()),
            "n_children": len(kids),
        })
    meta = pd.DataFrame(rows)
    meta["drop_reason"] = ""
    meta.loc[meta.category.isna(), "drop_reason"] = "unlabelled"
    meta.loc[meta.category.isin(drop_categories), "drop_reason"] = "non-chronic category"
    meta.loc[meta.root.str.fullmatch(r"10\d\d"), "drop_reason"] = "administrative 1000-series"
    meta.loc[meta.label.str.contains(C3_EXCLUDE_LABEL_RE, case=False, na=False),
             "drop_reason"] = "process-of-care marker"
    meta.loc[meta.root.isin(C3_EXCLUDE_ROOTS), "drop_reason"] = "circular with OSA exposure"
    if drop_sex_specific:
        restricted = sex_specific_phecodes()
        kids: dict[str, list[str]] = {}
        for pc in labels.phecode.dropna():
            kids.setdefault(str(pc).split(".")[0], []).append(str(pc))
        sex_roots = {r for r in meta.root
                     if r in C3_SEX_ORGAN_ROOTS
                     or any(k in restricted for k in kids.get(r, []))}
        meta.loc[meta.root.isin(sex_roots), "drop_reason"] = "sex-specific"
    meta.loc[(meta.drop_reason == "") & (meta.prevalence < min_prev),
             "drop_reason"] = f"prevalence < {min_prev:.0%}"
    meta["kept"] = meta.drop_reason == ""
    return meta.loc[meta.kept, "root"].tolist(), meta


def c3_utilisation_index(X: pd.DataFrame, feats: list[str]) -> pd.Series:
    """Distinct phecode count -- a proxy for healthcare contact, NEVER a model input.

    The prior clustering run found the high-multimorbidity cluster had *lower* AHI,
    i.e. these indicators partly measure how much care a patient receives rather
    than how sick they are. Carrying this alongside lets script 06 label a
    high-utilisation class as such instead of over-reading it as multimorbidity.
    """
    return X[feats].sum(axis=1).astype(int)
