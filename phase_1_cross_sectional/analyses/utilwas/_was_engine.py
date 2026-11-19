"""Shared UtilWAS association engine.

Unadjusted screen + nested adjusted models (M1 crude / M3 +demo+SES+smk / M4 +BMI) across the seven
OSA-severity contrasts (omnibus 4-group Wald, trend on severity_ord, Mild/Moderate/Severe-vs-None,
AHI>=5, AHI>=15), with FDR (BH) + Bonferroni per family. Mirrors analyses/labwas/10_run_labwas.py.
Two outcome kinds:
  RATE   (encounters/yr) -> Poisson GLM on count, offset log(obs_years), robust HC0 SE, IRR;
          unadjusted = Kruskal-Wallis / Spearman-trend / Mann-Whitney on the per-patient rate.
  BINARY (0/1 presence)  -> logistic GLM, OR; unadjusted = chi-square omnibus / Cochran-Armitage
          trend / 2x2 crude OR (Fisher) pairwise & AHI.
DE-IDENTIFIED aggregates only (no patient rows).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utilwas_utils as U

MIN_GRP = 20
PAIRS = [("mild_vs_none", "Mild"), ("moderate_vs_none", "Moderate"), ("severe_vs_none", "Severe")]
SEV = "C(severity, Treatment(reference='None'))"
ORD = {"Mild": 1, "Moderate": 2, "Severe": 3}
_DEMO = "age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp)"
COV_RATE = {"m1": "", "m3": f" + {_DEMO}", "m4": f" + {_DEMO} + bmi_z"}
# obs_window is a covariate for binary (exposure-time proxy); for rate it is the offset instead.
COV_BIN = {"m1": "", "m3": f" + {_DEMO} + obs_window_z", "m4": f" + {_DEMO} + obs_window_z + bmi_z"}


def prep_cohort() -> pd.DataFrame:
    c = pd.read_parquet(U.COHORT)
    for col in ["age", "bmi", "obs_window"]:
        c[col + "_z"] = (c[col] - c[col].mean()) / c[col].std()
    c["ahi_ge5"] = (c["AHI"] >= 5).astype(int)
    c["ahi_ge15"] = (c["AHI"] >= 15).astype(int)
    keep = ["id", "severity", "severity_ord", "ahi_ge5", "ahi_ge15", "age_z", "sex_M",
            "bmi_z", "obs_window_z", "obs_window", "study_year", "RACE_ETH", "smoking_status", "income_gp"]
    return c[keep].copy()


def obs_years(cohort: pd.DataFrame, years: int) -> pd.Series:
    return np.minimum(cohort["obs_window"], years).clip(lower=1.0 / 365)


_COH_CACHE = None


def cohort_cached() -> pd.DataFrame:
    """Load-once-per-process cohort (so joblib workers don't re-pickle it per task)."""
    global _COH_CACHE
    if _COH_CACHE is None:
        _COH_CACHE = prep_cohort()
    return _COH_CACHE


def row(feature, part, contrast, model, n, etype, effect, se, lo, hi, p, note=""):
    return dict(feature=feature, part=part, contrast=contrast, model=model, n=int(n), effect_type=etype,
                effect=effect, se=se, ci_low=lo, ci_high=hi, p=p,
                neglog10p=(-np.log10(p) if (p and p > 0) else np.nan), unstable=note)


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


# ---------------- unadjusted ----------------
def _mwu(g, ref, feature, part, cname):
    if len(ref) < MIN_GRP or len(g) < MIN_GRP:
        return None
    r = _safe(lambda: stats.mannwhitneyu(g, ref, alternative="two-sided"))
    if r is None:
        return None
    U_, p = r
    rbc = 2 * U_ / (len(g) * len(ref)) - 1
    return row(feature, part, cname, "unadjusted", len(g) + len(ref), "rank_biserial", rbc, np.nan, np.nan, np.nan, p)


def rate_unadjusted(df, ycol, feature, part="rate"):
    y = df[ycol].to_numpy(dtype=float)
    grp = df["severity_ord"].to_numpy()
    out = []
    groups = [y[grp == k] for k in range(4)]
    if all(len(g) >= MIN_GRP for g in groups):
        r = _safe(lambda: stats.kruskal(*groups))
        if r is not None:
            H, p = r
            eps2 = (H - 3) / (len(y) - 4) if len(y) > 4 else np.nan
            out.append(row(feature, part, "omnibus", "unadjusted", len(y), "epsilon2", eps2, np.nan, np.nan, np.nan, p))
    if len(np.unique(grp)) >= 3:
        r = _safe(lambda: stats.spearmanr(y, grp))
        if r is not None and np.isfinite(r[1]):
            out.append(row(feature, part, "trend", "unadjusted", len(y), "spearman_rho", r[0], np.nan, np.nan, np.nan, r[1]))
    none = y[grp == 0]
    for cname, lab in PAIRS:
        rr = _mwu(y[grp == ORD[lab]], none, feature, part, cname)
        if rr:
            out.append(rr)
    for cname in ["ahi_ge5", "ahi_ge15"]:
        col = df[cname].to_numpy()
        rr = _mwu(y[col == 1], y[col == 0], feature, part, cname)
        if rr:
            out.append(rr)
    return out


def _ca_trend(binary, scores):
    N = len(binary); M = float(binary.sum())
    if M == 0 or M == N:
        return np.nan, np.nan
    z = scores.astype(float)
    T = float((binary * z).sum()); sz, sz2 = float(z.sum()), float((z * z).sum())
    var = M * (N - M) / (N * N * (N - 1)) * (N * sz2 - sz * sz)
    if var <= 0:
        return np.nan, np.nan
    Z = (T - M / N * sz) / np.sqrt(var)
    return Z, 2 * stats.norm.sf(abs(Z))


def binary_unadjusted(df, ycol, feature, part="presence"):
    y = df[ycol].to_numpy()
    grp = df["severity_ord"].to_numpy()
    out = []
    # omnibus: chi-square severity(4) x outcome(2), effect = Cramer's V
    tab = pd.crosstab(grp, y)
    if tab.shape == (4, 2) and (tab.values.min() >= 5):
        r = _safe(lambda: stats.chi2_contingency(tab))
        if r is not None:
            chi2, p = r[0], r[1]
            v = np.sqrt(chi2 / (len(y) * 1))
            out.append(row(feature, part, "omnibus", "unadjusted", len(y), "cramers_v", v, np.nan, np.nan, np.nan, p))
    # trend: Cochran-Armitage
    z, p = _ca_trend(y, grp)
    if np.isfinite(p):
        out.append(row(feature, part, "trend", "unadjusted", len(y), "ca_z", z, np.nan, np.nan, np.nan, p))
    # pairwise + AHI: crude OR from 2x2
    def crude_or(mask_pos, mask_ref, cname):
        a = int(((y == 1) & mask_pos).sum()); b = int(((y == 0) & mask_pos).sum())
        cc = int(((y == 1) & mask_ref).sum()); d = int(((y == 0) & mask_ref).sum())
        if min(mask_pos.sum(), mask_ref.sum()) < MIN_GRP or min(a, b, cc, d) == 0:
            return None
        orr = (a * d) / (b * cc)
        se = np.sqrt(1 / a + 1 / b + 1 / cc + 1 / d)
        rr = _safe(lambda: stats.fisher_exact([[a, b], [cc, d]]))
        p = rr[1] if rr else np.nan
        return row(feature, part, cname, "unadjusted", int(mask_pos.sum() + mask_ref.sum()), "OR",
                   orr, se, float(np.exp(np.log(orr) - 1.96 * se)), float(np.exp(np.log(orr) + 1.96 * se)), p)
    none = (grp == 0)
    for cname, lab in PAIRS:
        r = crude_or(grp == ORD[lab], none, cname)
        if r:
            out.append(r)
    for cname in ["ahi_ge5", "ahi_ge15"]:
        col = df[cname].to_numpy()
        r = crude_or(col == 1, col == 0, cname)
        if r:
            out.append(r)
    return out


# ---------------- adjusted ----------------
def _fit_rate(df, formula):
    return smf.glm(formula, data=df, family=sm.families.Poisson(),
                   offset=np.log(df["obs_years"].to_numpy())).fit(cov_type="HC0")


def _fit_logit(df, formula):
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit()


def _eff_exp(res, name):
    b = res.params[name]; se = res.bse[name]; p = res.pvalues[name]; ci = res.conf_int().loc[name]
    return (float(np.exp(b)), float(se), float(np.exp(ci[0])), float(np.exp(ci[1])), float(p))


def _adjusted(df, feature, part, covs, dep, fit, etype, wtype):
    out = []
    grp = df["severity_ord"].to_numpy()
    have_sev = all((grp == k).sum() >= MIN_GRP for k in range(4))
    for model, cov in covs.items():
        if have_sev:
            try:
                res = fit(df, f"{dep} ~ {SEV}{cov}")
                dummies = [n for n in res.params.index if "severity" in n and "T." in n]
                wt = res.wald_test(", ".join(f"{n} = 0" for n in dummies), scalar=True)
                out.append(row(feature, part, "omnibus", model, len(df), wtype,
                               float(np.ravel(wt.statistic)[0]), np.nan, np.nan, np.nan, float(wt.pvalue)))
                for cname, lab in PAIRS:
                    nm = [n for n in dummies if lab in n]
                    if nm:
                        e, se, lo, hi, p = _eff_exp(res, nm[0])
                        out.append(row(feature, part, cname, model, len(df), etype, e, se, lo, hi, p))
            except Exception as ex:
                out.append(row(feature, part, "omnibus", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        try:
            res = fit(df, f"{dep} ~ severity_ord{cov}")
            e, se, lo, hi, p = _eff_exp(res, "severity_ord")
            out.append(row(feature, part, "trend", model, len(df), etype, e, se, lo, hi, p))
        except Exception as ex:
            out.append(row(feature, part, "trend", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        for cname in ["ahi_ge5", "ahi_ge15"]:
            if df[cname].nunique() == 2 and df[cname].sum() >= MIN_GRP and (len(df) - df[cname].sum()) >= MIN_GRP:
                try:
                    res = fit(df, f"{dep} ~ {cname}{cov}")
                    e, se, lo, hi, p = _eff_exp(res, cname)
                    out.append(row(feature, part, cname, model, len(df), etype, e, se, lo, hi, p))
                except Exception as ex:
                    out.append(row(feature, part, cname, model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
    return out


def rate_adjusted(df, feature, part="rate"):
    return _adjusted(df, feature, part, COV_RATE, "n_enc", _fit_rate, "IRR", "wald_chi2")


def binary_adjusted(df, feature, part="presence"):
    return _adjusted(df, feature, part, COV_BIN, "y", _fit_logit, "OR", "wald_chi2")


def group_descriptives(df, ycol, feature, part):
    """Per-severity mean/median of ycol over the given df (population if df=full cohort)."""
    out = []
    for g, sub in df.groupby("severity"):
        v = sub[ycol].to_numpy(dtype=float)
        out.append(dict(feature=feature, part=part, severity=g, n=len(v),
                        mean=round(float(v.mean()), 4), median=float(np.median(v))))
    return out


def add_fdr_bonferroni(res: pd.DataFrame, group_cols) -> pd.DataFrame:
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
