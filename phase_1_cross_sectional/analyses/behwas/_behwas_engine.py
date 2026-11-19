"""Shared BehWAS association engine (adapted from analyses/utilwas/_was_engine.py
and the OLS-on-INT path in analyses/labwas/10_run_labwas.py).

Every behavioral outcome is tested against the seven OSA-severity contrasts (omnibus
4-group Wald, trend on severity_ord, Mild/Moderate/Severe-vs-None, AHI>=5, AHI>=15)
under the nested adjusted models M1 crude / M3 (+demo+SES+smoking+obs_window) /
M4 (+BMI), plus an unadjusted screen, with BH-FDR + Bonferroni per family.

Two outcome kinds -- BehWAS stores BOTH forms of every feature so the forest-plot
effect scale can be chosen late:
  BINARY   (0/1)      -> logistic GLM, OR; unadjusted = chi2 omnibus / Cochran-Armitage
                         trend / 2x2 crude OR (Fisher) pairwise & AHI.
  CONTINUOUS (real)   -> OLS on rank-inverse-normal-transformed value, beta; unadjusted =
                         Kruskal-Wallis / Spearman-trend / Mann-Whitney.

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
import behwas_utils as U

MIN_GRP = 20  # min patients per group for a contrast
PAIRS = [("mild_vs_none", "Mild"), ("moderate_vs_none", "Moderate"), ("severe_vs_none", "Severe")]
ORD = {"Mild": 1, "Moderate": 2, "Severe": 3}
SEV = "C(severity, Treatment(reference='None'))"
_DEMO = "age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp)"
# One covariate ladder for both kinds: obs_window is a covariate (exposure-time proxy),
# not an offset (BehWAS has no rate/Poisson outcome). Callers may pass their own ladder
# via the `covs=` argument of the *_adjusted functions (QWAS supplies an m1-m4 version).
COV = {"m1": "", "m3": f" + {_DEMO} + obs_window_z", "m4": f" + {_DEMO} + obs_window_z + bmi_z"}


def int_transform(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x, method="average")
    return stats.norm.ppf((r - 0.5) / len(r))


def prep_cohort() -> pd.DataFrame:
    c = pd.read_parquet(U.COHORT)
    for col in ["age", "bmi", "obs_window"]:
        c[col + "_z"] = (c[col] - c[col].mean()) / c[col].std()
    c["ahi_ge5"] = (c["AHI"] >= 5).astype(int)
    c["ahi_ge15"] = (c["AHI"] >= 15).astype(int)
    keep = ["id", "severity", "severity_ord", "ahi_ge5", "ahi_ge15", "age_z", "sex_M",
            "bmi_z", "obs_window_z", "obs_window", "study_year", "RACE_ETH",
            "smoking_status", "income_gp"]
    return c[keep].copy()


def row(outcome, contrast, model, n, etype, effect, se, lo, hi, p, note=""):
    return dict(outcome=outcome, contrast=contrast, model=model, n=int(n), effect_type=etype,
                effect=effect, se=se, ci_low=lo, ci_high=hi, p=p,
                neglog10p=(-np.log10(p) if (p and p > 0) else np.nan), unstable=note)


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


# ---------------- unadjusted: continuous ----------------
def _mwu(g, ref, outcome, cname):
    if len(ref) < MIN_GRP or len(g) < MIN_GRP:
        return None
    r = _safe(lambda: stats.mannwhitneyu(g, ref, alternative="two-sided"))
    if r is None:
        return None
    U_, p = r
    rbc = 2 * U_ / (len(g) * len(ref)) - 1  # rank-biserial effect size
    return row(outcome, cname, "unadjusted", len(g) + len(ref), "rank_biserial", rbc, np.nan, np.nan, np.nan, p)


def cont_unadjusted(df, ycol, outcome):
    y = df[ycol].to_numpy(dtype=float)
    grp = df["severity_ord"].to_numpy()
    out = []
    groups = [y[grp == k] for k in range(4)]
    if all(len(g) >= MIN_GRP for g in groups):
        r = _safe(lambda: stats.kruskal(*groups))
        if r is not None:
            H, p = r
            eps2 = (H - 3) / (len(y) - 4) if len(y) > 4 else np.nan  # epsilon-squared
            out.append(row(outcome, "omnibus", "unadjusted", len(y), "epsilon2", eps2, np.nan, np.nan, np.nan, p))
    if len(np.unique(grp)) >= 3:
        r = _safe(lambda: stats.spearmanr(y, grp))
        if r is not None and np.isfinite(r[1]):
            out.append(row(outcome, "trend", "unadjusted", len(y), "spearman_rho", r[0], np.nan, np.nan, np.nan, r[1]))
    none = y[grp == 0]
    for cname, lab in PAIRS:
        rr = _mwu(y[grp == ORD[lab]], none, outcome, cname)
        if rr:
            out.append(rr)
    for cname in ["ahi_ge5", "ahi_ge15"]:
        col = df[cname].to_numpy()
        rr = _mwu(y[col == 1], y[col == 0], outcome, cname)
        if rr:
            out.append(rr)
    return out


# ---------------- unadjusted: binary ----------------
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


def binary_unadjusted(df, ycol, outcome):
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
            out.append(row(outcome, "omnibus", "unadjusted", len(y), "cramers_v", v, np.nan, np.nan, np.nan, p))
    # trend: Cochran-Armitage
    z, p = _ca_trend(y, grp)
    if np.isfinite(p):
        out.append(row(outcome, "trend", "unadjusted", len(y), "ca_z", z, np.nan, np.nan, np.nan, p))

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
        return row(outcome, cname, "unadjusted", int(mask_pos.sum() + mask_ref.sum()), "OR",
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


# ---------------- adjusted (shared across kinds) ----------------
def _fit_logit(df, formula):
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit()


def _fit_ols(df, formula):
    return smf.ols(formula, data=df).fit()


def _eff_exp(res, name):
    """OR / IRR: exponentiate coefficient + CI."""
    b = res.params[name]; se = res.bse[name]; p = res.pvalues[name]; ci = res.conf_int().loc[name]
    return (float(np.exp(b)), float(se), float(np.exp(ci[0])), float(np.exp(ci[1])), float(p))


def _eff_raw(res, name):
    """beta: coefficient + CI as-is."""
    b = res.params[name]; se = res.bse[name]; p = res.pvalues[name]; ci = res.conf_int().loc[name]
    return (float(b), float(se), float(ci[0]), float(ci[1]), float(p))


def _adjusted(df, outcome, dep, fit, eff_fn, etype, wtype, covs=None):
    out = []
    grp = df["severity_ord"].to_numpy()
    have_sev = all((grp == k).sum() >= MIN_GRP for k in range(4))
    for model, cov in (covs or COV).items():
        # factor severity -> omnibus Wald + 3 pairwise
        if have_sev:
            try:
                res = fit(df, f"{dep} ~ {SEV}{cov}")
                dummies = [n for n in res.params.index if "severity" in n and "T." in n]
                wt = res.wald_test(", ".join(f"{n} = 0" for n in dummies), scalar=True)
                out.append(row(outcome, "omnibus", model, len(df), wtype,
                               float(np.ravel(wt.statistic)[0]), np.nan, np.nan, np.nan, float(wt.pvalue)))
                for cname, lab in PAIRS:
                    nm = [n for n in dummies if lab in n]
                    if nm:
                        e, se, lo, hi, p = eff_fn(res, nm[0])
                        out.append(row(outcome, cname, model, len(df), etype, e, se, lo, hi, p))
            except Exception as ex:
                out.append(row(outcome, "omnibus", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        # ordinal trend
        try:
            res = fit(df, f"{dep} ~ severity_ord{cov}")
            e, se, lo, hi, p = eff_fn(res, "severity_ord")
            out.append(row(outcome, "trend", model, len(df), etype, e, se, lo, hi, p))
        except Exception as ex:
            out.append(row(outcome, "trend", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        # dichotomized AHI
        for cname in ["ahi_ge5", "ahi_ge15"]:
            if df[cname].nunique() == 2 and df[cname].sum() >= MIN_GRP and (len(df) - df[cname].sum()) >= MIN_GRP:
                try:
                    res = fit(df, f"{dep} ~ {cname}{cov}")
                    e, se, lo, hi, p = eff_fn(res, cname)
                    out.append(row(outcome, cname, model, len(df), etype, e, se, lo, hi, p))
                except Exception as ex:
                    out.append(row(outcome, cname, model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
    return out


def binary_adjusted(df, ycol, outcome, covs=None):
    d = df.assign(y=df[ycol].astype(float))
    return _adjusted(d, outcome, "y", _fit_logit, _eff_exp, "OR", "wald_chi2", covs)


def cont_adjusted(df, ycol, outcome, covs=None):
    d = df.assign(y_int=int_transform(df[ycol].to_numpy(dtype=float)))
    return _adjusted(d, outcome, "y_int", _fit_ols, _eff_raw, "beta", "wald_F", covs)


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
