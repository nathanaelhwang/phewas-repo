# %% [markdown]
# Step 10 - LabWAS association testing.
# Per (lab x target x window x contrast):
#   Unadjusted screen: Kruskal-Wallis (omnibus 4-group), Spearman trend (severity_ord),
#     Mann-Whitney (Mild/Moderate/Severe-vs-None, AHI>=5, AHI>=15).
#   Adjusted (nested M1 crude / M3 +demo+smk+SES+obs_window / M4 +BMI):
#     value targets (median, mean) -> OLS on rank-inverse-normal-transformed target;
#     rate target (order_rate)     -> Poisson GLM on count, offset log(obs_years), robust SE.
# FDR (BH) + Bonferroni across labs within each (target x window x contrast x model) family.
# Output: analysis/results_{1yr,5yr}.csv.  DE-IDENTIFIED aggregates only.
# Usage: python 10_run_labwas.py [--labs N] [--jobs K]

# %%
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

import labwas_utils as U

warnings.simplefilter("ignore")
V2_COHORT = (U.PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()
ANALYSIS = U.RESULTS / "analysis"
WINDOWS = [("1yr", 1), ("5yr", 5)]
TARGETS = ["order_rate", "median", "mean"]
MIN_GRP = 20  # min patients per group for a contrast

# covariate blocks (formula fragments). obs_window excluded for the rate (it's the offset).
COV_VALUE = {
    "m1": "",
    "m3": " + age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp) + obs_window_z",
    "m4": " + age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp) + obs_window_z + bmi_z",
}
COV_RATE = {
    "m1": "",
    "m3": " + age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp)",
    "m4": " + age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp) + bmi_z",
}
SEV = "C(severity, Treatment(reference='None'))"
PAIRS = [("mild_vs_none", "Mild"), ("moderate_vs_none", "Moderate"), ("severe_vs_none", "Severe")]


def int_transform(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x, method="average")
    return stats.norm.ppf((r - 0.5) / len(r))


def prep_cohort() -> pd.DataFrame:
    c = pd.read_parquet(V2_COHORT)
    for col in ["age", "bmi", "obs_window"]:
        c[col + "_z"] = (c[col] - c[col].mean()) / c[col].std()
    c["ahi_ge5"] = (c["AHI"] >= 5).astype(int)
    c["ahi_ge15"] = (c["AHI"] >= 15).astype(int)
    keep = ["id", "severity", "severity_ord", "ahi_ge5", "ahi_ge15", "age_z", "sex_M",
            "bmi_z", "obs_window_z", "obs_window", "study_year", "RACE_ETH",
            "smoking_status", "income_gp"]
    return c[keep]


def _row(target, contrast, model, n, etype, effect, se, lo, hi, p, note=""):
    return dict(target=target, contrast=contrast, model=model, n=int(n), effect_type=etype,
                effect=effect, se=se, ci_low=lo, ci_high=hi, p=p,
                neglog10p=(-np.log10(p) if (p and p > 0) else np.nan),
                unstable=note)


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def _mwu(g, ref, target, cname):
    if len(ref) < MIN_GRP or len(g) < MIN_GRP:
        return None
    r = _safe(lambda: stats.mannwhitneyu(g, ref, alternative="two-sided"))
    if r is None:
        return None
    U_, p = r
    rbc = 2 * U_ / (len(g) * len(ref)) - 1   # rank-biserial effect size
    return _row(target, cname, "unadjusted", len(g) + len(ref), "rank_biserial", rbc, np.nan, np.nan, np.nan, p)


def unadjusted(df: pd.DataFrame, target: str) -> list:
    y = df[target].to_numpy(dtype=float)
    grp = df["severity_ord"].to_numpy()
    out = []
    groups = [y[grp == k] for k in range(4)]
    if all(len(g) >= MIN_GRP for g in groups):
        r = _safe(lambda: stats.kruskal(*groups))
        if r is not None:
            H, p = r
            eps2 = (H - 3) / (len(y) - 4) if len(y) > 4 else np.nan   # epsilon-squared
            out.append(_row(target, "omnibus", "unadjusted", len(y), "epsilon2", eps2, np.nan, np.nan, np.nan, p))
    if len(np.unique(grp)) >= 3:
        r = _safe(lambda: stats.spearmanr(y, grp))
        if r is not None and np.isfinite(r[1]):
            out.append(_row(target, "trend", "unadjusted", len(y), "spearman_rho", r[0], np.nan, np.nan, np.nan, r[1]))
    none = y[grp == 0]
    for cname, lab in PAIRS:
        row = _mwu(y[grp == {"Mild": 1, "Moderate": 2, "Severe": 3}[lab]], none, target, cname)
        if row:
            out.append(row)
    for cname in ["ahi_ge5", "ahi_ge15"]:
        col = df[cname].to_numpy()
        row = _mwu(y[col == 1], y[col == 0], target, cname)
        if row:
            out.append(row)
    return out


def _fit(df, formula, is_rate):
    if is_rate:
        m = smf.glm(formula, data=df, family=sm.families.Poisson(),
                    offset=np.log(df["obs_years"].to_numpy()))
        return m.fit(cov_type="HC0")
    return smf.ols(formula, data=df).fit()


def _eff(res, name, is_rate):
    b = res.params[name]; se = res.bse[name]; p = res.pvalues[name]
    ci = res.conf_int().loc[name]
    if is_rate:  # report IRR
        return ("IRR", float(np.exp(b)), float(se), float(np.exp(ci[0])), float(np.exp(ci[1])), float(p))
    return ("beta", float(b), float(se), float(ci[0]), float(ci[1]), float(p))


def adjusted(df: pd.DataFrame, target: str) -> list:
    is_rate = (target == "order_rate")
    covs = COV_RATE if is_rate else COV_VALUE
    dep = "count" if is_rate else "y_int"
    if not is_rate:
        df = df.assign(y_int=int_transform(df[target].to_numpy(dtype=float)))
    out = []
    grp = df["severity_ord"].to_numpy()
    have_sev = all((grp == k).sum() >= MIN_GRP for k in range(4))
    for model, cov in covs.items():
        # factor model -> omnibus + pairwise
        if have_sev:
            try:
                res = _fit(df, f"{dep} ~ {SEV}{cov}", is_rate)
                dummies = [n for n in res.params.index if "severity" in n and "T." in n]
                # omnibus joint test
                hyp = ", ".join(f"{n} = 0" for n in dummies)
                wt = res.wald_test(hyp, scalar=True)
                pom = float(wt.pvalue); stat = float(np.ravel(wt.statistic)[0])
                out.append(_row(target, "omnibus", model, len(df), "wald_chi2" if is_rate else "wald_F", stat, np.nan, np.nan, np.nan, pom))
                for cname, lab in PAIRS:
                    nm = [n for n in dummies if lab in n]
                    if nm:
                        et, e, se, lo, hi, p = _eff(res, nm[0], is_rate)
                        out.append(_row(target, cname, model, len(df), et, e, se, lo, hi, p))
            except Exception as ex:
                out.append(_row(target, "omnibus", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        # trend
        try:
            res = _fit(df, f"{dep} ~ severity_ord{cov}", is_rate)
            et, e, se, lo, hi, p = _eff(res, "severity_ord", is_rate)
            out.append(_row(target, "trend", model, len(df), et, e, se, lo, hi, p))
        except Exception as ex:
            out.append(_row(target, "trend", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        # dichotomized AHI
        for cname in ["ahi_ge5", "ahi_ge15"]:
            if df[cname].nunique() == 2 and df[cname].sum() >= MIN_GRP and (len(df) - df[cname].sum()) >= MIN_GRP:
                try:
                    res = _fit(df, f"{dep} ~ {cname}{cov}", is_rate)
                    et, e, se, lo, hi, p = _eff(res, cname, is_rate)
                    out.append(_row(target, cname, model, len(df), et, e, se, lo, hi, p))
                except Exception as ex:
                    out.append(_row(target, cname, model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
    return out


def run_lab(loinc, lab_df, cohort, obs_years_col):
    df = lab_df.merge(cohort, on="id", how="inner")
    df = df.assign(obs_years=df[obs_years_col])
    rows = []
    for t in TARGETS:
        d = df[df[t].notna() & np.isfinite(df[t])]
        if len(d) < 4 * MIN_GRP or d[t].nunique() < 2:   # skip empty / zero-variance targets
            continue
        try:
            rows += unadjusted(d, t)
            rows += adjusted(d, t)
        except Exception as ex:
            rows.append(_row(t, "omnibus", "unadjusted", len(d), "na", np.nan, np.nan, np.nan, np.nan, np.nan, f"lab_error:{str(ex)[:30]}"))
    for r in rows:
        r["loinc_code"] = loinc
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--labs", type=int, default=0); ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    cohort = prep_cohort()
    cat = pd.read_csv(U.RESULTS / "loinc_category.csv")[["loinc_code", "category"]]

    for label, years in WINDOWS:
        tg = pd.read_parquet(ANALYSIS / f"targets_{label}.parquet")
        # obs_years denominator (LEAST(obs_window, N)); >=1 for all so 1yr -> 1
        oy = pd.read_parquet(V2_COHORT, columns=["id", "obs_window"])
        oy["obs_years"] = np.minimum(oy["obs_window"], years).clip(lower=1.0 / 365)
        tg = tg.merge(oy[["id", "obs_years"]], on="id", how="left")
        labs = sorted(tg["loinc_code"].unique())
        if a.labs:
            labs = labs[:a.labs]
        print(f"[{label}] {len(labs)} labs, {len(tg):,} rows", flush=True)

        from joblib import Parallel, delayed
        grouped = {k: v for k, v in tg.groupby("loinc_code")}
        results = Parallel(n_jobs=a.jobs, verbose=5)(
            delayed(run_lab)(lc, grouped[lc], cohort, "obs_years") for lc in labs)
        rows = [r for sub in results for r in sub]
        res = pd.DataFrame(rows)
        res["window"] = label
        res = res.merge(cat, on="loinc_code", how="left")

        # FDR (BH) + Bonferroni across labs within each (target, contrast, model) family
        res["sig_bon"] = False; res["sig_fdr"] = False
        for _, idx in res[res["p"].notna()].groupby(["target", "contrast", "model"]).groups.items():
            sub = res.loc[idx]; n = len(sub)
            res.loc[idx, "sig_bon"] = sub["p"] < 0.05 / n
            order = sub["p"].sort_values(); ranks = np.arange(1, n + 1)
            crit = 0.05 * ranks / n; passed = order.to_numpy() <= crit
            thr = order.to_numpy()[passed].max() if passed.any() else -1
            res.loc[idx, "sig_fdr"] = sub["p"] <= thr

        cols = ["window", "target", "loinc_code", "category", "contrast", "model", "n",
                "effect_type", "effect", "se", "ci_low", "ci_high", "p", "neglog10p",
                "sig_bon", "sig_fdr", "unstable"]
        out = ANALYSIS / f"results_{label}.csv"
        res[cols].to_csv(out, index=False)
        nsig = res.groupby("model")["sig_fdr"].sum()
        print(f"[{label}] {len(res):,} tests -> {out.name}; FDR-sig by model:\n{nsig.to_string()}", flush=True)

    U.log_milestone("Step 10 LabWAS: association results written (analysis/results_{1yr,5yr}.csv).")


if __name__ == "__main__":
    main()
