# %% [markdown]
# Step 12 - Two-part (hurdle) model, PART 1: ordering PROPENSITY (extensive margin).
# Complements the order_rate analysis (Part 2, intensive margin, tested-only).
# For each lab x window: outcome = ordered (1 if the patient has >=1 of that lab in the
# window else 0) over the FULL v2 cohort (zeros = never-tested).
#   Unadjusted: chi-square (omnibus 4-group, Cramer's V), Cochran-Armitage (trend),
#     2x2 chi-square + unadjusted OR (Mild/Moderate/Severe-vs-None, AHI>=5, AHI>=15).
#   Adjusted logistic (M1 crude / M3 +demo+smk+SES+obs_window / M4 +BMI): OR per contrast.
# FDR (BH) + Bonferroni across labs per (contrast x model). obs_window kept as a covariate
# (longer observation -> more chance of any order). Ubiquitous labs (prev>0.99) flagged.
# Output: analysis/results_propensity_{1yr,5yr}.csv + analysis/twopart_{1yr,5yr}.csv.
# Usage: python 12_order_propensity.py [--labs N] [--jobs K]

# %%
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
from scipy import stats

import labwas_utils as U

warnings.simplefilter("ignore")
V2_COHORT = (U.PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()
ANALYSIS = U.RESULTS / "analysis"
WINDOWS = ["1yr", "5yr"]
MIN_GRP = 20
COV = {
    "m1": "",
    "m3": " + age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp) + obs_window_z",
    "m4": " + age_z + sex_M + C(RACE_ETH) + C(study_year) + C(smoking_status) + C(income_gp) + obs_window_z + bmi_z",
}
SEV = "C(severity, Treatment(reference='None'))"
PAIRS = [("mild_vs_none", 1), ("moderate_vs_none", 2), ("severe_vs_none", 3)]
_COH = None  # per-worker global (loaded in initializer)


def prep_cohort() -> pd.DataFrame:
    c = pd.read_parquet(V2_COHORT)
    for col in ["age", "bmi", "obs_window"]:
        c[col + "_z"] = (c[col] - c[col].mean()) / c[col].std()
    c["ahi_ge5"] = (c["AHI"] >= 5).astype(int)
    c["ahi_ge15"] = (c["AHI"] >= 15).astype(int)
    return c[["id", "severity", "severity_ord", "ahi_ge5", "ahi_ge15", "age_z", "sex_M",
              "bmi_z", "obs_window_z", "study_year", "RACE_ETH", "smoking_status", "income_gp"]]


def _init(_):
    global _COH
    _COH = prep_cohort()


def _row(contrast, model, n, etype, effect, se, lo, hi, p, prev, note=""):
    return dict(contrast=contrast, model=model, n=int(n), effect_type=etype, effect=effect,
                se=se, ci_low=lo, ci_high=hi, p=p,
                neglog10p=(-np.log10(p) if (p and p > 0) else np.nan), prevalence=prev, unstable=note)


def _2x2_or(ordered, ingrp):
    a = int(((ordered == 1) & ingrp).sum()); b = int(((ordered == 0) & ingrp).sum())
    cc = int(((ordered == 1) & ~ingrp).sum()); d = int(((ordered == 0) & ~ingrp).sum())
    orr = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (cc + 0.5))            # Haldane-corrected OR
    try:
        _, p, _, _ = stats.chi2_contingency([[a, b], [cc, d]])
    except Exception:
        p = np.nan
    return orr, p, a + b + cc + d


def cochran_armitage(ordered, dose):
    d = np.array([0, 1, 2, 3]); N_i = np.array([(dose == k).sum() for k in range(4)])
    R_i = np.array([((dose == k) & (ordered == 1)).sum() for k in range(4)])
    N = N_i.sum(); R = R_i.sum(); pbar = R / N
    T = np.sum(d * (R_i - N_i * pbar))
    var = pbar * (1 - pbar) * (np.sum(N_i * d**2) - (np.sum(N_i * d))**2 / N)
    if var <= 0:
        return np.nan, np.nan
    z = T / np.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


def unadjusted(df, prev):
    ordered = df["ordered"].to_numpy(); sev = df["severity_ord"].to_numpy()
    out = []
    ct = np.array([[((sev == k) & (ordered == v)).sum() for v in (0, 1)] for k in range(4)])
    if (ct.sum(1) >= MIN_GRP).all():
        try:
            chi2, p, _, _ = stats.chi2_contingency(ct)
            out.append(_row("omnibus", "unadjusted", ct.sum(), "cramers_v", np.sqrt(chi2 / ct.sum()), np.nan, np.nan, np.nan, p, prev))
        except Exception:
            pass
    z, p = cochran_armitage(ordered, sev)
    if np.isfinite(p):
        out.append(_row("trend", "unadjusted", len(df), "ca_z_r", z / np.sqrt(len(df)), np.nan, np.nan, np.nan, p, prev))
    for cname, k in PAIRS:
        m = np.isin(sev, [0, k])
        orr, p, n = _2x2_or(ordered[m], sev[m] == k)
        out.append(_row(cname, "unadjusted", n, "OR", orr, np.nan, np.nan, np.nan, p, prev))
    for cname in ["ahi_ge5", "ahi_ge15"]:
        orr, p, n = _2x2_or(ordered, df[cname].to_numpy() == 1)
        out.append(_row(cname, "unadjusted", n, "OR", orr, np.nan, np.nan, np.nan, p, prev))
    return out


def _or(res, name):
    b = res.params[name]; se = res.bse[name]; ci = res.conf_int().loc[name]
    return float(np.exp(b)), float(se), float(np.exp(ci[0])), float(np.exp(ci[1])), float(res.pvalues[name])


def adjusted(df, prev):
    out = []
    for model, cov in COV.items():
        # factor -> omnibus + pairwise
        try:
            res = smf.logit(f"ordered ~ {SEV}{cov}", data=df).fit(disp=0, maxiter=60)
            dummies = [n for n in res.params.index if "severity" in n and "T." in n]
            wt = res.wald_test(", ".join(f"{n} = 0" for n in dummies), scalar=True)
            out.append(_row("omnibus", model, len(df), "wald_chi2", float(np.ravel(wt.statistic)[0]), np.nan, np.nan, np.nan, float(wt.pvalue), prev))
            for cname, k in PAIRS:
                nm = [n for n in dummies if f"[T.{['None','Mild','Moderate','Severe'][k]}]" in n]
                if nm:
                    orr, se, lo, hi, p = _or(res, nm[0])
                    out.append(_row(cname, model, len(df), "OR", orr, se, lo, hi, p, prev,
                                    "" if abs(np.log(orr)) < 10 and se < 10 else "unstable"))
        except Exception as ex:
            out.append(_row("omnibus", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, prev, str(ex)[:30]))
        # trend
        try:
            res = smf.logit(f"ordered ~ severity_ord{cov}", data=df).fit(disp=0, maxiter=60)
            orr, se, lo, hi, p = _or(res, "severity_ord")
            out.append(_row("trend", model, len(df), "OR", orr, se, lo, hi, p, prev,
                            "" if abs(np.log(orr)) < 10 and se < 10 else "unstable"))
        except Exception as ex:
            out.append(_row("trend", model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, prev, str(ex)[:30]))
        # dichotomized AHI
        for cname in ["ahi_ge5", "ahi_ge15"]:
            try:
                res = smf.logit(f"ordered ~ {cname}{cov}", data=df).fit(disp=0, maxiter=60)
                orr, se, lo, hi, p = _or(res, cname)
                out.append(_row(cname, model, len(df), "OR", orr, se, lo, hi, p, prev))
            except Exception as ex:
                out.append(_row(cname, model, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, prev, str(ex)[:30]))
    return out


def run_lab(loinc, targets_path):
    tested = pq.read_table(targets_path, columns=["id"],
                           filters=[("loinc_code", "==", loinc)]).column("id").to_pylist()
    df = _COH.copy()
    df["ordered"] = df["id"].isin(set(tested)).astype(int)
    prev = float(df["ordered"].mean())
    rows = unadjusted(df, prev)
    if 0.01 < prev < 0.99:                       # skip adjusted for ~ubiquitous / ~absent (separation)
        rows += adjusted(df, prev)
    else:
        for m in COV:
            rows.append(_row("omnibus", m, len(df), "na", np.nan, np.nan, np.nan, np.nan, np.nan, prev, "prev_extreme_skip"))
    for r in rows:
        r["loinc_code"] = loinc
    return rows


def main() -> None:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(); ap.add_argument("--labs", type=int, default=0); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    cat = pd.read_csv(U.RESULTS / "loinc_category.csv")[["loinc_code", "category"]]

    for label in WINDOWS:
        tpath = (ANALYSIS / f"targets_{label}.parquet").as_posix()
        labs = sorted(pd.read_parquet(tpath, columns=["loinc_code"])["loinc_code"].unique())
        if a.labs:
            labs = labs[:a.labs]
        print(f"[{label}] {len(labs)} labs (full-cohort logistic)", flush=True)
        with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init, initargs=(None,)) as ex:
            results = list(ex.map(run_lab, labs, [tpath] * len(labs)))
        res = pd.DataFrame([r for sub in results for r in sub])
        res["window"] = label
        res = res.merge(cat, on="loinc_code", how="left")
        # FDR + Bonferroni across labs per (contrast, model)
        res["sig_bon"] = False; res["sig_fdr"] = False
        for _, idx in res[res["p"].notna()].groupby(["contrast", "model"]).groups.items():
            sub = res.loc[idx]; n = len(sub)
            res.loc[idx, "sig_bon"] = sub["p"] < 0.05 / n
            o = sub["p"].sort_values(); crit = 0.05 * np.arange(1, n + 1) / n
            passed = o.to_numpy() <= crit
            thr = o.to_numpy()[passed].max() if passed.any() else -1
            res.loc[idx, "sig_fdr"] = sub["p"] <= thr
        cols = ["window", "loinc_code", "category", "contrast", "model", "n", "prevalence",
                "effect_type", "effect", "se", "ci_low", "ci_high", "p", "neglog10p", "sig_bon", "sig_fdr", "unstable"]
        res[cols].to_csv(ANALYSIS / f"results_propensity_{label}.csv", index=False)

        # combined two-part table: Part1 OR (propensity) x Part2 IRR (order_rate)
        p2 = pd.read_csv(ANALYSIS / f"results_{label}.csv")
        p2 = p2[p2["target"] == "order_rate"][["loinc_code", "contrast", "model", "effect", "p", "sig_fdr"]]
        p1 = res[["loinc_code", "category", "contrast", "model", "effect", "p", "sig_fdr", "prevalence"]]
        tp = p1.merge(p2, on=["loinc_code", "contrast", "model"], how="outer", suffixes=("_p1_OR", "_p2_IRR"))
        tp.to_csv(ANALYSIS / f"twopart_{label}.csv", index=False)
        nsig = res[res.model == "m4"].groupby("contrast")["sig_fdr"].sum()
        print(f"[{label}] {len(res):,} tests -> results_propensity_{label}.csv; M4 FDR-sig by contrast:\n{nsig.to_string()}", flush=True)

    U.log_milestone("Step 12 hurdle Part 1: ordering-propensity results + two-part tables written.")


if __name__ == "__main__":
    main()
