# %% [markdown]
# MeasWAS Step 03 - continuous cardiopulmonary MEASUREMENT association.
#
# ProcWAS proper asks how OFTEN a procedure happens (Poisson procedures/yr, non-performers = a
# structural 0 over the full 222,109 cohort). This arm asks what the procedure MEASURED.
#
# *** COMPLETE-CASE, NOT ZERO-FILLED. ***  A patient contributes to a feature only if they have a
# non-null per-patient value for it in [index - 5yr, index). A patient with no echo has an UNKNOWN
# LVEF, not an LVEF of 0. Zero-filling here (the ProcWAS reflex) would be a silent, catastrophic
# error: it would turn "was the test done" into the outcome. So the analysed n varies per feature
# (echo 3,577-46,223 of 48,644 retained; PFT 5,152-7,874 of 7,875) and is reported per row.
#
# Outcome  : per-patient mean of the unit-repaired, gated measurement (already gated by steps 01/02).
#            Rank-inverse-normal transformed PER FEATURE, computed AFTER subsetting to that
#            feature's complete cases -> effect is a raw BETA in SD units of the RINT outcome.
#            NOT an IRR/OR: the utilwas engine's _eff_exp would be wrong here, so the adjusted
#            fitter is cloned locally as OLS while the unadjusted screen is reused as-is.
# Exposure : OSA severity, reference = None. Seven contrasts (omnibus, trend, mild/moderate/
#            severe-vs-none, ahi_ge5, ahi_ge15) - identical to every other WAS stage.
# Models   : unadjusted nonparametric screen (KW epsilon-squared / Spearman trend / Mann-Whitney
#            rank-biserial) + nested OLS ladder m1 crude / m2 demographics+index-year+obs_window /
#            m3 (+smoking+income) / m4 (+BMI). obs_window_z stays in (LabWAS keeps it; unlike
#            ProcWAS this is not a rate outcome with obs_window as the offset).
#            C(study_year) is MANDATORY: measurement availability is mechanically coupled to index
#            year (echo era drift; PFT coverage 13.2% for 2016 index dates vs 4.7% for 2024).
# Floors   : U.MIN_PATIENTS=500 complete cases per feature; MIN_GRP=20 per severity group.
# Multiplicity: FDR (BH) + Bonferroni per (contrast, model), computed SEPARATELY within each domain
#            ('echo' / 'pft') - different arms, different coverage, no shared denominator.
#
# -> analysis/results_5yr.csv, analysis/descriptives_5yr.csv.  DE-IDENTIFIED aggregates only.

# %%
import os
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats

import measwas_utils as U
sys.path.insert(0, str((Path(__file__).resolve().parent.parent.parent / "utilwas")))
import _was_engine as E

warnings.simplefilter("ignore")

# Same shape as the ProcWAS ladder, plus LabWAS's obs_window_z (see header).
_M2 = "age_z + sex_M + C(RACE_ETH) + C(study_year) + obs_window_z"
_M3 = _M2 + " + C(smoking_status) + C(income_gp)"
COV = {"m1": "", "m2": f" + {_M2}", "m3": f" + {_M3}", "m4": f" + {_M3} + bmi_z"}

# Echo concepts are declared in measwas_utils; spirometry has no such table, so name them here.
# PEF_REF is a PREDICTED value (a function of age/sex/height), not a measurement of the patient's
# lungs - it is in the user-specified phenome but reads as an anthropometric proxy, not physiology.
PFT_CONCEPT = {
    "FEV1_FVC_pre": "Airflow obstruction (ratio)",
    "FEV1_FVC_post": "Airflow obstruction (ratio, post-BD)",
    "FEV1_PRE": "Expiratory flow (FEV1 volume)",
    "FEV1_POST": "Expiratory flow (FEV1 volume, post-BD)",
    "FVC_PRE": "Lung volume (FVC)",
    "FVC_POST": "Lung volume (FVC, post-BD)",
    "PEF_POST": "Peak expiratory flow (post-BD)",
    "PEF_REF": "Peak expiratory flow (REFERENCE/predicted - anthropometric, not measured)",
    "PEF_PRE_PREDICTED": "Peak flow, fraction of reference",
    "PEF_POST_PREDICTED": "Peak flow, fraction of reference (post-BD)",
    "FEV1_PRE_PREDICTED": "FEV1, fraction of reference",
    "FEV1_POST_PREDICTED": "FEV1, fraction of reference (post-BD)",
    "FEF25_75_PRE": "Small-airway flow",
    "FEF25_75_POST": "Small-airway flow (post-BD)",
}
ECHO_CONCEPT = {f["name"]: f["concept"] for f in U.ECHO_FEATURES}

DOMAINS = [
    ("echo", U.ECHO_FEAT, U.ECHO_FEATURE_NAMES, ECHO_CONCEPT),
    ("pft", U.PFT_FEAT, U.PFT_FEATURES, PFT_CONCEPT),   # U.PFT_INPUTS_ONLY deliberately excluded
]


# %%
def int_transform(x: np.ndarray) -> np.ndarray:
    """Rank-inverse-normal (LabWAS int_transform). Ties -> average rank, so heavily tied
    quasi-ordinal features (estimated_rap) collapse onto few distinct normal scores."""
    r = stats.rankdata(x, method="average")
    return stats.norm.ppf((r - 0.5) / len(r))


def _eff_ols(res, name):
    b = res.params[name]; se = res.bse[name]; p = res.pvalues[name]
    ci = res.conf_int().loc[name]
    return (float(b), float(se), float(ci[0]), float(ci[1]), float(p))


def adjusted_ols(df, feature, domain, dep="y_int"):
    """Clone of E._adjusted with the exponentiation removed: OLS -> raw beta, F-test omnibus."""
    out = []
    grp = df["severity_ord"].to_numpy()
    have_sev = all((grp == k).sum() >= E.MIN_GRP for k in range(4))
    for model, cov in COV.items():
        if have_sev:
            try:
                res = smf.ols(f"{dep} ~ {E.SEV}{cov}", data=df).fit()
                dummies = [n for n in res.params.index if "severity" in n and "T." in n]
                wt = res.wald_test(", ".join(f"{n} = 0" for n in dummies), scalar=True)
                out.append(E.row(feature, domain, "omnibus", model, len(df), "wald_F",
                                 float(np.ravel(wt.statistic)[0]), np.nan, np.nan, np.nan,
                                 float(wt.pvalue)))
                for cname, lab in E.PAIRS:
                    nm = [n for n in dummies if lab in n]
                    if nm:
                        e, se, lo, hi, p = _eff_ols(res, nm[0])
                        out.append(E.row(feature, domain, cname, model, len(df), "beta", e, se, lo, hi, p))
            except Exception as ex:
                out.append(E.row(feature, domain, "omnibus", model, len(df), "na",
                                 np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        try:
            res = smf.ols(f"{dep} ~ severity_ord{cov}", data=df).fit()
            e, se, lo, hi, p = _eff_ols(res, "severity_ord")
            out.append(E.row(feature, domain, "trend", model, len(df), "beta", e, se, lo, hi, p))
        except Exception as ex:
            out.append(E.row(feature, domain, "trend", model, len(df), "na",
                             np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
        for cname in ["ahi_ge5", "ahi_ge15"]:
            npos = int(df[cname].sum())
            if df[cname].nunique() == 2 and npos >= E.MIN_GRP and (len(df) - npos) >= E.MIN_GRP:
                try:
                    res = smf.ols(f"{dep} ~ {cname}{cov}", data=df).fit()
                    e, se, lo, hi, p = _eff_ols(res, cname)
                    out.append(E.row(feature, domain, cname, model, len(df), "beta", e, se, lo, hi, p))
                except Exception as ex:
                    out.append(E.row(feature, domain, cname, model, len(df), "na",
                                     np.nan, np.nan, np.nan, np.nan, np.nan, str(ex)[:40]))
    return out


def run_feature(feature, domain, d):
    """d = complete cases for `feature` already joined to the cohort. Raw value in column `feature`."""
    d = d.assign(y_int=int_transform(d[feature].to_numpy(dtype=float)))
    # E.rate_unadjusted is 'rate' in name only - it is a generic continuous KW / Spearman-trend /
    # Mann-Whitney screen. Run it on the RAW value (rank-based, so the RINT is a no-op anyway) and
    # pass the domain through its `part` slot.
    tests = E.rate_unadjusted(d, feature, feature, part=domain) + adjusted_ols(d, feature, domain)
    descr = []
    for g, gg in d.groupby("severity"):
        v = gg[feature].to_numpy(dtype=float)
        q25, q75 = np.percentile(v, [25, 75])
        descr.append(dict(domain=domain, feature=feature, severity=g, n=len(v),
                          mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                          median=float(np.median(v)), q25=float(q25), q75=float(q75),
                          iqr=float(q75 - q25)))
    return {"tests": tests, "descr": descr}


COLS = ["domain", "feature", "concept", "contrast", "model", "n", "effect_type", "effect", "se",
        "ci_low", "ci_high", "p", "neglog10p", "sig_bon", "sig_fdr", "unstable"]


# %%
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    U.ensure_dirs()
    cohort = E.prep_cohort()
    print(f"cohort spine: {len(cohort):,} patients", flush=True)

    all_tests, all_descr = [], []
    for domain, path, features, concepts in DOMAINS:
        tbl = pd.read_parquet(path)
        df = tbl.merge(cohort, on="id", how="inner")
        print(f"\n[{domain}] {len(tbl):,} rows in store -> {len(df):,} joined to cohort; "
              f"{len(features)} candidate features", flush=True)

        jobs, dropped = [], []
        keep_cols = [c for c in cohort.columns if c != "id"]
        for f in features:
            d = df[df[f].notna() & np.isfinite(df[f])]
            if len(d) < U.MIN_PATIENTS or d[f].nunique() < 2:
                dropped.append((f, len(d)))
                continue
            jobs.append((f, d[[f] + keep_cols].copy()))
        if dropped:
            print(f"[{domain}] dropped by MIN_PATIENTS={U.MIN_PATIENTS} floor: "
                  + ", ".join(f"{f} (n={n:,})" for f, n in dropped), flush=True)
        print(f"[{domain}] testing {len(jobs)} features; complete-case n range "
              f"{min(len(d) for _, d in jobs):,}-{max(len(d) for _, d in jobs):,}", flush=True)

        res = Parallel(n_jobs=a.jobs, verbose=1)(
            delayed(run_feature)(f, domain, d) for f, d in jobs)
        tests = pd.DataFrame([r for x in res for r in x["tests"]]).rename(columns={"part": "domain"})
        tests = E.add_fdr_bonferroni(tests, ["contrast", "model"])   # within-domain denominator
        all_tests.append(tests)
        all_descr += [r for x in res for r in x["descr"]]

    res = pd.concat(all_tests, ignore_index=True)
    res["concept"] = res["feature"].map({**ECHO_CONCEPT, **PFT_CONCEPT})
    res["window"] = U.WINDOW_LABEL
    out = U.ANALYSIS / f"results_{U.WINDOW_LABEL}.csv"
    res[COLS].to_csv(out, index=False)
    dsc = pd.DataFrame(all_descr)
    dsc.to_csv(U.ANALYSIS / f"descriptives_{U.WINDOW_LABEL}.csv", index=False)

    # ---------------- de-identified summary ----------------
    print(f"\n{len(res):,} tests -> {out.name}; {len(dsc):,} descriptive rows", flush=True)
    print("\nFDR-significant tests by domain x model:")
    print(res.pivot_table(index="model", columns="domain", values="sig_fdr", aggfunc="sum",
                          fill_value=0).to_string())
    print("\nFDR-significant tests by domain x contrast:")
    print(res.pivot_table(index="contrast", columns="domain", values="sig_fdr", aggfunc="sum",
                          fill_value=0).to_string())

    for cname in ["severe_vs_none", "trend"]:
        sub = res[(res.contrast == cname) & (res.model == "m4") & res["p"].notna()]
        sub = sub.sort_values("p").head(10)
        print(f"\ntop m4 {cname} hits (beta = SD of rank-INT outcome):")
        for _, r in sub.iterrows():
            print(f"    {r['domain']:4s} {str(r['feature'])[:26]:26s} beta={r['effect']:+.4f} "
                  f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] n={r['n']:,} p={r['p']:.2e} "
                  f"{'FDR' if r['sig_fdr'] else ''}{' BON' if r['sig_bon'] else ''}")

    pc = res[(res.feature == "syst_bp_rest_supine") & (res.contrast == "severe_vs_none")
             & (res.model == "m4")]
    if len(pc):
        r = pc.iloc[0]
        print(f"\nPOSITIVE CONTROL syst_bp_rest_supine severe_vs_none (m4): beta={r['effect']:+.4f} "
              f"p={r['p']:.2e} n={r['n']:,}")

    U.log_milestone("Step 03 MeasWAS association: results_5yr.csv + descriptives_5yr.csv "
                    "(complete-case rank-INT OLS m1-m4 + unadjusted screen, 7 contrasts, "
                    "FDR+Bonferroni within echo/pft domains).")


if __name__ == "__main__":
    main()
