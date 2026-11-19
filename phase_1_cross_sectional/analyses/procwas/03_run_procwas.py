# %% [markdown]
# ProcWAS Step 03 - Procedure-frequency association (Poisson rate WAS).
#
# Per (Clear descriptor x window): per-patient procedures/yr in that descriptor
# (count / obs_years, obs_years = LEAST(obs_window, N)) over the FULL 222,109 cohort
# (non-performers = 0). Tested vs OSA severity with the shared utilwas rate engine:
#   unadjusted  KW / Spearman-trend / Mann-Whitney (rank-biserial) on the per-patient rate;
#   adjusted    Poisson-rate GLM (offset log obs_years, HC0) -> IRR, Wald-joint omnibus,
#               nested ladder m1 (crude) / m2 (demographics+index-year) / m3 (+smoking+income) / m4 (+BMI).
#               obs_window is the rate offset, so it is NOT also a covariate.
# Contrasts: omnibus, trend, mild/moderate/severe-vs-none, ahi_ge5, ahi_ge15. FDR+Bonferroni
# per (contrast, model) across the descriptor phenome. Both Parts (Procedure type / Body system)
# share this run; the category columns are attached for the plot step.
# -> analysis/results_{1yr,5yr}.csv, descriptives_{1yr,5yr}.csv, results_total_{1yr,5yr}.csv

# %%
import os
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import procwas_utils as U
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "utilwas")))
import _was_engine as E

# m1 crude / m2 demographics / m3 (+smoking+income, == other WAS stages' m3) / m4 (+BMI) ladder.
_M2 = "age_z + sex_M + C(RACE_ETH) + C(study_year)"
_M3 = _M2 + " + C(smoking_status) + C(income_gp)"
COV = {"m1": "", "m2": f" + {_M2}", "m3": f" + {_M3}", "m4": f" + {_M3} + bmi_z"}


def _tests_for(df, feature):
    return (E.rate_unadjusted(df, "rate", feature)
            + E._adjusted(df, feature, "rate", COV, "n_proc", E._fit_rate, "IRR", "wald_chi2"))


def run_one(feature, present, coh):
    df = coh.merge(present[["id", "n_proc"]], on="id", how="left")
    df["n_proc"] = df["n_proc"].fillna(0.0)
    df["rate"] = df["n_proc"] / df["obs_years"]
    descr = []
    for denom, sub in [("population", df), ("among_performers", df[df["n_proc"] > 0])]:
        for g, gg in sub.groupby("severity"):
            v = gg["rate"].to_numpy(dtype=float)
            descr.append(dict(feature=feature, denom=denom, severity=g, n=len(v),
                              mean_rate=round(float(v.mean()), 5) if len(v) else np.nan,
                              median_rate=float(np.median(v)) if len(v) else np.nan))
    return {"tests": _tests_for(df, feature), "descr": descr}


COLS = ["window", "feature", "procedure_type", "body_system", "part", "contrast", "model", "n",
        "effect_type", "effect", "se", "ci_low", "ci_high", "p", "neglog10p", "sig_bon", "sig_fdr", "unstable"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    cohort = E.prep_cohort()
    cat = pd.read_csv(U.ANALYSIS / "descriptor_categories.csv")

    for label, years in U.WINDOWS:
        coh = cohort.assign(obs_years=E.obs_years(cohort, years).to_numpy())

        # ---- headline: total procedures/yr over the whole cohort ----
        tot = pd.read_parquet(U.ANALYSIS / f"total_{label}.parquet")
        dft = coh.merge(tot[["id", "n_proc"]], on="id", how="left")
        dft["n_proc"] = dft["n_proc"].fillna(0.0)
        dft["rate"] = dft["n_proc"] / dft["obs_years"]
        tot_tests = pd.DataFrame(_tests_for(dft, "ALL_PROCEDURES"))
        tot_tests["window"] = label
        tot_tests.to_csv(U.ANALYSIS / f"results_total_{label}.csv", index=False)

        # ---- per-descriptor WAS ----
        tg = pd.read_parquet(U.ANALYSIS / f"targets_{label}.parquet")
        groups = list(tg.groupby("clear_descriptor", sort=False))
        print(f"[{label}] {len(groups)} descriptors; cohort {len(coh):,}", flush=True)

        res = Parallel(n_jobs=a.jobs, verbose=3)(
            delayed(run_one)(feat, sub, coh) for feat, sub in groups)
        tests = pd.DataFrame([r for x in res for r in x["tests"]])
        descr = pd.DataFrame([r for x in res for r in x["descr"]])
        tests["window"] = label
        descr["window"] = label
        tests = E.add_fdr_bonferroni(tests, ["contrast", "model"])
        tests = tests.merge(cat, left_on="feature", right_on="clear_descriptor", how="left").drop(columns="clear_descriptor")

        tests[COLS].to_csv(U.ANALYSIS / f"results_{label}.csv", index=False)
        descr.to_csv(U.ANALYSIS / f"descriptives_{label}.csv", index=False)

        adj = tests[tests.model.isin(["m1", "m2", "m3", "m4"])]
        nsig = adj.groupby("model")["sig_fdr"].sum()
        print(f"[{label}] {len(tests):,} tests; adjusted FDR-sig by model:\n{nsig.to_string()}", flush=True)
        m4 = tests[(tests.contrast == "severe_vs_none") & (tests.model == "m4")]
        top = m4.reindex(m4["effect"].sort_values(ascending=False).index).head(6)
        print(f"[{label}] severe_vs_none top IRR (m4):")
        for _, r in top.iterrows():
            print(f"    {str(r['feature'])[:34]:34s} IRR={r['effect']:.2f}  p={r['p']:.1e}  {'FDR' if r['sig_fdr'] else ''}")
        tt = tot_tests[(tot_tests.contrast == "severe_vs_none") & (tot_tests.model == "m4")]
        if len(tt):
            r = tt.iloc[0]
            print(f"[{label}] ALL procedures severe_vs_none IRR (m4) = {r['effect']:.3f} (p={r['p']:.1e})", flush=True)

    U.log_milestone("Step 03 ProcWAS association: results_{1yr,5yr}.csv (Poisson-rate m2/m3/m4 + "
                    "unadjusted, 7 contrasts, FDR+Bonferroni) + total-procedures headline.")


if __name__ == "__main__":
    main()
