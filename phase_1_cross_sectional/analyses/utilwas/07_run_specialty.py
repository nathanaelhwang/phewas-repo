# %% [markdown]
# Step 07 - UtilWAS specialty encounter-frequency association (physician_level #1, allied_health #1).
#
# Per (family in {physician_level, allied_health} x DEPT_SPECIALTY x window): the per-patient
# encounters/yr in that specialty (count / obs_years, obs_years = LEAST(obs_window, N)) over the
# FULL cohort (non-visitors = 0). Tested vs OSA severity with the shared engine:
#   unadjusted KW / Spearman-trend / Mann-Whitney (rank-biserial) on the per-patient rate;
#   adjusted Poisson-rate GLM (offset log obs_years) M1/M3/M4 -> IRR, Wald-joint omnibus.
# Contrasts: omnibus, trend, mild/moderate/severe-vs-none, ahi_ge5, ahi_ge15. FDR+Bonferroni per
# (family, contrast, model). Also reports mean/median rate per severity, population & among-visitors.
# -> analysis/results_specialty_{1yr,5yr}.csv, analysis/specialty_descriptives_{1yr,5yr}.csv

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import utilwas_utils as U
import _was_engine as E


def run_one(family, specialty, present, coh):
    df = coh.merge(present[["id", "n_enc"]], on="id", how="left")
    df["n_enc"] = df["n_enc"].fillna(0.0)
    df["rate"] = df["n_enc"] / df["obs_years"]
    tests = E.rate_unadjusted(df, "rate", specialty) + E.rate_adjusted(df, specialty)
    for r in tests:
        r["family"] = family
    descr = []
    for denom, sub in [("population", df), ("among_visitors", df[df["n_enc"] > 0])]:
        for g, gg in sub.groupby("severity"):
            v = gg["rate"].to_numpy(dtype=float)
            descr.append(dict(family=family, specialty=specialty, denom=denom, severity=g, n=len(v),
                              mean_rate=round(float(v.mean()), 4) if len(v) else np.nan,
                              median_rate=float(np.median(v)) if len(v) else np.nan))
    return {"tests": tests, "descr": descr}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()

    cohort = E.prep_cohort()
    for label, years in U.WINDOWS:
        coh = cohort.assign(obs_years=E.obs_years(cohort, years).to_numpy())
        tg = pd.read_parquet(U.ANALYSIS / f"specialty_targets_{label}.parquet")
        groups = list(tg.groupby(["family", "dept_specialty"], sort=False))
        print(f"[{label}] {len(groups)} (family,specialty) features", flush=True)

        res = Parallel(n_jobs=a.jobs, verbose=4)(
            delayed(run_one)(fam, spec, sub, coh) for (fam, spec), sub in groups)
        tests = pd.DataFrame([r for x in res for r in x["tests"]])
        descr = pd.DataFrame([r for x in res for r in x["descr"]])
        tests["window"] = label
        descr["window"] = label
        tests = E.add_fdr_bonferroni(tests, ["family", "contrast", "model"])

        cols = ["window", "family", "feature", "part", "contrast", "model", "n", "effect_type",
                "effect", "se", "ci_low", "ci_high", "p", "neglog10p", "sig_bon", "sig_fdr", "unstable"]
        tests[cols].to_csv(U.ANALYSIS / f"results_specialty_{label}.csv", index=False)
        descr.to_csv(U.ANALYSIS / f"specialty_descriptives_{label}.csv", index=False)

        adj = tests[tests.model.isin(["m1", "m3", "m4"])]
        nsig = adj.groupby(["model"])["sig_fdr"].sum()
        print(f"[{label}] {len(tests):,} tests; adjusted FDR-sig by model:\n{nsig.to_string()}", flush=True)
        # headline: strongest severe_vs_none IRR (m4) among physician_level
        m4 = tests[(tests.contrast == "severe_vs_none") & (tests.model == "m4") & (tests.family == "physician_level")]
        top = m4.reindex(m4["effect"].sort_values(ascending=False).index).head(5)
        print(f"[{label}] physician_level severe_vs_none, top IRR (m4):")
        for _, r in top.iterrows():
            print(f"    {r['feature']:28s} IRR={r['effect']:.2f}  p={r['p']:.1e}  {'FDR' if r['sig_fdr'] else ''}")

    U.log_milestone("Step 07 specialty-frequency association: results_specialty_{1yr,5yr}.csv "
                    "(Poisson-rate M1/M3/M4 + unadjusted, 7 contrasts, FDR+Bonferroni).")


if __name__ == "__main__":
    main()
