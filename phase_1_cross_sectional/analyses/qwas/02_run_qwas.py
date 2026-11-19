# %% [markdown]
# QWAS Step 02 - questionnaire-item association driver (the prevalence WAS).
#
# Every questionnaire feature is tested against the same seven OSA-severity contrasts as
# every other WAS stage (omnibus 4-group Wald, trend on severity_ord, Mild/Moderate/
# Severe-vs-None, AHI>=5, AHI>=15) under an unadjusted screen plus the nested ladder
#   m1 crude
#   m2 + age_z + sex_M + C(RACE_ETH) + C(study_year)
#   m3 + C(smoking_status) + C(income_gp) + obs_window_z      (== every other stage's m3)
#   m4 + bmi_z
# using the shared BehWAS engine. Two outcome forms per feature where both apply:
#   CONTINUOUS  rank-inverse-normal -> OLS, effect = beta; unadjusted = Kruskal-Wallis /
#               Spearman trend / Mann-Whitney (rank-biserial).
#   BINARY      logistic GLM, effect = OR; unadjusted = chi2 omnibus / Cochran-Armitage
#               trend / 2x2 crude OR (Fisher).
#
# Analysis is COMPLETE-CASE PER FEATURE -- unlike the other stages there is no shared
# spine, because a missing questionnaire answer is unknown, not a structural zero. So n
# varies by feature and is reported on every row.
#
# BH-FDR + Bonferroni are applied within each (contrast, model, effect_type) family, so
# the OR phenome and the beta phenome are corrected separately.
#
# -> analysis/results.csv, analysis/descriptives.csv
# DE-IDENTIFIED aggregates only (no patient rows).

# %%
import os
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import qwas_utils as U
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "behwas"))
import _behwas_engine as E

MIN_N = 4 * E.MIN_GRP        # minimum answered patients to attempt any test

RESULT_COLS = ["feature", "domain", "form", "referral_item", "contrast", "model", "n",
               "effect_type", "effect", "se", "ci_low", "ci_high", "p", "neglog10p",
               "sig_bon", "sig_fdr", "unstable"]


def binarize(v: pd.Series, op: str, val: float) -> pd.Series:
    return (v >= val).astype(float) if op == ">=" else (v <= val).astype(float)


def run_feature(spec: dict, d: pd.DataFrame) -> dict:
    """All tests + descriptives for one feature (both forms where applicable)."""
    feat, forms = spec["feature"], spec["forms"]
    d = d[d["v"].notna()].copy()
    tests, descr = [], []

    def skip(form, why):
        tests.append(E.row(feat, "omnibus", "unadjusted", len(d), "skip",
                           np.nan, np.nan, np.nan, np.nan, np.nan, why) | {"form": form})

    # ---- continuous form ----
    if forms in ("continuous", "both"):
        if len(d) < MIN_N or d["v"].nunique() < 2:
            skip("continuous", "low_n_or_zero_variance")
        else:
            rows = (E.cont_unadjusted(d, "v", feat)
                    + E.cont_adjusted(d, "v", feat, covs=U.COV))
            tests += [r | {"form": "continuous"} for r in rows]
            for g, sub in d.groupby("severity"):
                descr.append(dict(feature=feat, form="continuous", severity=g, n=len(sub),
                                  mean=round(float(sub["v"].mean()), 4),
                                  median=float(sub["v"].median()), prevalence=np.nan))

    # ---- binary form ----
    if forms in ("binary", "both"):
        b = d["v"] if forms == "binary" else binarize(d["v"], spec["cut_op"], spec["cut_val"])
        d = d.assign(b=b.astype(float))
        pos = float(d["b"].sum())
        if len(d) < MIN_N or min(pos, len(d) - pos) < U.MIN_EVENTS:
            skip("binary", "low_n_or_minority_class")
        else:
            rows = (E.binary_unadjusted(d, "b", feat)
                    + E.binary_adjusted(d, "b", feat, covs=U.COV))
            tests += [r | {"form": "binary"} for r in rows]
            for g, sub in d.groupby("severity"):
                descr.append(dict(feature=feat, form="binary", severity=g, n=len(sub),
                                  mean=np.nan, median=np.nan,
                                  prevalence=round(float(sub["b"].mean()), 5)))

    return {"tests": tests, "descr": descr}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    U.ensure_dirs()
    cohort = E.prep_cohort()
    tg = pd.read_parquet(U.TARGETS_DIR / "targets.parquet")
    sp = pd.read_csv(U.TARGETS_DIR / "feature_spec.csv", keep_default_na=False, na_values=[""])
    sp["referral_item"] = sp["referral_item"].astype(str).str.lower().isin(["true", "1"])

    base = cohort.merge(tg[["id"]], on="id", how="inner")
    tg = tg.set_index("id").loc[base["id"]].reset_index()
    print(f"analytic sample: {len(base):,} patients; {len(sp)} features "
          f"({sp['forms'].value_counts().to_dict()})", flush=True)

    res = Parallel(n_jobs=a.jobs, verbose=3)(
        delayed(run_feature)(r, base.assign(
            v=tg[r["feature"]].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)))
        for r in sp.to_dict("records"))

    tests = pd.DataFrame([t for x in res for t in x["tests"]]).rename(columns={"outcome": "feature"})
    descr = pd.DataFrame([t for x in res for t in x["descr"]])

    meta = sp[["feature", "domain", "referral_item"]]
    tests = tests.merge(meta, on="feature", how="left")
    tests = E.add_fdr_bonferroni(tests, ["contrast", "model", "effect_type"])
    tests[RESULT_COLS].to_csv(U.ANALYSIS / "results.csv", index=False)
    descr.merge(meta, on="feature", how="left").to_csv(U.ANALYSIS / "descriptives.csv", index=False)

    # ---- console validation ----
    skipped = tests[tests.effect_type == "skip"]
    print(f"\n{len(tests):,} tests written; {skipped['feature'].nunique()} feature-forms skipped")
    adj = tests[tests.model.isin(["m1", "m2", "m3", "m4"])]
    print("adjusted FDR-significant by model x effect_type:")
    print(adj.pivot_table(index="model", columns="effect_type", values="sig_fdr",
                          aggfunc="sum").to_string())
    for et, lab in [("beta", "INT beta"), ("OR", "odds ratio")]:
        m4 = tests[(tests.contrast == "trend") & (tests.model == "m4") & (tests.effect_type == et)]
        if not len(m4):
            continue
        top = m4.reindex(m4["neglog10p"].sort_values(ascending=False).index).head(8)
        print(f"\ntop severity-trend hits, m4, {lab}:")
        for _, r in top.iterrows():
            flag = " [referral]" if r["referral_item"] else ""
            print(f"    {str(r['feature'])[:22]:22s} {r['effect']:+.3f}  p={r['p']:.1e}"
                  f"  n={r['n']:,}  {'FDR' if r['sig_fdr'] else ''}{flag}")

    U.log_milestone(f"Step 02 QWAS association: {len(tests):,} tests over {len(sp)} questionnaire "
                    f"features (binary OR + continuous beta forms, unadjusted + m1-m4, 7 contrasts, "
                    f"FDR+Bonferroni per contrast x model x effect_type) -> analysis/results.csv.")


if __name__ == "__main__":
    main()
