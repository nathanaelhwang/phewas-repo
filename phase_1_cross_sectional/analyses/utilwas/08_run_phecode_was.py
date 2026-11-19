# %% [markdown]
# Step 08 - UtilWAS per-phecode two-part diagnosis association (a PheWAS per combo).
#
# For each (combo x phecode x window), off the step-06 targets (>=500-patient floor):
#   PART 1 - binary presence (patient has >=1 such [primary/secondary] dx of the phecode in-window)
#            over the FULL cohort -> logistic (M1/M3/M4, OR) + unadjusted chi2/Cochran-Armitage/crude-OR.
#   PART 2 - count among-present (encounters carrying the phecode, per yr) among patients with >=1
#            -> Poisson-rate (M1/M3/M4, IRR) + unadjusted KW/Spearman/Mann-Whitney.
# Combos: physician_primary; ED_primary/secondary; inpatient_primary/secondary; urgent_primary/secondary.
# Contrasts: omnibus, trend, mild/moderate/severe-vs-none, ahi_ge5, ahi_ge15. FDR+Bonferroni per
# (combo, part, contrast, model). -> analysis/results_phecode_{1yr,5yr}.csv  DE-IDENTIFIED.
# Usage: python 08_run_phecode_was.py [--jobs 12] [--combo NAME]

# %%
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"  # avoid BLAS oversubscription under joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import utilwas_utils as U
import _was_engine as E


def run_one(combo, phecode, phenotype, category, present, years):
    coh = E.cohort_cached().assign(obs_years=E.obs_years(E.cohort_cached(), years).to_numpy())
    df = coh.merge(present[["id", "n_enc"]], on="id", how="left")
    df["n_enc"] = df["n_enc"].fillna(0.0)
    df["y"] = (df["n_enc"] > 0).astype(int)
    rows = E.binary_unadjusted(df, "y", phecode) + E.binary_adjusted(df, phecode)
    # part 2: count among-present
    dpos = df[df["y"] == 1].copy()
    if len(dpos) >= 4 * E.MIN_GRP and dpos["n_enc"].nunique() >= 2:
        dpos["rate"] = dpos["n_enc"] / dpos["obs_years"]
        rows += E.rate_unadjusted(dpos, "rate", phecode, part="count_among_present")
        rows += E.rate_adjusted(dpos, phecode, part="count_among_present")
    for r in rows:
        r["combo"], r["phenotype"], r["category"] = combo, phenotype, category
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--combo", type=str, default="")
    a = ap.parse_args()

    for label, years in U.WINDOWS:
        tg = pd.read_parquet(U.ANALYSIS / f"phecode_targets_{label}.parquet")
        kept = pd.read_csv(U.ANALYSIS / f"kept_phecodes_{label}.csv",
                           keep_default_na=False, dtype={"phecode": str})
        if a.combo:
            tg = tg[tg["combo"] == a.combo]
            kept = kept[kept["combo"] == a.combo]
        lab = {(r.combo, r.phecode): (r.phenotype, r.category) for r in kept.itertuples()}
        tg["phecode"] = tg["phecode"].astype(str)
        groups = list(tg.groupby(["combo", "phecode"], sort=False))
        print(f"[{label}] {len(groups)} (combo,phecode) features", flush=True)

        res = Parallel(n_jobs=a.jobs, verbose=3)(
            delayed(run_one)(combo, phe, *lab.get((combo, phe), ("", "")), sub, years)
            for (combo, phe), sub in groups)
        rows = [r for sub in res for r in sub]
        d = pd.DataFrame(rows)
        d["window"] = label
        d = E.add_fdr_bonferroni(d, ["combo", "part", "contrast", "model"])

        cols = ["window", "combo", "feature", "phenotype", "category", "part", "contrast", "model",
                "n", "effect_type", "effect", "se", "ci_low", "ci_high", "p", "neglog10p",
                "sig_bon", "sig_fdr", "unstable"]
        out = U.ANALYSIS / f"results_phecode_{label}.csv"
        d[cols].to_csv(out, index=False)

        adj = d[(d.model.isin(["m1", "m3", "m4"]))]
        print(f"[{label}] {len(d):,} tests -> {out.name}; FDR-sig (adjusted) by part/model:")
        print(adj.groupby(["part", "model"])["sig_fdr"].sum().to_string(), flush=True)
        # headline: presence trend m4 top ORs
        h = d[(d.part == "presence") & (d.contrast == "trend") & (d.model == "m4") & d.sig_fdr]
        h = h.reindex(h["effect"].sort_values(ascending=False).index).head(6)
        print(f"[{label}] presence trend (m4) top OR/step-severity, FDR-sig:")
        for _, r in h.iterrows():
            print(f"    [{r['combo']:19s}] {str(r['phenotype'])[:34]:34s} OR={r['effect']:.2f} p={r['p']:.1e}")

    U.log_milestone("Step 08 phecode PheWAS: two-part (logistic presence + Poisson count-among-present) "
                    "M1/M3/M4 + unadjusted, 7 combos, 7 contrasts, FDR+Bonferroni.")


if __name__ == "__main__":
    main()
