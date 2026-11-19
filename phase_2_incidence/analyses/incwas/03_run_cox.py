# %% [markdown]
# Step 03 - Per-phecode incidence Cox PheWAS (the association step).
#
# For each phecode (>= MIN_EVENTS incident events), fit cause-specific Cox PH of OSA severity
# on time-to-incident-onset among at-risk (baseline-free) patients, across the seven contrasts
# (omnibus / trend / mild/moderate/severe-vs-none / AHI>=5 / AHI>=15) and the nested m1-m4 ladder,
# with a per-exposure proportional-hazards (Schoenfeld) p-value and BH-FDR + Bonferroni per
# (contrast, model). CPAP is NOT adjusted (mediator).
#
# Usage:
#   python 03_run_cox.py --pilot            # the ~20 pilot phecodes (fast)
#   python 03_run_cox.py --phecode 401.1    # a single phecode
#   python 03_run_cox.py --jobs 12          # full phenome scan
# Emits analysis/results_cox[_pilot].csv + kept_phecodes[_pilot].csv. DE-IDENTIFIED.

# %%
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"  # avoid BLAS oversubscription under joblib
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import incwas_utils as U
import _cox_engine as E


def run_one(phecode, ev_slice):
    foll, blocks = E.prep_followup()
    prev, excl = E.prev_index(), E.exclude_map()
    return E.run_phecode(phecode, ev_slice, foll, blocks, prev, excl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--phecode", type=str, default="")
    a = ap.parse_args()

    U.ensure_dirs()
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: (r.phenotype, r.category) for r in dim.itertuples()}

    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    if a.phecode:
        ev = ev[ev["phecode"] == a.phecode]
        tag = f"_{a.phecode.replace('.', '_')}"
    elif a.pilot:
        ev = ev[ev["phecode"].isin(U.PILOT_PHECODES)]
        tag = "_pilot"
    else:
        tag = ""
    groups = list(ev.groupby("phecode", sort=False))
    print(f"phecodes to fit: {len(groups)}  (event floor = {U.MIN_EVENTS} incident events)", flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_one)(phe, sub) for phe, sub in groups)
    rows = [r for sub in res for r in sub]
    if not rows:
        print("no phecode met the incident-event floor.")
        return
    d = pd.DataFrame(rows)
    d["phenotype"] = d["phecode"].map(lambda p: lab.get(p, ("", ""))[0])
    d["category"] = d["phecode"].map(lambda p: lab.get(p, ("", ""))[1])
    d = E.add_fdr_bonferroni(d, ["contrast", "model"])

    cols = ["phecode", "phenotype", "category", "contrast", "model", "n_atrisk", "n_events",
            "effect_type", "effect", "se", "ci_low", "ci_high", "p", "neglog10p", "ph_p",
            "sig_bon", "sig_fdr", "unstable"]
    out = U.ANALYSIS / f"results_cox{tag}.csv"
    d[cols].to_csv(out, index=False)

    n_kept = d["phecode"].nunique()
    kept = (d.groupby(["phecode", "phenotype"])["n_events"].first()
            .reset_index().sort_values("n_events", ascending=False))
    kept.to_csv(U.ANALYSIS / f"kept_phecodes{tag}.csv", index=False)

    dt = time.time() - t0
    print(f"{len(d):,} tests over {n_kept} phecodes in {dt/60:.1f} min -> {out.name}", flush=True)
    print("FDR-significant by model (severe_vs_none):")
    sv = d[d.contrast == "severe_vs_none"]
    print(sv.groupby("model")["sig_fdr"].sum().to_string())
    # headline: severe-vs-none m4, FDR-sig, sorted by HR
    h = d[(d.contrast == "severe_vs_none") & (d.model == "m4") & d.sig_fdr & (d.effect.notna())]
    h = h.reindex(h["effect"].sort_values(ascending=False).index).head(15)
    print("\nsevere-vs-none (m4) FDR-sig, top HR:")
    for _, r in h.iterrows():
        flag = " [PH?]" if (np.isfinite(r["ph_p"]) and r["ph_p"] < 0.05) else ""
        print(f"    {str(r['phenotype'])[:38]:38s} HR={r['effect']:.2f} "
              f"({r['ci_low']:.2f}-{r['ci_high']:.2f}) p={r['p']:.1e} n_ev={r['n_events']}{flag}")

    U.log_milestone(
        f"Step 03 Cox PheWAS{tag}: {len(d):,} tests / {n_kept} phecodes (>= {U.MIN_EVENTS} events), "
        f"7 contrasts x m1-m4, cause-specific, ridge Cox, PH-tested, FDR+Bonferroni. In {dt/60:.1f} min.")


if __name__ == "__main__":
    main()
