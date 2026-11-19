# %% [markdown]
# Step 04 - Interval-specific incidence-rate buckets (the PH-free, absolute-risk complement to Cox).
#
# For each phecode x severity x year-interval (0-1, 1-2, 2-3 yr post-index), among the harmonized
# at-risk set: incident events with event date in the interval / person-time at risk within the
# interval -> incidence density per 1,000 person-years, plus the severe-vs-none rate ratio. A
# front-loaded year-1 spike that decays flags detection/prevalence artifact; a sustained/rising
# gradient is genuine incident risk. Reuses the engine's harmonized-washout at-risk frame.
#
# Usage: python 04_incidence_buckets.py [--pilot] [--jobs 12]
# Emits analysis/incidence_buckets[_pilot].csv. DE-IDENTIFIED.

# %%
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import incwas_utils as U
import _cox_engine as E


def buckets_one(phecode, ev_slice):
    foll, _ = E.prep_followup()
    prev, excl = E.prev_index(), E.exclude_map()
    d = E.build_survival(phecode, ev_slice, foll, prev, excl)
    if len(d) == 0 or int(d["event"].sum()) < U.MIN_EVENTS:
        return []
    rows = []
    for sev in U.SEVERITIES:
        s = d[d["severity"] == sev]
        t = s["time"].to_numpy(dtype=float)
        e = s["event"].to_numpy()
        for k in U.HORIZONS:                      # interval (k-1, k]
            lo, hi = k - 1, k
            events = int(((e == 1) & (t > lo) & (t <= hi)).sum())
            py = float(np.clip(np.minimum(t, hi) - lo, 0, 1).sum())   # person-years in the interval
            rows.append(dict(phecode=phecode, severity=sev, year=k, n_atrisk=len(s),
                             events=events, person_years=round(py, 1),
                             ir_per_1000=(round(1000 * events / py, 2) if py > 0 else np.nan)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    U.ensure_dirs()

    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: r.phenotype for r in dim.itertuples()}

    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    tag = ""
    if a.pilot:
        ev = ev[ev["phecode"].isin(U.PILOT_PHECODES)]
        tag = "_pilot"
    groups = list(ev.groupby("phecode", sort=False))
    print(f"phecodes: {len(groups)}", flush=True)

    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(buckets_one)(p, s) for p, s in groups)
    rows = [r for sub in res for r in sub]
    d = pd.DataFrame(rows)
    d["phenotype"] = d["phecode"].map(lambda p: lab.get(p, ""))

    # severe-vs-none rate ratio per (phecode, year)
    none = (d[d.severity == "None"].set_index(["phecode", "year"])["ir_per_1000"])
    d["irr_vs_none"] = d.apply(
        lambda r: (round(r["ir_per_1000"] / none.get((r["phecode"], r["year"]), np.nan), 2)
                   if none.get((r["phecode"], r["year"]), 0) else np.nan), axis=1)

    cols = ["phecode", "phenotype", "severity", "year", "n_atrisk", "events", "person_years",
            "ir_per_1000", "irr_vs_none"]
    out = U.ANALYSIS / f"incidence_buckets{tag}.csv"
    d[cols].sort_values(["phecode", "year", "severity"]).to_csv(out, index=False)
    print(f"{len(d):,} rows -> {out.name}")

    # headline: severe-vs-none IRR by year for a few anchors
    show = d[(d.severity == "Severe") & (d.phecode.isin(["250.2", "401.1", "428.2"]))]
    print("\nSevere-vs-None incidence-rate ratio by year (anchors):")
    for _, r in show.sort_values(["phenotype", "year"]).iterrows():
        print(f"  {str(r['phenotype'])[:26]:26s} yr{r['year']}: IR={r['ir_per_1000']:.1f}/1k  IRR={r['irr_vs_none']}")

    U.log_milestone(f"Step 04 incidence buckets{tag}: interval IR/1k PY + severe-vs-none IRR "
                    f"(yr 1/2/3) for {d['phecode'].nunique()} phecodes.")


if __name__ == "__main__":
    main()
