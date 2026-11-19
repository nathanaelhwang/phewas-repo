"""Body-system incidence models for the full 3-domain phenotype (the user's fallback).

`20_pheno_cox.py --tier B` established that the 3-domain phenotype cohort (25,380
patients, 35,994 person-years, 1.42 y mean follow-up) is underpowered at phecode
resolution: **zero of 368 phecodes reach FDR significance for any of the nine exposure
terms**. This script runs the pre-agreed fallback — the 18 phecode body systems — where
the events aggregate into something estimable.

The outcome rule matters more than usual here. The obvious "clean" definition, at-risk =
free of EVERY phecode in the system at baseline, cannot be used: it retains only **0.2%**
of high-burden patients for the respiratory system, 0.6% for endocrine/metabolic and 1.2%
for circulatory, and mean score_c3 among the survivors collapses from 0.83 to 0.18-0.32.
It would delete the exposure it is meant to measure. The primary rule is therefore
**first-new-in-system**: everyone stays at risk, and the event is the first incident
phecode in that system the patient did not already carry. That changes the estimand to a
diagnostic-accrual hazard — "how fast does this phenotype acquire a NEW diagnosis in this
organ system" — which is stated rather than hidden. The strict rule is still run, as a
sensitivity, with its at-risk collapse reported alongside it.

Two whole systems act as negative controls: `sense organs` (refraction, cataract, otitis)
and `dermatologic` have no credible causal path from an OSA phenotype but share the
coding process, so their hazard ratios are the ascertainment floor.

Writes results/incwas_results/phenotype_exposure/results_system_*.csv.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import incwas_utils as U
import _cox_engine as E
import _pheno_engine as P

NCO_SYSTEMS = ["sense organs", "dermatologic"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["anynew", "strict", "both"], default="both")
    a = ap.parse_args()
    P.OUT.mkdir(parents=True, exist_ok=True)

    d = P.load_exposure(("c1_physiologic", "c2_symptomatic", "c3_comorbidities"),
                        with_archetype=True)
    exps, adj = P.exposure_blocks(d), P.adjustment_sets(d)
    print(f"3-domain cohort: {len(d):,} patients, {d.followup_years.sum():,.0f} py, "
          f"mean {d.followup_years.mean():.2f} y", flush=True)

    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    ev = ev[ev["id"].isin(set(d["id"]))]
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    systems = sorted({c for c in dim["category"].unique() if c})

    rules = ["anynew", "strict"] if a.rule == "both" else [a.rule]
    for rule in rules:
        rows, t0 = [], time.time()
        for s in systems:
            surv = P.system_events(d, ev, dim, s, rule=rule)
            n_ev = int(surv["event"].sum())
            if n_ev < U.MIN_EVENTS or len(surv) < 200:
                print(f"  skip {s}: {n_ev} events / {len(surv)} at-risk", flush=True)
                continue
            # how much exposure contrast survives this rule's at-risk selection
            retained = (surv["c3_comorbidities"] == "high-burden").sum() / max(
                (d["c3_comorbidities"] == "high-burden").sum(), 1)
            for spec in ("scores", "archetype"):
                ecols = [c for c in exps[spec] if surv[c].nunique() > 1]
                if not ecols:
                    continue
                for model, cov in adj.items():
                    # n_eligible_z enters from m2 on: it separates depletion of
                    # susceptibles (fewer codes left to acquire) from surveillance bias
                    cov = cov + (["n_eligible_z"] if model != "m1" else [])
                    npar = len(ecols) + len(cov)
                    note = "epv<10" if n_ev < U.EPV * max(npar, 1) else ""
                    try:
                        cph, kept = E._fit(surv, ecols + cov)
                        for col in ecols:
                            if col not in kept:
                                continue
                            e, se, lo, hi, p = E._eff(cph, col)
                            r = E.row(s, f"{spec}:{col}", model, len(surv), n_ev,
                                      e, se, lo, hi, p, np.nan, note)
                            r["mean_score_c3_atrisk"] = round(float(surv.score_c3.mean()), 3)
                            r["pct_highburden_retained"] = round(float(retained), 4)
                            r["rule"] = rule
                            rows.append(r)
                    except Exception as ex:
                        pass
            print(f"  {s:24s} {rule:7s} at-risk {len(surv):>6,} events {n_ev:>5,} "
                  f"high-burden retained {retained:5.1%}", flush=True)

        r = pd.DataFrame(rows).rename(columns={"phecode": "body_system"})
        r = E.add_fdr_bonferroni(r, ["contrast", "model"])
        out = P.OUT / f"results_system_{rule}.csv"
        r.to_csv(out, index=False)
        print(f"\n{len(r):,} tests over {r.body_system.nunique()} systems "
              f"in {(time.time()-t0)/60:.1f} min -> {out.name}")
        for c in sorted(r.contrast.unique()):
            s2 = r[(r.contrast == c) & (r.model == "m4")]
            print(f"  {c:52s} FDR-sig m4: {int(s2.sig_fdr.sum()):2d} / {len(s2)}")

        nco = r[(r.model == "m4") & r.body_system.isin(NCO_SYSTEMS) & r.effect.notna()]
        if len(nco):
            fl = nco.groupby("contrast")["effect"].apply(
                lambda x: float(np.exp(np.log(x).median()))).round(4)
            print("\n  ascertainment floor (median HR over sense-organ + dermatologic):")
            print("   ", fl.to_dict())
    U.log_milestone(
        "Phenotype-exposure incidence WAS: body-system fallback for the 3-domain "
        f"phenotype ({len(d):,} patients). Phecode level was underpowered (0/368 FDR).")


if __name__ == "__main__":
    main()
