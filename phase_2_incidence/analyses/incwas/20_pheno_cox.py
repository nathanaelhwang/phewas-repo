"""Phenotype-exposure incidence WAS — Cox scan with the latent phenotypes as exposure.

Repeats the incwas design with OSA severity replaced by the cross-domain latent
phenotypes. The event store, follow-up/censoring frame and harmonized control-exclusion
washout are reused unchanged; only the exposure and the covariate ladder differ.

Two tiers, because the cohort size depends on which axes the exposure needs:

  tierA  exposure = score_c1 + score_c3 (2 df), phecode-level scan.
         112,555 patients / 328,352 person-years / 2.92 y mean follow-up.
         Only the *archetype label* needs all three domains; the SCORES are per-domain,
         and c2 (questionnaire) is the small one. Reconstructing the scores from each
         domain's own assignment file therefore buys 9x the person-time.

  tierB  exposure = score_c1 + score_c2 + score_c3 (3 df) and, separately, the six corner
         archetypes vs unclassified. 25,380 patients / 35,994 py / 1.42 y.
         Run at phecode level where events allow, and at body-system level otherwise.

Two corrections that are not optional:
  * 327.x is removed from the outcome set. `327.3 Sleep apnea` is the single largest
    "incident" code here because every patient had a diagnostic sleep study at index.
  * A 90-day landmark is available (`--grace 90`). 19.3% of all incident events fall
    within 90 days of index — AF 45%, HF 42%, T2D 39% — which is the post-sleep-study
    workup cascade recording prevalent-but-undiagnosed disease as incident.

Usage:
  python 20_pheno_cox.py --tier A --jobs 12
  python 20_pheno_cox.py --tier B --grace 90
DE-IDENTIFIED aggregates only.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
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
import _pheno_engine as P

_D = None      # exposure frame, cached per worker process


def frame(tier: str, grace: int):
    global _D
    if _D is None:
        doms = (("c1_physiologic", "c3_comorbidities") if tier == "A"
                else ("c1_physiologic", "c2_symptomatic", "c3_comorbidities"))
        d = P.load_exposure(doms, with_archetype=(tier == "B"))
        _D = (d, P.exposure_blocks(d), P.adjustment_sets(d))
    return _D


def run_one(phecode, ev_slice, tier, grace, prev, excl, gr):
    d, exps, adj = frame(tier, grace)
    surv = E.build_survival(phecode, ev_slice, d, prev, excl,
                            grace_days=grace, grace_by_phe=gr)
    n_ev = int(surv["event"].sum())
    if n_ev < U.MIN_EVENTS:
        return []
    specs = [("scores", exps["scores"])]
    if tier == "B" and exps["archetype"]:
        specs.append(("archetype", exps["archetype"]))
    out = []
    for spec_name, ecols in specs:
        ecols = [c for c in ecols if surv[c].nunique() > 1]
        if not ecols:
            continue
        for model, cov in adj.items():
            npar = len(ecols) + len(cov)
            note = "epv<10" if n_ev < U.EPV * max(npar, 1) else ""
            try:
                cph, kept = E._fit(surv, ecols + cov)
                php = (E._ph_sub(surv, ecols + cov, ecols) if model == "m4" else np.nan)
                for col in ecols:
                    if col not in kept:
                        continue
                    e, se, lo, hi, p = E._eff(cph, col)
                    bad = note + ("|unstable" if (abs(np.log(e)) > 10 or se > 10) else "")
                    r = E.row(phecode, f"{spec_name}:{col}", model, len(surv), n_ev,
                              e, se, lo, hi, p, php, bad)
                    out.append(r)
            except Exception as ex:
                out.append(E.row(phecode, f"{spec_name}:fit", model, len(surv), n_ev,
                                 np.nan, np.nan, np.nan, np.nan, np.nan,
                                 note=str(ex)[:40]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default="A")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--grace", type=int, default=0)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()

    P.OUT.mkdir(parents=True, exist_ok=True)
    d, exps, adj = frame(a.tier, a.grace)
    print(f"tier {a.tier}: {len(d):,} patients, {d['followup_years'].sum():,.0f} person-years, "
          f"mean follow-up {d['followup_years'].mean():.2f} y, grace={a.grace}d", flush=True)
    print(f"  exposures: {[f'{k}({len(v)})' for k, v in exps.items() if v]}", flush=True)
    print(f"  models: {[f'{k}({len(v)})' for k, v in adj.items()]}", flush=True)

    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: (r.phenotype, r.category) for r in dim.itertuples()}

    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    ev = ev[ev["id"].isin(set(d["id"]))]
    n_before = ev["phecode"].nunique()
    ev = ev[~ev["phecode"].map(P.is_circular_outcome)]
    print(f"  outcomes: dropped {n_before - ev['phecode'].nunique()} circular sleep "
          f"phecodes (327.x); {ev['phecode'].nunique()} remain", flush=True)
    if a.pilot:
        ev = ev[ev["phecode"].isin(U.PILOT_PHECODES)]

    E._PREP = (d, {})                      # make _cox_engine reuse OUR exposure frame
    E._ID2IX = {v: i for i, v in enumerate(d["id"].to_numpy())}
    prev, excl = E.prev_index(), E.exclude_map()
    gr = E.grace_index(a.grace) if a.grace else None

    groups = list(ev.groupby("phecode", sort=False))
    print(f"  phecodes to fit: {len(groups)} (floor {U.MIN_EVENTS} events)", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(run_one)(p, s, a.tier, a.grace, prev, excl, gr) for p, s in groups)
    rows = [r for sub in res for r in sub]
    if not rows:
        print("no phecode met the event floor.")
        return

    r = pd.DataFrame(rows)
    r["phenotype"] = r["phecode"].map(lambda p: lab.get(p, ("", ""))[0])
    r["category"] = r["phecode"].map(lambda p: lab.get(p, ("", ""))[1])
    r = E.add_fdr_bonferroni(r, ["contrast", "model"])
    tag = f"tier{a.tier}" + (f"_grace{a.grace}" if a.grace else "")
    cols = ["phecode", "phenotype", "category", "contrast", "model", "n_atrisk",
            "n_events", "effect_type", "effect", "se", "ci_low", "ci_high", "p",
            "neglog10p", "ph_p", "sig_bon", "sig_fdr", "unstable"]
    out = P.OUT / f"results_{tag}.csv"
    r[cols].to_csv(out, index=False)

    dt = time.time() - t0
    print(f"\n{len(r):,} tests over {r.phecode.nunique()} phecodes in {dt/60:.1f} min "
          f"-> {out.name}", flush=True)
    for c in sorted(r.contrast.unique()):
        s = r[(r.contrast == c) & (r.model == "m4")]
        print(f"  {c:22s} FDR-sig at m4: {int(s.sig_fdr.sum()):4d} / {len(s)}")
    U.log_milestone(
        f"Phenotype-exposure incidence WAS {tag}: {len(r):,} tests / {r.phecode.nunique()} "
        f"phecodes on {len(d):,} patients ({d['followup_years'].sum():,.0f} py). "
        f"Exposure = latent phenotype scores; 327.x excluded as circular. {dt/60:.1f} min.")


if __name__ == "__main__":
    main()
