"""Octant-phenotype incidence models — omnibus first, one-vs-rest gated behind it.

Exposure = the eight octant phenotypes (each gradient score cut at its median), which
classify every patient rather than the 24.6% the corner archetypes reached.

The testing scheme is hierarchical, and that is the point:

  STAGE 1  OMNIBUS. A single joint Wald test that all seven non-reference octant
           coefficients are zero — "do the eight phenotypes differ at all for this
           outcome?" One 7-df test per outcome instead of eight separate contrasts,
           so the multiplicity burden falls eightfold. FDR + Bonferroni across outcomes.

  STAGE 2  ONE-VS-REST, only for outcomes whose omnibus survives. Each phenotype
           against the pooled other seven — the comparison actually of interest for a
           taxonomy — fitted as a single binary indicator per phenotype. Gating on the
           omnibus means these contrasts do not pay a multiplicity penalty across
           outcomes where nothing is happening.

Note on the ridge. `_cox_engine` uses penalizer=0.1, which shrinks a rare binary dummy
far harder than a continuous covariate because the L2 penalty is scale-free but the
covariate variances are not: the six corner archetypes had variance 0.017-0.074 against
0.547 for a score, and their coefficients came back attenuated ~65%. Median-split octants
sit at variance ~0.109, better but still 5x below a score. This script therefore fits at
penalizer=0.01 and falls back to 0.1 only if that fit fails, recording which was used.

Usage:  python 24_pheno_octant_cox.py --level phecode
        python 24_pheno_octant_cox.py --level system
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
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from scipy import stats

import incwas_utils as U
import _cox_engine as E
import _pheno_engine as P

OMNI_ALPHA = 0.05          # gate for stage 2
_STATE = None


def state():
    global _STATE
    if _STATE is None:
        d = P.load_exposure(("c1_physiologic", "c2_symptomatic", "c3_comorbidities"),
                            with_archetype=False)
        d = P.assign_octants(d)
        _STATE = (d, P.adjustment_sets(d))
    return _STATE


def _fit(df, cols, pen_try=(0.01, 0.1)):
    """Fit at the lightest penalty that converges; return (fitter, kept cols, penalty)."""
    keep = [c for c in cols if df[c].nunique() > 1]
    sub = df[["time", "event"] + keep]
    for pen in pen_try:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(sub, duration_col="time", event_col="event", batch_mode=True)
            return cph, keep, pen
        except Exception:
            continue
    raise RuntimeError("no penalty converged")


def run_one(key, surv, d, adj):
    """Omnibus + (if it passes) one-vs-rest, across the covariate ladder."""
    n_ev = int(surv["event"].sum())
    if n_ev < U.MIN_EVENTS:
        return []
    dummies = [f"oct_{P._slug(o)}" for o in P.OCTANTS if o != P.OCTANT_REF]
    dummies = [c for c in dummies if surv[c].nunique() > 1]
    if len(dummies) < 2:
        return []

    out = []
    for model, cov in adj.items():
        npar = len(dummies) + len(cov)
        note = "epv<10" if n_ev < U.EPV * max(npar, 1) else ""
        # ---- stage 1: omnibus ----
        try:
            cph, kept, pen = _fit(surv, dummies + cov)
            stat, p_om, df_om = E._wald_omnibus(cph, dummies)
            r = E.row(key, "omnibus", model, len(surv), n_ev, stat, np.nan, np.nan,
                      np.nan, p_om, np.nan, note)
            r["df"] = df_om
            r["penalizer"] = pen
            out.append(r)
        except Exception as ex:
            out.append(E.row(key, "omnibus", model, len(surv), n_ev, np.nan, np.nan,
                             np.nan, np.nan, np.nan, note=str(ex)[:40]))
            continue
        if not (np.isfinite(p_om) and p_om < OMNI_ALPHA):
            continue                                   # gate: no stage 2 for this model
        # ---- stage 2: one-vs-rest, this phenotype against the pooled other seven ----
        for o in P.OCTANTS:
            ind = (surv["octant"] == o).astype(float)
            if ind.nunique() < 2 or ind.sum() < 20:
                continue
            s2 = surv.assign(_ovr=ind)
            try:
                c2, k2, pen2 = _fit(s2, ["_ovr"] + cov)
                if "_ovr" not in k2:
                    continue
                e, se, lo, hi, p = E._eff(c2, "_ovr")
                rr = E.row(key, f"one_vs_rest:{o}", model, len(s2), n_ev,
                           e, se, lo, hi, p, np.nan,
                           note + ("|unstable" if (abs(np.log(e)) > 10 or se > 10) else ""))
                rr["df"] = 1
                rr["penalizer"] = pen2
                rr["n_in_group"] = int(ind.sum())
                out.append(rr)
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["phecode", "system"], default="phecode")
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    P.OUT.mkdir(parents=True, exist_ok=True)

    d, adj = state()
    cuts = P.octant_cuts()
    print(f"cohort {len(d):,} patients, {d.followup_years.sum():,.0f} py")
    print(f"octant cut points: { {k: round(v, 4) for k, v in cuts.items()} }")
    print(d["octant"].value_counts().rename("n").to_frame().assign(
        pct=lambda x: (x.n / len(d)).round(3)).to_string())

    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    ev = ev[ev["id"].isin(set(d["id"]))]
    ev = ev[~ev["phecode"].map(P.is_circular_outcome)]
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)

    E._PREP, E._ID2IX = (d, {}), {v: i for i, v in enumerate(d["id"].to_numpy())}
    prev, excl = E.prev_index(), E.exclude_map()

    t0 = time.time()
    if a.level == "phecode":
        groups = list(ev.groupby("phecode", sort=False))
        print(f"\nphecodes to fit: {len(groups)}", flush=True)
        res = Parallel(n_jobs=a.jobs, verbose=5)(
            delayed(lambda p, s: run_one(
                p, E.build_survival(p, s, d, prev, excl), d, adj))(p, s)
            for p, s in groups)
    else:
        systems = sorted({c for c in dim["category"].unique() if c})
        print(f"\nbody systems to fit: {len(systems)}", flush=True)
        res = [run_one(s, P.system_events(d, ev, dim, s, rule="anynew"), d, adj)
               for s in systems]

    rows = [r for sub in res for r in sub]
    if not rows:
        print("nothing met the event floor.")
        return
    r = pd.DataFrame(rows).rename(columns={"phecode": "outcome"})
    if a.level == "phecode":
        lab = {x.phecode: (x.phenotype, x.category) for x in dim.itertuples()}
        r["phenotype"] = r["outcome"].map(lambda p: lab.get(p, ("", ""))[0])
        r["category"] = r["outcome"].map(lambda p: lab.get(p, ("", ""))[1])
    # multiplicity is applied WITHIN contrast x model, so the omnibus is corrected
    # across outcomes on its own and the gated one-vs-rest tests are not diluted by it
    r = E.add_fdr_bonferroni(r, ["contrast", "model"])
    out = P.OUT / f"results_octant_{a.level}.csv"
    r.to_csv(out, index=False)

    om = r[(r.contrast == "omnibus") & (r.model == "m4")]
    print(f"\n{len(r):,} tests over {r.outcome.nunique()} outcomes in "
          f"{(time.time()-t0)/60:.1f} min -> {out.name}")
    print(f"OMNIBUS (m4): p<0.05 {int((om.p < 0.05).sum())} | "
          f"FDR {int(om.sig_fdr.sum())} | Bonferroni {int(om.sig_bon.sum())} "
          f"of {len(om)} outcomes")
    ovr = r[r.contrast.str.startswith("one_vs_rest") & (r.model == "m4")]
    if len(ovr):
        print(f"ONE-VS-REST (m4, gated): {len(ovr)} tests | "
              f"FDR {int(ovr.sig_fdr.sum())} | Bonferroni {int(ovr.sig_bon.sum())}")
        top = ovr[ovr.sig_fdr].reindex(
            ovr[ovr.sig_fdr].p.sort_values().index).head(12)
        if len(top):
            cols = [c for c in ("outcome", "phenotype", "contrast", "effect", "ci_low",
                                "ci_high", "p", "n_events", "n_in_group") if c in top]
            print(top[cols].to_string(index=False))
    U.log_milestone(
        f"Octant-phenotype incidence models ({a.level}): 8 median-split phenotypes, "
        f"hierarchical omnibus -> gated one-vs-rest, {len(r):,} tests.")


if __name__ == "__main__":
    main()
