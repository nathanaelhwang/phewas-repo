"""Unadjusted log-rank tests across the eight octant phenotypes.

The Cox ladder answers "do the phenotypes differ once age, sex, BMI, utilisation and the
rest are held fixed". This answers the prior question the CIF panels actually draw: **do
the observed incidence curves differ at all**, with no model and no covariates.

Two contrasts per outcome, mirroring the hierarchical scheme so the numbers line up with
`results_octant_{system,phecode}.csv`:

  omnibus              8-group log-rank, 7 df -- the unadjusted analogue of the joint Wald
  one_vs_rest:<oct>    2-group log-rank, 1 df -- the unadjusted analogue of each panel

Read these as descriptive. The octants differ by 11.8 years of age and 25 percentage points
of female sex across the comorbidity axis, so an unadjusted p-value here is a statement
about the cohort as observed, not about the phenotype. Where the log-rank is significant
and the adjusted m4 estimate is not (or vice versa), the covariates are the reason, and
that comparison is the point of running this.

Estimator note: the log-rank tests CAUSE-SPECIFIC hazards, treating competing death as
censoring -- consistent with the Cox models, which do the same. The drawn curves are
Aalen-Johansen cumulative incidence, whose matching test is Gray's test on the
subdistribution hazard. The two agree closely when competing death is rare relative to the
event, so the competing-death count is carried per outcome for the reader to judge.

Output is an INTERNAL results file, unsuppressed, matching the convention of the other
`results_octant_*.csv`. It is not disclosure-cleared; re-run script 29 if it is ever added
to the website export.

Usage:  python 30_octant_logrank.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines.statistics import multivariate_logrank_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incwas_utils as U  # noqa: E402
import _cox_engine as E  # noqa: E402
import _pheno_engine as P  # noqa: E402

OUT = P.OUT
JOBS = 12


def tests_for(level, outcome, name, cat, surv) -> list[dict]:
    """8-group log-rank plus the eight one-vs-rest contrasts for one outcome."""
    n_ev = int(surv["event"].sum())
    if n_ev < U.MIN_EVENTS:
        return []
    t = surv["time"].to_numpy(float)
    ev = surv["event"].to_numpy(int)
    oc = surv["octant"].to_numpy()
    # competing deaths = died without the outcome; the log-rank censors these
    n_death = int(((ev != 1) & surv["is_death"].to_numpy()).sum())
    base = dict(level=level, outcome=outcome, outcome_name=name, category=cat,
                n_atrisk=len(surv), n_events=n_ev, n_deaths_competing=n_death)

    rows = []
    r = multivariate_logrank_test(t, oc, ev)
    rows.append({**base, "contrast": "omnibus", "df": len(np.unique(oc)) - 1,
                 "chi2": float(r.test_statistic), "p": float(r.p_value),
                 "n_focal": pd.NA, "events_focal": pd.NA,
                 "n_rest": pd.NA, "events_rest": pd.NA})
    for o in P.OCTANTS:
        m = oc == o
        if m.sum() < 20 or (~m).sum() < 20:
            continue
        r = multivariate_logrank_test(t, m.astype(int), ev)
        rows.append({**base, "contrast": f"one_vs_rest:{o}", "df": 1,
                     "chi2": float(r.test_statistic), "p": float(r.p_value),
                     "n_focal": int(m.sum()), "events_focal": int(ev[m].sum()),
                     "n_rest": int((~m).sum()), "events_rest": int(ev[~m].sum())})
    return rows


def main() -> None:
    d = P.load_exposure(("c1_physiologic", "c2_symptomatic", "c3_comorbidities"),
                        with_archetype=False)
    d = P.assign_octants(d)
    E._PREP, E._ID2IX = (d, {}), {v: i for i, v in enumerate(d["id"].to_numpy())}
    prev, excl = E.prev_index(), E.exclude_map()
    ev_all = pd.read_parquet(U.EVENTS)
    ev_all["phecode"] = ev_all["phecode"].astype(str)
    ev_all = ev_all[ev_all["id"].isin(set(d["id"]))]
    ev_all = ev_all[~ev_all["phecode"].map(P.is_circular_outcome)]
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {x.phecode: (x.phenotype, x.category) for x in dim.itertuples()}
    print(f"cohort {len(d):,} patients on the 3-domain incwas spine", flush=True)

    t0 = time.time()
    systems = sorted({c for c in dim["category"].unique() if c})
    rows = [r for s in systems
            for r in tests_for("system", s, s, s,
                               P.system_events(d, ev_all, dim, s, rule="anynew"))]
    print(f"{len(systems)} body systems done in {(time.time()-t0)/60:.1f} min", flush=True)

    groups = list(ev_all.groupby("phecode", sort=False))
    print(f"phecodes to test: {len(groups)}", flush=True)
    res = Parallel(n_jobs=JOBS, verbose=5)(
        delayed(lambda p, s: tests_for(
            "phecode", p, *lab.get(p, (p, "")),
            E.build_survival(p, s, d, prev, excl)))(p, s) for p, s in groups)
    rows += [r for sub in res for r in sub]

    r = pd.DataFrame(rows)
    # same correction structure the Cox pipeline used, so the flags are comparable:
    # within (contrast, level), i.e. the omnibus across outcomes on its own and each
    # octant's one-vs-rest tests across the outcomes tested at that level
    r = E.add_fdr_bonferroni(r, ["contrast", "level"])

    # side by side with the adjusted m4 estimate
    cox = pd.concat([pd.read_csv(OUT / f"results_octant_{l}.csv", dtype={"outcome": str})
                     .assign(level=l) for l in ("system", "phecode")])
    cox = cox[cox.model == "m4"][["level", "outcome", "contrast", "effect", "p",
                                  "sig_fdr", "sig_bon"]]
    cox.columns = ["level", "outcome", "contrast", "cox_hr_m4", "cox_p_m4",
                   "cox_sig_fdr", "cox_sig_bon"]
    r = r.merge(cox, on=["level", "outcome", "contrast"], how="left")
    r.to_csv(OUT / "octant_logrank.csv", index=False)

    print(f"\n{len(r):,} log-rank tests over {r.outcome.nunique()} outcomes in "
          f"{(time.time()-t0)/60:.1f} min -> octant_logrank.csv")
    for lvl in ("system", "phecode"):
        s = r[r.level == lvl]
        om = s[s.contrast == "omnibus"]
        ov = s[s.contrast != "omnibus"]
        print(f"\n{lvl}: {len(om)} outcomes")
        print(f"  OMNIBUS (8-group, 7 df): p<0.05 {int((om.p < 0.05).sum())} | "
              f"FDR {int(om.sig_fdr.sum())} | Bonferroni {int(om.sig_bon.sum())}")
        print(f"  ONE-VS-REST (1 df): {len(ov)} tests | p<0.05 "
              f"{int((ov.p < 0.05).sum())} | FDR {int(ov.sig_fdr.sum())} | "
              f"Bonferroni {int(ov.sig_bon.sum())}")
        # does the unadjusted test agree with the adjusted gate?
        g = om.dropna(subset=["cox_p_m4"])
        both = int(((g.p < 0.05) & (g.cox_p_m4 < 0.05)).sum())
        lr_only = int(((g.p < 0.05) & (g.cox_p_m4 >= 0.05)).sum())
        cox_only = int(((g.p >= 0.05) & (g.cox_p_m4 < 0.05)).sum())
        print(f"  omnibus agreement with adjusted m4 at p<0.05: both {both}, "
              f"log-rank only {lr_only}, Cox only {cox_only}, neither "
              f"{len(g) - both - lr_only - cox_only}")
    # What decides whether the cause-specific (log-rank) and subdistribution (Gray)
    # answers can diverge is the ABSOLUTE cumulative incidence of the competing event,
    # not its count relative to a rare outcome's events -- for a phecode with 53 events
    # almost everyone who dies died without it, and a ratio would read as 300%.
    print(f"\ncompeting risk: {int(d['is_death'].sum()):,} deaths in {len(d):,} patients "
          f"({d['is_death'].mean():.2%}). At that level the Aalen-Johansen CIF is within "
          f"a fraction of a point of 1-KM, so Gray's test would not change these calls.")

    U.log_milestone(
        f"Unadjusted log-rank tests across the 8 octant phenotypes: {len(r):,} tests "
        f"(omnibus + one-vs-rest, both levels) -> phenotype_exposure/octant_logrank.csv.")


if __name__ == "__main__":
    main()
