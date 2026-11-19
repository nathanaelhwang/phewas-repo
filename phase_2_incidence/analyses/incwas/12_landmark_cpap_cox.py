# %% [markdown]
# Step 12 - Landmark CPAP-adherence Cox (immortal-time-corrected sensitivity analysis).
#
# A grace-period LANDMARK design (see 09/10, which were biased): follow-up starts at index+GRACE; any
# phecode onset in (index, index+GRACE] is reclassified prevalent; only patients event-free/alive/
# enrolled past the landmark enter; and each patient's CPAP adherence is a fixed label assigned at the
# landmark. Exposure = 5-level adherence {4+ (ref), 2-4, 0-2 hr/night, never-started, unknown}, among
# OSA patients only, adjusted for m3/m4 baseline covariates + OSA severity. Run for GRACE = 90 and 180
# days (180 = genuine: the therapy-anchored 90-day compliance window has closed pre-landmark for ~82%;
# 90 = approximate, forward-peek for ~81%).
#
# CAVEAT: the landmark removes IMMORTAL-TIME bias only. Healthy-adherer / confounding-by-indication
# remains -> associational, not causal. A negative-control-outcome panel (benign phecodes unrelated to
# CPAP) quantifies the residual-confounding floor: HRs there should be ~1.0.
#
# Usage: python 12_landmark_cpap_cox.py [--jobs 8]
# Emits analysis/results_cpap_landmark_{90,180}.csv. DE-IDENTIFIED.

# %%
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

CPAP = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
GRACE_WINDOWS = [90, 180]
ADH_DUMMY = {"adh_0_2": "0-2 hr/night", "adh_2_4": "2-4 hr/night",
             "adh_never": "never-started", "adh_unknown": "unknown"}   # all vs 4+ hr/night (reference)
NEG_CATS = {"sense organs", "dermatologic", "injuries & poisonings"}
SEV_CONTRASTS = ["omnibus", "trend", "mild_vs_none", "moderate_vs_none",
                 "severe_vs_none", "ahi_ge5", "ahi_ge15"]

_ADH = None


def foll_adh():
    """Cache-per-process: follow-up frame + the CPAP fields (mean_pap_usage, setup_date). Adherence is
    grace-window-dependent, so it is computed per window in fit_one, not here."""
    global _ADH
    if _ADH is None:
        foll, blocks = E.prep_followup()
        cpap = pd.read_parquet(CPAP).set_index("id")
        foll = foll.copy()
        foll["mean_pap_usage"] = foll["id"].map(cpap["mean_pap_usage"])
        foll["setup_date"] = foll["id"].map(cpap["setup_date"])
        _ADH = (foll, blocks)
    return _ADH


def fit_one(phe, pheno, cat, ev_slice, window, is_neg):
    foll, blocks = foll_adh()
    prev, excl, grace = E.prev_index(), E.exclude_map(), E.grace_index(window)
    d = E.build_survival(phe, ev_slice, foll, prev, excl, grace_days=window, grace_by_phe=grace)
    d = d[d["severity"] != "None"].copy()             # OSA only; adherence undefined for No-OSA
    n_at, n_ev = len(d), int(d["event"].sum())
    if n_ev < U.MIN_EVENTS:
        return []
    # adherence derived strictly from the [index, index+window] window
    d["adherence"] = E.adherence_landmark(d["mean_pap_usage"], d["setup_date"], d["idx"], window)
    for c, lvl in ADH_DUMMY.items():
        d[c] = (d["adherence"] == lvl).astype(float)
    n_ref = int((d["adherence"] == "4+ hr/night").sum())
    if n_ref < U.MIN_EVENTS:                           # within-window levels not estimable (e.g. grace=90)
        r = E.row(phe, "adherence_levels_infeasible", "m4", n_at, n_ev,
                  np.nan, np.nan, np.nan, np.nan, np.nan, note=f"ref(4+h) n={n_ref}")
        r["phenotype"], r["category"], r["window_days"], r["neg_control"] = pheno, cat, window, is_neg
        return [r]
    out = []
    adh_cols = [c for c in ADH_DUMMY if d[c].nunique() > 1]
    for model in ("m3", "m4"):
        cov = blocks["models"][model] + ["severity_ord"]    # adjust baseline health + OSA severity
        note = "epv<10" if n_ev < U.EPV * max(len(adh_cols) + len(cov), 1) else ""
        try:
            cph, kept = E._fit(d, adh_cols + cov)
            stat, p_om, _ = E._wald_omnibus(cph, adh_cols)
            out.append(E.row(phe, "adherence_omnibus", model, n_at, n_ev, stat,
                             np.nan, np.nan, np.nan, p_om, note=note))
            for c in adh_cols:
                if c in kept:
                    e, se, lo, hi, p = E._eff(cph, c)
                    nt = note + ("|unstable" if (abs(np.log(e)) > 10 or se > 10) else "")
                    out.append(E.row(phe, f"{ADH_DUMMY[c]}_vs_4plus", model, n_at, n_ev, e, se, lo, hi, p, note=nt))
        except Exception as ex:
            out.append(E.row(phe, "adherence_omnibus", model, n_at, n_ev,
                             np.nan, np.nan, np.nan, np.nan, np.nan, note=str(ex)[:40]))
    for r in out:
        r["phenotype"], r["category"], r["window_days"], r["neg_control"] = pheno, cat, window, is_neg
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    U.ensure_dirs()

    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: (r.phenotype, r.category) for r in dim.itertuples()}
    res = pd.read_csv(U.ANALYSIS / "results_cox.csv", dtype={"phecode": str})
    m4 = res[(res.model == "m4") & res.contrast.isin(SEV_CONTRASTS)]
    fdr = sorted(m4[m4.sig_fdr]["phecode"].unique())

    # negative controls: benign-category phecodes, >=500 incident, NOT OSA-FDR-significant
    ev_meta = pd.read_parquet(U.EVENTS, columns=["phecode", "has_pre", "second_post_date"])
    ev_meta["phecode"] = ev_meta["phecode"].astype(str)
    inc = ev_meta[(~ev_meta.has_pre) & ev_meta.second_post_date.notna()].groupby("phecode").size()
    negc = [p for p in inc.index if lab.get(p, ("", ""))[1] in NEG_CATS
            and p not in set(fdr) and inc[p] >= 500]
    negc = sorted(negc, key=lambda p: -inc[p])[:10]
    phelist = [(p, False) for p in fdr] + [(p, True) for p in negc]
    print(f"FDR phecodes: {len(fdr)}; negative controls: {len(negc)} -> "
          f"{[lab.get(p, ('', ''))[0][:20] for p in negc]}", flush=True)

    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    keep_phe = {p for p, _ in phelist}
    ev_groups = {p: g for p, g in ev.groupby("phecode", sort=False) if p in keep_phe}

    # adherence category distribution among OSA landmark-survivors, per window (shows the 90d collapse)
    foll0, _ = foll_adh()
    for w in GRACE_WINDOWS:
        land = foll0["idx"] + pd.to_timedelta(w, unit="D")
        osa = foll0[(foll0["severity"] != "None") & (foll0["censor_date"] > land)]
        adh = E.adherence_landmark(osa["mean_pap_usage"], osa["setup_date"], osa["idx"], w)
        print(f"[grace {w}d] within-window adherence among OSA landmark-survivors (n={len(osa):,}): "
              + ", ".join(f"{k}={v:,}" for k, v in adh.value_counts().items()), flush=True)

    tasks = [(p, isneg, w) for w in GRACE_WINDOWS for (p, isneg) in phelist if p in ev_groups]
    t0 = time.time()
    res_rows = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(fit_one)(p, lab.get(p, ("", ""))[0], lab.get(p, ("", ""))[1], ev_groups[p], w, isneg)
        for (p, isneg, w) in tasks)
    allrows = [r for sub in res_rows for r in sub]
    D = pd.DataFrame(allrows)
    print(f"fit {len(tasks)} phecode-window tasks in {(time.time()-t0)/60:.1f} min", flush=True)

    for window in GRACE_WINDOWS:
        d = D[D.window_days == window].copy()
        # FDR/Bonferroni within (contrast, model), among the intended tests only (exclude neg controls)
        real = E.add_fdr_bonferroni(d[~d.neg_control], ["contrast", "model"])
        neg = d[d.neg_control].assign(sig_bon=False, sig_fdr=False)
        d = pd.concat([real, neg], ignore_index=True)
        cols = ["phecode", "phenotype", "category", "window_days", "neg_control", "contrast", "model",
                "n_atrisk", "n_events", "effect_type", "effect", "se", "ci_low", "ci_high", "p",
                "neglog10p", "sig_bon", "sig_fdr", "unstable"]
        out = U.ANALYSIS / f"results_cpap_landmark_{window}.csv"
        d[cols].to_csv(out, index=False)

        h = d[(d.contrast == "0-2 hr/night_vs_4plus") & (d.model == "m4") & (~d.neg_control) & d.effect.notna()]
        h = h.reindex(h["effect"].sort_values(ascending=False).index).head(12)
        print(f"\n[grace {window}d] {len(d):,} rows / {d.phecode.nunique()} phecodes -> {out.name}")
        print(f"[grace {window}d] worst-vs-best adherence (0-2h vs 4+h), m4, top HR:")
        for _, r in h.iterrows():
            s = "*" if r["sig_fdr"] else " "
            print(f"  {s} {str(r['phenotype'])[:32]:32s} HR={r['effect']:.2f} "
                  f"({r['ci_low']:.2f}-{r['ci_high']:.2f}) p={r['p']:.1e} n_ev={int(r['n_events'])}")
        nc = d[(d.contrast == "0-2 hr/night_vs_4plus") & (d.model == "m4") & (d.neg_control) & d.effect.notna()]
        if len(nc):
            print(f"[grace {window}d] NEG CONTROLS (0-2 vs 4+ should be ~1.0): median HR="
                  f"{nc['effect'].median():.2f} range {nc['effect'].min():.2f}-{nc['effect'].max():.2f}, "
                  f"{int((nc['effect'] > 1).sum())}/{len(nc)} with HR>1")

    U.log_milestone("Step 12 landmark CPAP Cox: 5-level adherence vs incident disease, immortal-time-"
                    "corrected (grace 90 & 180d), m3/m4 + severity-adjusted, FDR, + negative-control panel.")


if __name__ == "__main__":
    main()
