"""Stage 2 -- latent class cluster analysis of the symptom domain (c2).

Mixed-mode LCCA is the right family here, but not on the 90 raw columns. Two of
those columns are not what they look like and two instruments are not independent
items, so the primary model runs on derived indicators:

* BEDTIME_* and WAKETIME_* are each ONE 12-level nominal variable that was one-hot
  expanded (row sums are exactly 1). Twenty-four mutually exclusive Bernoulli
  indicators in a conditional-independence LCA is a deterministic local-dependence
  violation, so both blocks are folded back into single chronotype nominals.
* ESS (alpha 0.86), ISI (0.85) and FOSQ (0.84) are highly internally consistent.
  Entered item-by-item under conditional independence they force the model to spend
  classes on per-instrument severity strata. The primary model therefore uses scale
  scores, which is also what the OSA symptom-subtype literature does, so the classes
  are comparable to published phenotypes rather than to nothing.

The item-level model is fitted as a sensitivity arm (with the one-hot fix still
applied, since that one is a bug rather than a choice) and compared by ARI.

Missingness is block-structured by intake-form version. FIML is the primary
treatment; the complete-case arm and the class x form-pattern Cramer's V are what
tell us whether the classes are symptom phenotypes or questionnaire cohorts.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fitting, prep, validate  # noqa: E402
from common.config import ARM_STARTS, K_RANGE, RESULTS, SEED, WORK  # noqa: E402

OUT = RESULTS / "c2_symptomatic"
OUT.mkdir(parents=True, exist_ok=True)

scales = pd.read_parquet(WORK / "c2_model_scales.parquet")
items = pd.read_parquet(WORK / "c2_model_items.parquet")
aux = pd.read_parquet(WORK / "c2_aux.parquet")
prep_report = json.loads((RESULTS / "00_prep_report.json").read_text())
assert (scales.MRN.to_numpy() == aux.MRN.to_numpy()).all()

MM = pd.DataFrame(prep_report["c2_symptomatic"]["measurement_models"])
GROUPS = {m: MM.loc[MM.model == m, "indicator"].tolist()
          for m in ("gaussian", "bernoulli", "multinoulli")}


def _descriptor(gauss, binary, nominal):
    """StepMix nested measurement descriptor; block order must match column order.

    Which block an indicator lands in was decided in stage 0 from its own support,
    not from its nominal item type. The reason is that a Gaussian component can
    collapse onto a point mass and send the likelihood to infinity -- and several
    c2 items carry 75-90% of their mass on one response. StepMix's
    ``gaussian_diag_nan`` does not guard against this: given a zero atom plus
    missingness it silently returns a NaN log-likelihood rather than raising, which
    would poison every information criterion downstream.
    """
    d = {}
    if gauss:
        d["gaussian"] = {"model": "gaussian_diag_nan", "n_columns": len(gauss)}
    if binary:
        d["binary"] = {"model": "bernoulli_nan", "n_columns": len(binary)}
    if nominal:
        d["nominal"] = {"model": "multinoulli_nan", "n_columns": len(nominal),
                        "integer_codes": True}
    return d


def build_matrix(df, gauss, binary, nominal):
    cols = gauss + binary + nominal
    X = df[cols].to_numpy(float)
    # z-score the Gaussian block only, on observed cells, so indicators on very
    # different ranges (ESS 0-24 vs SLEEPTIME 2-12 vs FOSQ 0-40) contribute
    # comparably rather than by accident of units
    if gauss:
        A = X[:, : len(gauss)]
        X[:, : len(gauss)] = (A - np.nanmean(A, 0)) / np.nanstd(A, 0)
    return X, cols


GAUSS, BIN, NOM = GROUPS["gaussian"], GROUPS["bernoulli"], GROUPS["multinoulli"]
X, COLS = build_matrix(scales, GAUSS, BIN, NOM)
DESC = _descriptor(GAUSS, BIN, NOM)
print(f"c2 scale-score arm: {X.shape[0]:,} patients x {X.shape[1]} indicators "
      f"({len(GAUSS)} gaussian / {len(BIN)} bernoulli / {len(NOM)} multinoulli); "
      f"{np.isnan(X).mean():.1%} of cells missing")


def make_scales(K, seed, max_iter):
    return StepMix(n_components=K, measurement=DESC, n_init=1, max_iter=max_iter,
                   abs_tol=1e-8, random_state=seed, progress_bar=0,
                   init_params="random")


# ------------------------------------------------------------------ primary
print("\n[primary] mixed-mode LCCA on scale scores, FIML")
t0 = time.perf_counter()
fit, models = fitting.sweep(make_scales, X, K_RANGE, seed=SEED)
print(f"  sweep took {(time.perf_counter()-t0)/60:.1f} min")

print("\n[stability] split-half ARI")
stab = fitting.split_half_stability(make_scales, X, K_RANGE, n_rep=10, seed=SEED)
table = fitting.apply_selection(fit, stab)

# ------------------------------------------------------- sensitivity arms
print("\n[sensitivity A] item-level LCCA (one-hot and variance-collapse fixes retained)")
IG, item_meta = prep.assign_measurement_models(
    items, [c for c in items.columns if c != "MRN"],
    force_binary=prep.C2_SINGLE_BINARY + prep.C2_CC)
for c in ("BEDTIME_band", "WAKETIME_band"):
    for g in IG.values():
        if c in g:
            g.remove(c)
IG["multinoulli"] += ["BEDTIME_band", "WAKETIME_band"]
Xi, ITEM_COLS = build_matrix(items, IG["gaussian"], IG["bernoulli"], IG["multinoulli"])
DESC_I = _descriptor(IG["gaussian"], IG["bernoulli"], IG["multinoulli"])
print(f"  {Xi.shape[1]} indicators ({len(IG['gaussian'])} gaussian / "
      f"{len(IG['bernoulli'])} bernoulli / {len(IG['multinoulli'])} multinoulli)")


def make_items(K, seed, max_iter):
    return StepMix(n_components=K, measurement=DESC_I, n_init=1, max_iter=max_iter,
                   abs_tol=1e-8, random_state=seed, progress_bar=0,
                   init_params="random")


fit_i, models_i = fitting.sweep(make_items, Xi, K_RANGE, seed=SEED, **ARM_STARTS)

print("\n[sensitivity B] complete cases only (all 12 instrument blocks present)")
cc = aux.n_blocks_present.to_numpy() == aux.n_blocks_present.max()
print(f"  n = {cc.sum():,} ({cc.mean():.1%} of the cohort)")
fit_c, models_c = fitting.sweep(make_scales, X[cc], K_RANGE, seed=SEED, **ARM_STARTS)

rows = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    rows.append({
        "K": K,
        "ari_vs_item_level": float(adjusted_rand_score(lab, models_i[K].predict(Xi))),
        "ari_vs_complete_case": float(
            adjusted_rand_score(lab[cc], models_c[K].predict(X[cc]))),
        "bic_scales": float(fit.loc[fit.K == K, "bic"].iloc[0]),
        "bic_items": float(fit_i.loc[fit_i.K == K, "bic"].iloc[0]),
    })
sens = pd.DataFrame(rows)

# ---------------------------------------------------------------- diagnostics
# THE artifact check for this domain: if class tracks intake-form version, the
# "symptom phenotypes" are questionnaire cohorts.
form_rows = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    v_form, _ = validate.cramers_v(lab, aux.form_pattern)
    v_nblk, _ = validate.cramers_v(lab, aux.n_blocks_present)
    v_year, _ = validate.cramers_v(lab, aux.study_year.fillna(-1))
    form_rows.append({"K": K, "cramers_v_form_pattern": round(v_form, 4),
                      "cramers_v_n_blocks": round(v_nblk, 4),
                      "cramers_v_study_year": round(v_year, 4),
                      "mean_blocks_by_class": {
                          int(k): round(float(aux.n_blocks_present[lab == k].mean()), 2)
                          for k in range(K)}})
form_diag = pd.DataFrame(form_rows)

# Residual local dependence among the surviving indicators. Aggregating ESS/ISI/FOSQ
# to totals removes the worst of it but does not remove all of it, and the honest
# thing is to measure what is left rather than assert it is gone.
dep_rows = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    obs = pd.DataFrame(X, columns=COLS).corr(method="spearman", min_periods=500)
    within = (pd.DataFrame(X, columns=COLS).groupby(lab)
              .corr(method="spearman", min_periods=200).groupby(level=1).mean())
    iu = np.triu_indices(len(COLS), 1)
    d = pd.DataFrame({"K": K, "indicator_a": np.array(COLS)[iu[0]],
                      "indicator_b": np.array(COLS)[iu[1]],
                      "marginal_rho": obs.to_numpy()[iu],
                      "within_class_rho": within.reindex(COLS)[COLS].to_numpy()[iu]})
    dep_rows.append(d.reindex(d.within_class_rho.abs().sort_values(ascending=False).index)
                    .head(20))
resid_dep = pd.concat(dep_rows, ignore_index=True)

# ------------------------------------------------- external validation (AHI)
ahi_rows, sev_rows = [], []
for K in K_RANGE:
    if K == 1:
        continue
    P = models[K].predict_proba(X)
    lab = P.argmax(1)
    d = validate.distal_bch(P, aux.index_ahi.to_numpy())
    d["K"] = K
    kw = validate.kw_effect(aux.index_ahi.to_numpy(), lab)
    d["kw_H"] = round(kw["H"], 1)
    d["kw_p"] = kw["p"]
    d["kw_epsilon_sq"] = round(kw["epsilon_sq"], 4)
    ahi_rows.append(d)
    s = validate.severity_mix(aux.index_ahi.to_numpy(), lab).reset_index()
    s["K"] = K
    sev_rows.append(s)

# ------------------------------------------------------------------- outputs
table.to_csv(OUT / "c2_fit_indices.csv", index=False)
stab.to_csv(OUT / "c2_stability.csv", index=False)
sens.to_csv(OUT / "c2_sensitivity.csv", index=False)
form_diag.to_csv(OUT / "c2_form_artifact_check.csv", index=False)
resid_dep.to_csv(OUT / "c2_residual_dependence.csv", index=False)
item_meta.to_csv(OUT / "c2_item_measurement_models.csv", index=False)
pd.concat(ahi_rows, ignore_index=True).to_csv(OUT / "c2_ahi_validation.csv", index=False)
pd.concat(sev_rows, ignore_index=True).to_csv(OUT / "c2_ahi_severity_mix.csv", index=False)

prof = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    p = fitting.class_profile(scales[COLS], lab, K)      # original indicator scales
    p["K"] = K
    prof.append(p)
pd.concat(prof, ignore_index=True).to_csv(OUT / "c2_class_profiles.csv", index=False)

with open(WORK / "c2_models.pkl", "wb") as fh:
    pickle.dump({"primary": models, "items": models_i, "complete_case": models_c,
                 "cols": COLS, "item_cols": ITEM_COLS, "desc": DESC,
                 "gaussian": GAUSS, "bernoulli": BIN, "multinoulli": NOM}, fh)

(OUT / "c2_fit_summary.json").write_text(json.dumps({
    "n_patients": int(len(X)), "n_indicators": len(COLS),
    "gaussian": GAUSS, "bernoulli": BIN, "multinoulli": NOM,
    "missing_data": "FIML (StepMix *_nan emission models)",
    "bic_elbow_K": int(table.bic_elbow_K.iloc[0]),
    "recommended_K": int(table.loc[table.recommended, "K"].iloc[0])
    if table.recommended.any() else None,
    "seed": SEED,
}, indent=2))

print("\n=== c2 fit indices ===")
print(table[["K", "loglik", "n_params", "bic", "sabic", "rel_entropy",
             "min_class_frac", "min_component_sd", "degenerate", "split_half_ari",
             "ll_replicated", "recommended"]].to_string(index=False))
print(f"\nBIC elbow at K = {table.bic_elbow_K.iloc[0]}")
print("\n=== form-version artifact check (Cramer's V of class x intake form) ===")
print(form_diag.to_string(index=False))
print(f"\nwrote {OUT}")
