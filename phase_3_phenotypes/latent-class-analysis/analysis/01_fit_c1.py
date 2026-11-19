"""Stage 1 -- latent profile analysis of the physiologic domain (c1).

Primary model: K-component Gaussian mixture with an unconstrained (full) covariance
per profile, fitted by FIML so the whole 239,248-patient device-QC'd cohort is used.

Two structural choices drive this and both were settled empirically, not by taste:

* Full rather than diagonal covariance. The oximetry features correlate up to
  r = 0.91 within class; a pilot on the complete-case cohort put full covariance
  ~1.5 million BIC points ahead of diagonal at K=4. A diagonal mixture answers
  unmodelled correlation by spending extra classes on it, which is exactly the
  over-extraction we need to avoid. StepMix ships no ``gaussian_full_nan``, hence
  ``common.gmm_fiml`` (validated against sklearn to 7e-15 and against StepMix's
  ``gaussian_diag_nan`` to 1.5e-12).
* Yeo-Johnson rather than log1p. log1p leaves |skew| at 5.9; Yeo-Johnson brings the
  worst from 200 to 1.8 while staying invertible for the profile tables.

Four arms are fitted at every K so the reader can see what each choice cost:
  no_ODI4PCT         ODI4PCT correlates 0.961 with index AHI, so it is AHI in all but
                     name. Co-primary rather than sensitivity: it decides whether
                     "c1 tracks AHI severity" is a finding or a tautology.
  drop_supine_ratio  the 73%-missing ratio, absent for 100% of WatchPAT studies, so
                     its missingness is device-determined rather than MAR.
  no_AHI_derived     both engineered ratios removed -- the strict reading of the
                     study's rule that AHI is never an input.
  diag_covariance    the conventional local-independence LPA.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fitting, prep, validate  # noqa: E402
from common.config import ARM_STARTS, K_RANGE, RESULTS, SEED, TOL, WORK  # noqa: E402
from common.gmm_fiml import FimlGaussianMixture  # noqa: E402

OUT = RESULTS / "c1_physiologic"
OUT.mkdir(parents=True, exist_ok=True)

model = pd.read_parquet(WORK / "c1_model.parquet")
aux = pd.read_parquet(WORK / "c1_aux.parquet")
assert (model.MRN.to_numpy() == aux.MRN.to_numpy()).all(), "row alignment broken"
FEATS = [c for c in model.columns if c != "MRN"]
X = model[FEATS].to_numpy(float)
print(f"c1: {X.shape[0]:,} patients x {X.shape[1]} features "
      f"({np.isnan(X).mean():.2%} of cells missing)")


def make_full(K, seed, max_iter):
    return FimlGaussianMixture(K, "full", n_init=1, max_iter=max_iter, tol=TOL,
                               random_state=seed, init="kmeans")


def make_diag(K, seed, max_iter):
    return FimlGaussianMixture(K, "diag", n_init=1, max_iter=max_iter, tol=TOL,
                               random_state=seed, init="kmeans")


# ------------------------------------------------------------------ primary
print(f"\n[primary] full covariance, FIML, all {len(FEATS)} features")
t0 = time.perf_counter()
fit, models = fitting.sweep(make_full, X, K_RANGE, seed=SEED)
print(f"  sweep took {(time.perf_counter()-t0)/60:.1f} min")

print("\n[stability] split-half ARI")
stab = fitting.split_half_stability(make_full, X, K_RANGE, n_rep=10, seed=SEED)
table = fitting.apply_selection(fit, stab)

# --------------------------------------------------- co-primary and sensitivity
# ODI4PCT correlates 0.961 with index AHI. Leaving it in makes "c1 tracks AHI
# severity" nearly circular; taking it out unilaterally would discard a real
# physiologic measurement. Both are fitted and both are reported.
ARMS = {
    "no_ODI4PCT": [f for f in FEATS if f != prep.C1_AHI_SURROGATE],
    "drop_supine_ratio": [f for f in FEATS if f != "supine_nonsupine_AHI_ratio"],
    "no_AHI_derived": [f for f in FEATS if f not in prep.C1_ENGINEERED],
}
arm_fits, arm_models, arm_X = {}, {}, {}
for name, feats in ARMS.items():
    print(f"\n[arm] {name}: {len(feats)} features")
    arm_X[name] = model[feats].to_numpy(float)
    arm_fits[name], arm_models[name] = fitting.sweep(
        make_full, arm_X[name], K_RANGE, seed=SEED, **ARM_STARTS)

print("\n[arm] diag_covariance (conventional local-independence LPA)")
arm_X["diag_covariance"] = X
arm_fits["diag_covariance"], arm_models["diag_covariance"] = fitting.sweep(
    make_diag, X, K_RANGE, seed=SEED, **ARM_STARTS)

rows = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    row = {"K": K, "bic_primary": float(fit.loc[fit.K == K, "bic"].iloc[0])}
    for name in arm_X:
        row[f"ari_vs_{name}"] = float(
            adjusted_rand_score(lab, arm_models[name][K].predict(arm_X[name])))
        row[f"bic_{name}"] = float(arm_fits[name].loc[arm_fits[name].K == K, "bic"].iloc[0])
    rows.append(row)
sens = pd.DataFrame(rows)
for name, f in arm_fits.items():
    fitting.apply_selection(f).to_csv(OUT / f"c1_fit_indices_{name}.csv", index=False)

# ---------------------------------------------------------------- diagnostics
# Device composition per class at every K. The prior PPCA + k-means run produced a
# lab-PSG "missing oximetry" artifact class at k>=3; this is where we would see it.
diag_rows = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    comp = validate.composition(lab, aux.STUDY_DEVICE, "device")
    v_dev, _ = validate.cramers_v(lab, aux.STUDY_DEVICE)
    v_sup, _ = validate.cramers_v(lab, aux.supine_ratio_observed)
    v_yr, _ = validate.cramers_v(lab, aux.study_year)
    for k in range(K):
        d = {"K": K, "class": k, "n": int((lab == k).sum()),
             "cramers_v_device": round(v_dev, 4),
             "cramers_v_study_year": round(v_yr, 4),
             "cramers_v_supine_observed": round(v_sup, 4),
             "pct_supine_observed": round(float(aux.supine_ratio_observed[lab == k].mean()), 4),
             "median_study_year": float(aux.study_year[lab == k].median())}
        d.update({f"dev_{c}": float(comp.loc[k, c]) if c in comp.columns else 0.0
                  for c in comp.columns})
        diag_rows.append(d)
device_diag = pd.DataFrame(diag_rows)

# ------------------------------------------------- external validation (AHI)
# AHI was never an input. It enters only here, via 3-step BCH.
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
ahi = pd.concat(ahi_rows, ignore_index=True)
sev = pd.concat(sev_rows, ignore_index=True)

# ------------------------------------------------------------------- outputs
table.to_csv(OUT / "c1_fit_indices.csv", index=False)
stab.to_csv(OUT / "c1_stability.csv", index=False)
sens.to_csv(OUT / "c1_sensitivity.csv", index=False)
device_diag.to_csv(OUT / "c1_device_diagnostics.csv", index=False)
ahi.to_csv(OUT / "c1_ahi_validation.csv", index=False)
sev.to_csv(OUT / "c1_ahi_severity_mix.csv", index=False)

# class profiles on the ORIGINAL clinical scale, so the tables are readable
raw = pd.read_csv(Path(WORK).parent / "data" / "c1_physiologic" / "c1_features_maximal.csv",
                  dtype={"MRN": str})
raw = raw[~raw.STUDY_DEVICE.isin(prep.C1_EXCLUDE_DEVICES)].reset_index(drop=True)[FEATS]
assert len(raw) == len(X), "profile frame must align with the model matrix"
prof = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    p = fitting.class_profile(raw, lab, K)
    p["K"] = K
    prof.append(p)
pd.concat(prof, ignore_index=True).to_csv(OUT / "c1_class_profiles.csv", index=False)

with open(WORK / "c1_models.pkl", "wb") as fh:
    pickle.dump({"primary": models, "arms": arm_models, "features": FEATS,
                 "arm_features": ARMS}, fh)

(OUT / "c1_fit_summary.json").write_text(json.dumps({
    "n_patients": int(len(X)), "n_features": len(FEATS), "features": FEATS,
    "excluded_devices": prep.C1_EXCLUDE_DEVICES,
    "dropped_features": prep.C1_DROPPED_REASONS,
    "covariance": "full", "missing_data": "FIML (MAR)",
    "arms": {k: v for k, v in ARMS.items()},
    "bic_elbow_K": int(table.bic_elbow_K.iloc[0]),
    "recommended_K": int(table.loc[table.recommended, "K"].iloc[0])
    if table.recommended.any() else None,
    "seed": SEED,
}, indent=2))

print("\n=== c1 fit indices (primary) ===")
print(table[["K", "loglik", "n_params", "bic", "sabic", "rel_entropy",
             "min_class_frac", "min_component_sd", "degenerate", "split_half_ari",
             "ll_replicated", "recommended"]].to_string(index=False))
print(f"\nBIC elbow at K = {table.bic_elbow_K.iloc[0]}")
print("\n=== arms: ARI against the primary solution ===")
print(sens[["K"] + [c for c in sens.columns if c.startswith("ari_")]].to_string(index=False))
print("\n=== device / era diagnostics ===")
print(device_diag.groupby("K")[["cramers_v_device", "cramers_v_study_year",
                                "cramers_v_supine_observed"]].first().to_string())
print(f"\nwrote {OUT}")
