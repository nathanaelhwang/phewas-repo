"""Stage 3 -- latent class analysis of the comorbidity domain (c3).

Indicators are 193 binary chronic-disease phecodes, rolled up from the 656 raw
codes onto their integer parent and filtered to exclude symptom, injury, pregnancy,
refraction and administrative categories, process-of-care markers (encounter /
screening / personal history / abnormal finding), and phecode root 327 "Sleep
disorders" -- which absorbs sleep apnea (24%) and OSA (9%) and is therefore circular
with the exposure, as well as insomnia and restless legs, which would manufacture a
c2-c3 association out of nothing. Roll-up matters here for a reason specific
to this build: pyPheWAS maps each ICD to exactly one phecode without hierarchy
roll-up, so parent and child codes are largely disjoint (P(parent=1 | child=1) is
only 0.27) and a single condition ends up fragmented across sibling codes.

The domain's known hazard is that phecode presence partly measures healthcare
contact rather than disease -- the prior clustering run's high-multimorbidity
cluster had *lower* AHI. The distinct-code count is therefore carried as an
external descriptor (never an input) so a high-utilisation class can be labelled
as such rather than over-read as multimorbidity.

Local independence is assumed and is certainly violated (hypertension, hyperlipidemia
and diabetes co-occur beyond what any class explains). StepMix cannot add direct
effects, so the violation is measured rather than modelled: script 04 reports the
largest bivariate residuals so the reader can see which pairs the model fails on.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stepmix.stepmix import StepMix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fitting, validate  # noqa: E402
from common.config import K_RANGE, RESULTS, SEED, WORK  # noqa: E402

OUT = RESULTS / "c3_comorbidities"
OUT.mkdir(parents=True, exist_ok=True)

model = pd.read_parquet(WORK / "c3_model.parquet")
aux = pd.read_parquet(WORK / "c3_aux.parquet")
meta = pd.read_csv(RESULTS / "c3_indicator_selection.csv", dtype={"root": str})
assert (model.MRN.to_numpy() == aux.MRN.to_numpy()).all()
FEATS = [c for c in model.columns if c != "MRN"]
X = model[FEATS].to_numpy(np.float64)
print(f"c3: {X.shape[0]:,} patients x {X.shape[1]} binary indicators "
      f"(mean prevalence {X.mean():.3f}); no missing data")


def make_lca(K, seed, max_iter):
    return StepMix(n_components=K, measurement="bernoulli", n_init=1,
                   max_iter=max_iter, abs_tol=1e-8, random_state=seed,
                   progress_bar=0, init_params="random")


# ------------------------------------------------------------------ primary
print(f"\n[primary] Bernoulli LCA on {len(FEATS)} chronic-disease phecodes")
t0 = time.perf_counter()
fit, models = fitting.sweep(make_lca, X, K_RANGE, seed=SEED)
print(f"  sweep took {(time.perf_counter()-t0)/60:.1f} min")

print("\n[stability] split-half ARI")
stab = fitting.split_half_stability(make_lca, X, K_RANGE, n_rep=10, seed=SEED,
                                    max_iter=150)
table = fitting.apply_selection(fit, stab)

# ------------------------------------------- local-dependence diagnostics
def bivariate_residuals(Xb, P, feats, top=25):
    """Standardised chi-square residuals for every indicator pair.

    Under local independence the model-implied pair frequency is
    sum_k w_k * p_ik * p_jk. Large residuals mark pairs whose co-occurrence the
    latent classes fail to explain -- the standard LCA local-dependence check.
    """
    w = P.mean(0)
    p = (P.T @ Xb) / P.sum(0)[:, None]              # class x indicator probabilities
    exp = (w[:, None, None] * p[:, :, None] * p[:, None, :]).sum(0)
    obs = (Xb.T @ Xb) / len(Xb)
    n = len(Xb)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (obs - exp) * np.sqrt(n) / np.sqrt(np.clip(exp * (1 - exp), 1e-12, None))
    iu = np.triu_indices(len(feats), 1)
    out = pd.DataFrame({
        "indicator_a": np.array(feats)[iu[0]], "indicator_b": np.array(feats)[iu[1]],
        "observed": obs[iu], "expected": exp[iu], "std_residual": resid[iu],
    })
    return out.reindex(out.std_residual.abs().sort_values(ascending=False).index).head(top)


# --------------------------------------- is the inverse AHI gradient an artifact?
# The prior clustering run concluded that comorbidity burden runs *inversely* to AHI.
# That is exactly what a coding-volume confound looks like: patients with more
# healthcare contact accumulate more codes of every kind, and are also more likely to
# be worked up for milder sleep-disordered breathing. Holding coding volume fixed
# tests the confound directly.
chronic = aux.n_phecodes_kept.to_numpy(float)
total = aux.n_phecodes_all.to_numpy(float)
# The adjustment variable must be DISJOINT from the exposure of interest. Chronic
# burden is a subset of total burden, so residualising the part on the whole is
# close to correlating AHI with the negative of the complement -- an arithmetic
# identity dressed up as a finding. Adjust for the non-chronic residue instead.
nonchronic = total - chronic
ahi_v = aux.index_ahi.to_numpy(float)
ok = np.isfinite(ahi_v)


def _pcorr(a, b, given):
    """Pearson correlation of a and b after linearly removing `given` from each."""
    A = np.column_stack([np.ones_like(given), given])
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


burden = {
    "corr_total_burden_vs_ahi": round(float(np.corrcoef(total[ok], ahi_v[ok])[0, 1]), 4),
    "corr_chronic_burden_vs_ahi": round(float(np.corrcoef(chronic[ok], ahi_v[ok])[0, 1]), 4),
    "corr_nonchronic_burden_vs_ahi": round(
        float(np.corrcoef(nonchronic[ok], ahi_v[ok])[0, 1]), 4),
    "partial_corr_chronic_vs_ahi_given_nonchronic": round(
        _pcorr(chronic[ok], ahi_v[ok], nonchronic[ok]), 4),
    "note": ("Adjustment is on the DISJOINT non-chronic code count (total minus "
             "chronic), not on total burden, because chronic burden is a subset of "
             "total and residualising a part on its whole is near an identity. If the "
             "crude correlations are negative but the adjusted one is positive, the "
             "prior run's inverse comorbidity-AHI gradient is substantially a "
             "coding-volume artifact. Note the non-chronic count is itself plausibly "
             "a collider (both disease and care-seeking cause coding), so this is "
             "suggestive rather than a causal decomposition."),
}
print("\n[burden vs AHI] " + json.dumps(burden, indent=2))

# ------------------------------------------------- external validation (AHI)
# AHI is never an input; it enters only via 3-step BCH. The utilisation index is
# reported next to it because the two together are what distinguish a genuine
# multimorbidity class from a high-healthcare-contact class.
ahi_rows, sev_rows, util_rows, bvr_rows = [], [], [], []
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

    u = validate.distal_bch(P, aux.n_phecodes_all.to_numpy().astype(float))
    u["K"] = K
    u = u.rename(columns={"bch_mean": "bch_mean_n_phecodes",
                          "naive_mean": "naive_mean_n_phecodes",
                          "naive_median": "naive_median_n_phecodes"})
    util_rows.append(u)

    b = bivariate_residuals(X, P, FEATS)
    b["K"] = K
    bvr_rows.append(b)

# ------------------------------------------------------------------- outputs
table.to_csv(OUT / "c3_fit_indices.csv", index=False)
stab.to_csv(OUT / "c3_stability.csv", index=False)
pd.concat(ahi_rows, ignore_index=True).to_csv(OUT / "c3_ahi_validation.csv", index=False)
pd.concat(sev_rows, ignore_index=True).to_csv(OUT / "c3_ahi_severity_mix.csv", index=False)
pd.concat(util_rows, ignore_index=True).to_csv(OUT / "c3_utilisation_validation.csv",
                                               index=False)
pd.concat(bvr_rows, ignore_index=True).to_csv(OUT / "c3_bivariate_residuals.csv",
                                              index=False)

label_of = dict(zip(meta.root, meta.label))
prof = []
for K in K_RANGE:
    if K == 1:
        continue
    lab = models[K].predict(X)
    p = fitting.class_profile(model[FEATS], lab, K)     # means of binaries = prevalence
    p["K"] = K
    p["phenotype"] = p.indicator.map(label_of)
    p = p.rename(columns={"mean": "class_prevalence", "cohort_mean": "cohort_prevalence"})
    p["prevalence_ratio"] = p.class_prevalence / p.cohort_prevalence.replace(0, np.nan)
    prof.append(p)
pd.concat(prof, ignore_index=True).to_csv(OUT / "c3_class_profiles.csv", index=False)

with open(WORK / "c3_models.pkl", "wb") as fh:
    pickle.dump({"primary": models, "features": FEATS}, fh)

(OUT / "c3_fit_summary.json").write_text(json.dumps({
    "n_patients": int(len(X)), "n_indicators": len(FEATS),
    "measurement": "bernoulli (local independence assumed)",
    "burden_vs_ahi": burden,
    "bic_elbow_K": int(table.bic_elbow_K.iloc[0]),
    "recommended_K": int(table.loc[table.recommended, "K"].iloc[0])
    if table.recommended.any() else None,
    "seed": SEED,
}, indent=2))

print("\n=== c3 fit indices ===")
print(table[["K", "loglik", "n_params", "bic", "sabic", "rel_entropy",
             "min_class_frac", "split_half_ari", "ll_replicated",
             "recommended"]].to_string(index=False))
print(f"\nBIC elbow at K = {table.bic_elbow_K.iloc[0]}")
print(f"\nwrote {OUT}")
