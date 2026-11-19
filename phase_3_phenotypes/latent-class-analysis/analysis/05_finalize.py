"""Stage 5 -- finalise the chosen K per domain and write the deliverables.

Reads ``results/K_CHOICES.json`` (written by the user after reading
``results/MODEL_SELECTION.md``), then for each domain: labels the classes, writes
per-patient assignments with posterior probabilities, scores the shared cohort with
the maximal-cohort parameters, cross-tabulates the three domains, and compares
against the prior PPCA + k-means partition.

Classes are estimated on the maximal cohort and the shared cohort is *scored* with
those parameters rather than re-estimated, so the three domains describe the same
latent variables in both cohorts and the cross-tabulation is between comparable
things.
"""
from __future__ import annotations

import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import validate  # noqa: E402
from common.config import RESULTS, WORK  # noqa: E402

CHOICES_PATH = RESULTS / "K_CHOICES.json"
if not CHOICES_PATH.exists():
    rec = json.loads((RESULTS / "K_RECOMMENDED.json").read_text())
    raise SystemExit(
        f"{CHOICES_PATH} not found. Read results/MODEL_SELECTION.md, then write the "
        f"chosen K per domain, e.g.\n"
        + json.dumps({k: v["recommended_K"] for k, v in rec.items()}, indent=2))
CHOICES = json.loads(CHOICES_PATH.read_text())
print("K choices:", CHOICES)

LABELS_PATH = RESULTS / "CLASS_LABELS.json"
LABELS = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}

PRIOR = Path(r"C:\Users\natha\Projects\work\phewas\clustering\results")
SPEC = {
    "c1_physiologic": dict(prefix="c1", models="c1_models.pkl", matrix="c1_model.parquet",
                           aux="c1_aux.parquet"),
    "c2_symptomatic": dict(prefix="c2", models="c2_models.pkl", matrix="c2_model_scales.parquet",
                           aux="c2_aux.parquet"),
    "c3_comorbidities": dict(prefix="c3", models="c3_models.pkl", matrix="c3_model.parquet",
                             aux="c3_aux.parquet"),
}

shared = set(pd.read_parquet(WORK / "shared_mrns.parquet").MRN)
assignments, final = {}, {}

for domain, spec in SPEC.items():
    K = CHOICES.get(domain)
    if not K:
        print(f"[{domain}] skipped (no K chosen)")
        continue
    out = RESULTS / domain
    with open(WORK / spec["models"], "rb") as fh:
        blob = pickle.load(fh)
    model = blob["primary"][K]
    mat = pd.read_parquet(WORK / spec["matrix"])
    aux = pd.read_parquet(WORK / spec["aux"])

    cols = blob.get("cols") or [c for c in mat.columns if c != "MRN"]
    X = mat[cols].to_numpy(float)
    if domain == "c2_symptomatic":                      # Gaussian block was z-scored
        ng = len(blob["gaussian"])
        A = X[:, :ng]
        X[:, :ng] = (A - np.nanmean(A, 0)) / np.nanstd(A, 0)

    P = model.predict_proba(X)
    lab = P.argmax(1)

    # Label classes by size so the numbering is stable across reruns, and record the
    # indicators that most distinguish each one so the label is interpretable.
    order = np.argsort(-np.bincount(lab, minlength=K))
    remap = {int(o): i for i, o in enumerate(order)}
    lab = np.array([remap[int(v)] for v in lab])
    P = P[:, order]

    prof = pd.read_csv(out / f"{spec['prefix']}_class_profiles.csv")
    prof = prof[prof.K == K]
    val = "prevalence_ratio" if "prevalence_ratio" in prof.columns else "std_diff"
    marks = {}
    # rank by distance from the cohort value, so a class is described by what makes it
    # different rather than by whatever indicator happens to be most prevalent
    key = (prof[val] - 1).abs() if val == "prevalence_ratio" else prof[val].abs()
    prof = prof.assign(_rank=key)
    for k in range(K):
        sub = prof[prof["class"] == order[k]].dropna(subset=[val])
        top = sub.nlargest(6, "_rank")
        marks[k] = [f"{r.phenotype} ({r.indicator})" if "phenotype" in top.columns
                    and isinstance(r.phenotype, str) else str(r.indicator)
                    for r in top.itertuples()]

    a = pd.DataFrame({"MRN": mat.MRN.to_numpy(), "class": lab,
                      "max_posterior": P.max(1)})
    # Optional human-readable names. Kept in a separate, hand-written file rather than
    # derived on the fly: a label is an interpretation of the profile, and it should be
    # auditable against that profile instead of appearing by magic.
    names = LABELS.get(domain, {})
    if names and int(names.get("K", -1)) == K:
        a["class_label"] = [names.get(str(v), f"class{v}") for v in lab]
    for k in range(K):
        a[f"p_class{k}"] = P[:, k]
    a["in_shared_cohort"] = a.MRN.isin(shared)
    a.to_csv(out / f"{spec['prefix']}_final_assignments.csv", index=False)
    assignments[domain] = a

    P_shared = P[a.in_shared_cohort.to_numpy()]
    ahi = validate.distal_bch(P, aux.index_ahi.to_numpy())
    kw = validate.kw_effect(aux.index_ahi.to_numpy(), lab)
    sev = validate.severity_mix(aux.index_ahi.to_numpy(), lab)

    prior_file = PRIOR / domain / f"{domain}_final_assignments.csv"
    agree = (validate.agreement_with_prior(lab, mat.MRN, prior_file)
             if prior_file.exists() else {"note": "prior partition not found"})

    final[domain] = {
        "K": K,
        "labels": {int(k): names.get(str(k)) for k in range(K)} if names else None,
        # final label -> the class index used by the fitting stage. The profile and
        # validation CSVs written by 01/02/03 are keyed to the fitting index, which
        # is NOT the same as the size-ordered final label; anything joining the two
        # must go through this map or it will silently mislabel every class.
        "final_to_fit_index": {int(i): int(o) for i, o in enumerate(order)},
        "n_maximal": int(len(a)),
        "n_shared_scored": int(a.in_shared_cohort.sum()),
        "class_sizes": {int(k): int((lab == k).sum()) for k in range(K)},
        "class_fractions": {int(k): round(float((lab == k).mean()), 4) for k in range(K)},
        "mean_max_posterior": {int(k): round(float(P[lab == k].max(1).mean()), 4)
                               for k in range(K)},
        "relative_entropy": round(float(model.relative_entropy(X)), 4),
        "top_distinguishing_indicators": marks,
        "ahi_bch": ahi.to_dict("records"),
        "ahi_kruskal_wallis": kw,
        "ahi_severity_mix": sev.reset_index().to_dict("records"),
        "shared_cohort_class_fractions": {
            int(k): round(float((P_shared.argmax(1) == k).mean()), 4) for k in range(K)},
        "agreement_with_prior_kmeans": agree,
    }
    print(f"[{domain}] K={K}  sizes={final[domain]['class_sizes']}  "
          f"relE={final[domain]['relative_entropy']:.3f}  "
          f"AHI(BCH)={[r['bch_mean'] for r in final[domain]['ahi_bch']]}")

# ------------------------------------------------ cross-domain on the shared cohort
cross = {}
if len(assignments) >= 2:
    merged = None
    for domain, a in assignments.items():
        col = "class_label" if "class_label" in a.columns else "class"
        s = a.loc[a.in_shared_cohort, ["MRN", col]].rename(columns={col: domain})
        merged = s if merged is None else merged.merge(s, on="MRN", how="inner")
    merged.to_csv(RESULTS / "shared_cohort_class_assignments.csv", index=False)
    cross["n_shared"] = int(len(merged))
    for a_dom, b_dom in combinations([d for d in SPEC if d in assignments], 2):
        v, tab = validate.cramers_v(merged[a_dom], merged[b_dom])
        cross[f"{a_dom}__x__{b_dom}"] = {
            "cramers_v": round(v, 4),
            "table": tab.to_dict(),
        }
        tab.to_csv(RESULTS / f"cross_{a_dom}_x_{b_dom}.csv")
    if len(assignments) == 3:
        doms = list(assignments)
        three = merged.groupby(doms).size().rename("n").reset_index()
        three["pct"] = (three.n / three.n.sum()).round(4)
        three.to_csv(RESULTS / "cross_three_way.csv", index=False)
    print(f"\n[cross-domain] shared n={cross['n_shared']:,}")
    for k, v in cross.items():
        if isinstance(v, dict) and "cramers_v" in v:
            print(f"  {k}: Cramer's V = {v['cramers_v']:.4f}")

(RESULTS / "FINAL_lca.json").write_text(json.dumps(
    {"choices": CHOICES, "domains": final, "cross_domain": cross},
    indent=2, default=str))
print(f"\nwrote {RESULTS / 'FINAL_lca.json'}")
