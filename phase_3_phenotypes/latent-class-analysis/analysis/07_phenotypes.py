"""Stage 7 -- cross-domain phenotypes built from the three class solutions.

The linkage design is dictated by a result, not a preference: on the shared cohort the
three class variables are almost exactly independent (Cramer's V 0.031 / 0.032 / 0.011;
all 27 joint cells populated; observed/expected between 0.82 and 1.27). Two things
follow.

1. **The joint phenotype is essentially the Cartesian product.** Because the axes carry
   non-redundant information, crossing them is exactly the right operation and no
   combination is empty -- the smallest of the 27 cells still holds 609 patients. A
   composite label is therefore usable directly.
2. **A second-order latent class model over the three labels is pointless here.** Its
   null *is* independence, and the data are already independent to within +/-27% per
   cell. It is fitted anyway, once, as a documented negative control rather than as a
   method.

Because every domain turned out to be a *gradient* rather than a set of discrete
subpopulations (see the embedding diagnostic in stage 6), this script emits both
representations and recommends the continuous one for downstream association testing:

  composite_label    the 27-cell discrete product, for description and stratification
  score_c1/c2/c3     posterior-weighted expected class rank in [0, K-1] -- a
                     3-dimensional continuous phenotype that keeps the uncertainty the
                     modal label throws away
  archetype          a small set of interpretable corner phenotypes
  joint_confidence   product of the three maximum posteriors

Outputs results/phenotypes/ (per-patient table + summary tables).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import validate  # noqa: E402
from common.config import RESULTS, WORK  # noqa: E402

OUT = RESULTS / "phenotypes"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = json.loads((RESULTS / "CLASS_LABELS.json").read_text())

# Each domain's classes ordered along the gradient the domain actually measures, so
# rank 0 is always the "least" end. This ordering is a clinical reading of the class
# profiles, not something the model provides -- a latent class variable is nominal.
GRADIENT = {
    "c1_physiologic": ["mild-hypoxemia", "moderate-hypoxemia", "severe-hypoxemia"],
    "c2_symptomatic": ["minimally-symptomatic", "moderate-symptom", "high-symptom"],
    "c3_comorbidities": ["low-burden", "moderate-burden", "high-burden"],
}
PREFIX = {"c1_physiologic": "c1", "c2_symptomatic": "c2", "c3_comorbidities": "c3"}


def domain_table(domain: str) -> pd.DataFrame:
    """Per-patient modal label, gradient score and assignment confidence."""
    pre = PREFIX[domain]
    a = pd.read_csv(RESULTS / domain / f"{pre}_final_assignments.csv", dtype={"MRN": str})
    K = LABELS[domain]["K"]
    names = [LABELS[domain][str(k)] for k in range(K)]
    missing = set(names) - set(GRADIENT[domain])
    if missing:
        raise ValueError(f"{domain}: {missing} not in the declared gradient ordering")
    rank = np.array([GRADIENT[domain].index(n) for n in names], float)
    P = a[[f"p_class{k}" for k in range(K)]].to_numpy()
    return pd.DataFrame({
        "MRN": a.MRN,
        domain: a.class_label,
        f"score_{pre}": P @ rank,          # posterior-weighted expected rank
        f"conf_{pre}": P.max(1),
    })


tabs = [domain_table(d) for d in GRADIENT]
ph = tabs[0]
for t in tabs[1:]:
    ph = ph.merge(t, on="MRN", how="inner")
print(f"patients with all three domains: {len(ph):,}")

ph["composite_label"] = (ph.c1_physiologic + " | " + ph.c2_symptomatic
                         + " | " + ph.c3_comorbidities)
ph["joint_confidence"] = ph.conf_c1 * ph.conf_c2 * ph.conf_c3
ph["total_score"] = ph.score_c1 + ph.score_c2 + ph.score_c3

# --- archetypes -----------------------------------------------------------------
# Corner phenotypes: patients at an extreme of all three axes at once. These are the
# combinations a clinician would actually name, and they exist here only because the
# axes are independent -- under a single severity factor the off-diagonal corners
# would be empty.
def archetype(r):
    hi = {"c1": r.score_c1 >= 1.5, "c2": r.score_c2 >= 1.5, "c3": r.score_c3 >= 1.5}
    lo = {"c1": r.score_c1 <= 0.5, "c2": r.score_c2 <= 0.5, "c3": r.score_c3 <= 0.5}
    if hi["c1"] and hi["c2"] and hi["c3"]:
        return "triple-high"
    if lo["c1"] and lo["c2"] and lo["c3"]:
        return "triple-low"
    if hi["c1"] and lo["c2"]:
        return "severe-but-asymptomatic"      # the classic occult-OSA phenotype
    if lo["c1"] and hi["c2"]:
        return "symptomatic-without-hypoxemia"
    if hi["c3"] and lo["c1"]:
        return "comorbid-without-hypoxemia"
    if hi["c1"] and hi["c2"] and lo["c3"]:
        return "severe-symptomatic-otherwise-well"
    return "unclassified"


ph["archetype"] = ph.apply(archetype, axis=1)

# --- octants: every patient classified -------------------------------------------
# The corner archetypes above name only the extremes and leave 75.4% unclassified.
# Cutting each axis at its median instead gives eight boxes that cover everyone, and
# because the axes are near-independent those boxes come out near-equal (8,349-9,249 of 70,880)
# rather than the 536-9,596 spread that collapsing the latent classes produces. The
# cost is contrast -- hi-vs-lo separation drops from 1.91 to 1.07 on the 0-2 scale --
# but sqrt(arm) x contrast is unchanged, and the reference group stops being a
# 19,149-patient grab-bag. Cut points are cohort-derived: record, do not transfer.
OCT_NAMES = {(0, 0, 0): "mild-all", (1, 0, 0): "hypoxemia-predominant",
             (0, 1, 0): "symptom-predominant", (0, 0, 1): "comorbidity-predominant",
             (1, 1, 0): "hypoxemic-symptomatic", (1, 0, 1): "hypoxemic-comorbid",
             (0, 1, 1): "symptomatic-comorbid", (1, 1, 1): "high-all"}
CUTS = {s: float(ph[s].median()) for s in ("score_c1", "score_c2", "score_c3")}
bits = [(ph[s] > CUTS[s]).astype(int) for s in ("score_c1", "score_c2", "score_c3")]
ph["octant"] = [OCT_NAMES[(a, b, c)] for a, b, c in zip(*[x.to_numpy() for x in bits])]

aux = pd.read_parquet(WORK / "c1_aux.parquet")[["MRN", "index_ahi", "STUDY_DEVICE"]]
util = pd.read_parquet(WORK / "c3_aux.parquet")[["MRN", "n_phecodes_all"]]
ph = ph.merge(aux, on="MRN", how="left").merge(util, on="MRN", how="left")
ph.to_csv(OUT / "cross_domain_phenotypes.csv", index=False)
oct_tab = (ph.groupby("octant").agg(n=("MRN", "size"), median_ahi=("index_ahi", "median"),
                                    median_codes=("n_phecodes_all", "median"))
           .assign(pct=lambda x: (x.n / len(ph)).round(4)).sort_values("n", ascending=False))
oct_tab.to_csv(OUT / "octants.csv")
(OUT / "octant_cuts.json").write_text(json.dumps(CUTS, indent=2))
print("\noctants (median cut per axis: "
      + ", ".join(f"{k}={v:.4f}" for k, v in CUTS.items()) + "):")
print(oct_tab.to_string())
print(f"wrote {OUT / 'cross_domain_phenotypes.csv'}  {ph.shape}")

# --- 27-cell composite ----------------------------------------------------------
tot = len(ph)
marg = [ph[d].value_counts(normalize=True) for d in GRADIENT]
cells = (ph.groupby(list(GRADIENT) + ["composite_label"]).size()
         .rename("n").reset_index())
cells["expected"] = [tot * marg[0][a] * marg[1][b] * marg[2][c]
                     for a, b, c in zip(cells.iloc[:, 0], cells.iloc[:, 1], cells.iloc[:, 2])]
cells["obs_over_exp"] = (cells.n / cells.expected).round(3)
cells["pct"] = (cells.n / tot).round(4)
cells["median_ahi"] = [float(ph.loc[ph.composite_label == c, "index_ahi"].median())
                       for c in cells.composite_label]
cells["median_n_phecodes"] = [float(ph.loc[ph.composite_label == c, "n_phecodes_all"].median())
                              for c in cells.composite_label]
cells = cells.sort_values("n", ascending=False)
cells.to_csv(OUT / "composite_27_cells.csv", index=False)
chi2 = float((((cells.n - cells.expected) ** 2) / cells.expected).sum())
print(f"27-cell composite: all {len(cells)} populated, min {cells.n.min():,}, "
      f"chi2 vs independence {chi2:.0f} on {27-1-6} df, "
      f"obs/exp in [{cells.obs_over_exp.min()}, {cells.obs_over_exp.max()}]")

# --- archetype summary ------------------------------------------------------------
arch = (ph.groupby("archetype")
        .agg(n=("MRN", "size"), median_ahi=("index_ahi", "median"),
             mean_ahi=("index_ahi", "mean"), median_codes=("n_phecodes_all", "median"),
             mean_conf=("joint_confidence", "mean"))
        .round(3).sort_values("n", ascending=False))
arch["pct"] = (arch.n / tot).round(4)
arch.to_csv(OUT / "archetypes.csv")
print("\narchetypes:")
print(arch.to_string())

# --- how much does the linkage buy? ------------------------------------------------
def eta2(y, g):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y, g = y[ok], np.asarray(g)[ok]
    gm = y.mean()
    ssb = sum(((y[g == u].mean() - gm) ** 2) * (g == u).sum() for u in np.unique(g))
    return float(ssb / ((y - gm) ** 2).sum())


ahi = np.log1p(ph.index_ahi.to_numpy())
inc = {nm: round(eta2(ahi, g), 4) for nm, g in [
    ("c1_alone", ph.c1_physiologic), ("c2_alone", ph.c2_symptomatic),
    ("c3_alone", ph.c3_comorbidities), ("composite_27", ph.composite_label)]}
sp = ph[["score_c1", "score_c2", "score_c3"]].corr(method="spearman").round(3)
summary = {
    "n_shared": int(tot),
    "n_cells_populated": int(len(cells)),
    "min_cell_n": int(cells.n.min()),
    "chi2_vs_independence": round(chi2, 1),
    "obs_over_exp_range": [float(cells.obs_over_exp.min()), float(cells.obs_over_exp.max())],
    "score_spearman": sp.to_dict(),
    "eta2_log_ahi": inc,
    "joint_confidence": {
        "mean": round(float(ph.joint_confidence.mean()), 4),
        "pct_above_0.5": round(float((ph.joint_confidence > 0.5).mean()), 4),
        "pct_above_0.8": round(float((ph.joint_confidence > 0.8).mean()), 4)},
    "note": ("eta^2 on AHI is NOT the test of whether the linkage is useful -- AHI is "
             "c1's own construct, so c1 alone captures most of it and c2/c3 add little "
             "by construction. The composite exists for outcomes the three axes "
             "predict jointly (adherence, incident disease, utilisation); test it "
             "there. The evidence that the linkage is worth making is the "
             "independence itself: three orthogonal axes carry more information than "
             "any one of them."),
}
(OUT / "phenotype_summary.json").write_text(json.dumps(summary, indent=2))
print("\nvariance in log1p(AHI) explained:", inc)
print("score correlations (Spearman):")
print(sp.to_string())
print(f"\njoint confidence: mean {summary['joint_confidence']['mean']:.3f}, "
      f"{summary['joint_confidence']['pct_above_0.5']:.1%} above 0.5")

# --- negative control: second-order LCA over the three labels ----------------------
try:
    from stepmix.stepmix import StepMix
    codes = np.column_stack([
        pd.Categorical(ph[d], categories=GRADIENT[d]).codes for d in GRADIENT]).astype(float)
    rows = []
    for K in (1, 2, 3):
        m = StepMix(n_components=K, measurement="multinoulli", n_init=5, max_iter=500,
                    random_state=0, progress_bar=0,
                    measurement_params={"integer_codes": True}).fit(codes)
        rows.append({"K": K, "loglik": round(m.score(codes) * len(codes), 1),
                     "bic": round(m.bic(codes), 1),
                     "rel_entropy": round(m.relative_entropy(codes), 4) if K > 1 else 1.0})
    so = pd.DataFrame(rows)
    so.to_csv(OUT / "second_order_lca_negative_control.csv", index=False)
    print("\nsecond-order LCA over the three class labels (negative control):")
    print(so.to_string(index=False))
except Exception as e:                                    # pragma: no cover
    print(f"\nsecond-order LCA control skipped: {type(e).__name__}: {e}")

print(f"\nwrote {OUT}")
