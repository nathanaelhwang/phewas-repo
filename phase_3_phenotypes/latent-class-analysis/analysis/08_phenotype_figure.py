"""Stage 8 -- the clinical read of the cross-domain phenotypes.

Figures 1-9 are per-domain and diagnostic: they answer "is this class solution sound?".
This stage answers the different question a clinician asks -- "what do these phenotypes
look like, and are they distinguishable at the bedside?" -- so it works on the stage-7
archetypes rather than on any single domain's classes.

fig10  the phenotype SPACE: pairwise density of the three continuous scores. The
       corners being filled rather than a diagonal ridge is the whole argument for
       crossing the domains, so it is shown before any archetype summary.
fig11  the phenotype SIGNATURE: archetype x clinical variable. This is the figure to
       read when the question is clinical. Colour carries the standardised contrast
       (which phenotype is high on what) and every cell is annotated with the value in
       its own units, because a clinician needs "ESS 13.8", not "+1.2 SD".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import prep, viz  # noqa: E402
from common.config import DATA, RESULTS, WORK  # noqa: E402

viz.style()
PLOTS = RESULTS / "plots"
ph = pd.read_csv(RESULTS / "phenotypes" / "cross_domain_phenotypes.csv", dtype={"MRN": str})

# Order archetypes along a clinical narrative rather than by size: the two "mismatch"
# phenotypes first, because they are the ones the axes were crossed to find.
ORDER = ["triple-low", "symptomatic-without-hypoxemia", "comorbid-without-hypoxemia",
         "severe-but-asymptomatic", "severe-symptomatic-otherwise-well", "triple-high"]
ph = ph[ph.archetype.isin(ORDER)].copy()
ph["archetype"] = pd.Categorical(ph.archetype, categories=ORDER, ordered=True)


def save(fig, name, caption):
    fig.text(0.005, -0.012, caption, fontsize=7, color=viz.INK_MUTED, ha="left",
             va="top", wrap=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote plots/{name}.pdf/.png")


# ============================================================ fig10 phenotype space
full = pd.read_csv(RESULTS / "phenotypes" / "cross_domain_phenotypes.csv",
                   dtype={"MRN": str})
pairs = [("score_c1", "score_c2", "physiologic severity", "symptom burden"),
         ("score_c1", "score_c3", "physiologic severity", "comorbidity burden"),
         ("score_c2", "score_c3", "symptom burden", "comorbidity burden")]
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7))
rng = np.random.default_rng(0)
for ax, (a, b, la, lb) in zip(axes, pairs):
    j = rng.normal(0, 0.035, (len(full), 2))          # jitter: scores pile on 0/1/2
    hb = ax.hexbin(full[a] + j[:, 0], full[b] + j[:, 1], gridsize=60,
                   cmap=viz.SEQ_GREY, bins="log", mincnt=1, linewidths=0)
    r = full[[a, b]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel(la)
    ax.set_ylabel(lb)
    ax.set_title(f"Spearman r = {r:+.3f}")
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(-0.25, 2.25)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    for x in (0, 2):
        for y in (0, 2):
            ax.plot(x, y, marker="s", ms=13, mfc="none", mec=viz.CLASS_COLORS[1],
                    mew=1.8, zorder=5)
cb = fig.colorbar(hb, ax=axes, shrink=0.75, pad=0.015)
cb.set_label("patients per bin (log)", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
fig.suptitle("The phenotype space is filled, not a single severity ridge",
             fontsize=12, fontweight="bold", x=0.005, ha="left")
save(fig, "fig10_phenotype_space",
     "Each axis is a posterior-weighted expected class rank (0 = least, 2 = most). If OSA "
     "phenotype were one underlying severity factor these panels would show a diagonal "
     "ridge and the off-diagonal corners (orange squares) would be empty. They are not: "
     "all four corners of every pair are occupied and the three scores are mutually "
     "uncorrelated (|r| <= 0.04), which is what makes crossing the domains worth doing. "
     f"n = {len(full):,} patients with all three domains.")

# ======================================================== fig11 clinical signature
# Raw, clinically-readable values per archetype, pulled from each domain's own units.
c1 = pd.read_csv(DATA / "c1_physiologic" / "c1_features_maximal.csv", dtype={"MRN": str})
c1 = c1[~c1.STUDY_DEVICE.isin(prep.C1_EXCLUDE_DEVICES)]
c2 = pd.read_parquet(WORK / "c2_model_scales.parquet")
c3 = pd.read_parquet(WORK / "c3_model.parquet")
c3a = pd.read_parquet(WORK / "c3_aux.parquet")[["MRN", "n_phecodes_all"]]
meta3 = pd.read_csv(RESULTS / "c3_indicator_selection.csv", dtype={"root": str})
lab3 = dict(zip(meta3.root, meta3.label))

COND = ["278", "272", "401", "250", "300", "530"]      # obesity, lipids, HTN, IFG, anxiety, GERD
BLOCKS = [
    ("physiologic", [("ODI4PCT", "ODI 4% (events/h)", 1),
                     ("MINIMUM_SPO2", "minimum SpO2 (%)", 1),
                     ("MEAN_SPO2", "mean SpO2 (%)", 1),
                     ("T90%", "T90 (fraction)", 3)]),
    ("symptom", [("ESS_total", "Epworth total", 1),
                 ("ISI_total", "Insomnia Severity Index", 1),
                 ("FOSQ_impair", "FOSQ impairment", 1),
                 ("PHQ2_total", "PHQ-2", 2),
                 ("STOP_total", "STOP score", 2)]),
    ("comorbidity", [("n_phecodes_all", "distinct phecodes", 0),
                     ("278", "obesity", 2), ("272", "hyperlipidemia", 2),
                     ("401", "hypertension", 2), ("250", "impaired fasting glucose", 2),
                     ("300", "anxiety disorders", 2), ("530", "GERD", 2)]),
    ("external", [("index_ahi", "index AHI (events/h)", 1)]),
]

d = ph[["MRN", "archetype"]].merge(c1[["MRN", "index_ahi"] + prep.C1_PHYSIO_11[:0]
                                      + ["ODI4PCT", "MINIMUM_SPO2", "MEAN_SPO2", "T90%"]],
                                   on="MRN", how="left")
d = d.merge(c2[["MRN", "ESS_total", "ISI_total", "FOSQ_impair", "PHQ2_total", "STOP_total"]],
            on="MRN", how="left")
d = d.merge(c3a, on="MRN", how="left").merge(c3[["MRN"] + COND], on="MRN", how="left")

rows, labels, blocks, decs = [], [], [], []
for blk, items in BLOCKS:
    for col, lbl, dec in items:
        rows.append(d.groupby("archetype", observed=True)[col].mean())
        labels.append(lbl)
        blocks.append(blk)
        decs.append(dec)
M = pd.DataFrame(rows, index=labels)
cohort = pd.Series([d[c].mean() for _, its in BLOCKS for c, _, _ in its], index=labels)
sd = pd.Series([d[c].std() for _, its in BLOCKS for c, _, _ in its], index=labels)
Z = M.sub(cohort, axis=0).div(sd.replace(0, np.nan), axis=0)

n_by = ph.archetype.value_counts().reindex(ORDER)
fig, ax = plt.subplots(figsize=(9.4, 0.42 * len(M) + 3.0))
lim = float(np.nanmax(np.abs(Z.to_numpy())))
im = ax.imshow(Z.to_numpy(), cmap=viz.DIV, aspect="auto", vmin=-lim, vmax=lim)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v, z = M.iloc[i, j], Z.iloc[i, j]
        ax.text(j, i, f"{v:,.{decs[i]}f}", ha="center", va="center", fontsize=7,
                color=viz.SURFACE if abs(z) > 0.62 * lim else viz.INK_2)
ax.set_xticks(range(M.shape[1]))
ax.set_xticklabels([f"{a.replace('-', '-\n')}\nn={n_by[a]:,}" for a in M.columns],
                   fontsize=7.5)
ax.set_yticks(range(len(M)))
ax.set_yticklabels(labels, fontsize=7.5)
for b in ("symptom", "comorbidity", "external"):        # hairline block separators
    ax.axhline(blocks.index(b) - 0.5, color=viz.INK_MUTED, lw=0.9)
# domain tags sit clear of the tick labels, centred on their block
for b in ("physiologic", "symptom", "comorbidity", "external"):
    idx = [i for i, x in enumerate(blocks) if x == b]
    ax.annotate(b.upper(), xy=(0, np.mean(idx)), xycoords=("axes fraction", "data"),
                xytext=(-118, 0), textcoords="offset points", fontsize=6.5,
                color=viz.INK_MUTED, ha="center", va="center", rotation=90)
cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.02)
cb.set_label("SD from cohort mean", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
ax.set_title("Clinical signature of each cross-domain phenotype", pad=10)
ax.grid(False)
def val(row, col):
    return M.loc[row, col]


n_arch, n_shared = int(len(ph)), int(len(full))
save(fig, "fig11_clinical_phenotypes",
     "Cells are annotated in each variable's own units; colour is the standardised "
     "contrast against the whole shared cohort. Read the two mismatch phenotypes first — "
     "they exist only because the three axes are independent. severe-but-asymptomatic "
     f"carries ODI {val('ODI 4% (events/h)','severe-but-asymptomatic'):.1f} and minimum "
     f"SpO2 {val('minimum SpO2 (%)','severe-but-asymptomatic'):.0f}% on an Epworth of "
     f"{val('Epworth total','severe-but-asymptomatic'):.1f} and an ISI of "
     f"{val('Insomnia Severity Index','severe-but-asymptomatic'):.1f}; "
     "symptomatic-without-hypoxemia is its mirror image (ODI "
     f"{val('ODI 4% (events/h)','symptomatic-without-hypoxemia'):.1f}, Epworth "
     f"{val('Epworth total','symptomatic-without-hypoxemia'):.1f}, ISI "
     f"{val('Insomnia Severity Index','symptomatic-without-hypoxemia'):.1f}). Note also "
     "severe-symptomatic-otherwise-well: the highest AHI in the table "
     f"({val('index AHI (events/h)','severe-symptomatic-otherwise-well'):.0f}) with the "
     f"fewest diagnoses ({val('distinct phecodes','severe-symptomatic-otherwise-well'):.0f}). "
     "AHI is external and was never modelled. Archetypes cover "
     f"{n_arch:,} of {n_shared:,} shared-cohort patients ({n_arch/n_shared:.1%}); the rest "
     "sit between corners and are 'unclassified' by design.")
print("\ndone.")
