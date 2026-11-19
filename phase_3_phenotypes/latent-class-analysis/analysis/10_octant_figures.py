"""Stage 10 -- the octant phenotypes, replacing figs 10-11 for the corner archetypes.

The corner archetypes named only the extremes and left 75.4% of the cohort unlabelled,
which is why every archetype contrast in the incidence models came back null. The octants
cut each of the three gradient scores at its own cohort median, so every patient lands in
exactly one of eight cells and no one is discarded.

fig12  CONSTRUCTION. What the cut actually does: the three pairwise score planes with the
       median cross-hairs drawn on them, plus the resulting arm sizes. The point to read
       is that the four quadrants of every plane are near-equally occupied -- the scores
       are mutually uncorrelated, so a median split on three axes gives eight comparably
       sized arms rather than a diagonal ridge with two fat corners.
fig13  SIGNATURE. octant x clinical variable, the octant analogue of fig11. Colour carries
       the standardised contrast; every cell is annotated in its own units because the
       clinical question is "what is this group's Epworth", not "how many SD".

Colour note. Eight overlapping hues fail the all-pairs CVD check (worst pair dE 3.2), so
identity is NOT carried by hue here: the octants are axis categories, and which axes are
high is shown by a three-cell bit glyph. Colour does magnitude only.
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
PH = RESULTS / "phenotypes"
ph = pd.read_csv(PH / "cross_domain_phenotypes.csv", dtype={"MRN": str})
CUTS = json.loads((PH / "octant_cuts.json").read_text())

AXES = [("score_c1", "physiologic severity"), ("score_c2", "symptom burden"),
        ("score_c3", "comorbidity burden")]
# bits are (c1, c2, c3); ordered by how many axes are high, then by axis, so the table
# reads as a lattice rather than as an arbitrary list
BITS = {"mild-all": (0, 0, 0), "hypoxemia-predominant": (1, 0, 0),
        "symptom-predominant": (0, 1, 0), "comorbidity-predominant": (0, 0, 1),
        "hypoxemic-symptomatic": (1, 1, 0), "hypoxemic-comorbid": (1, 0, 1),
        "symptomatic-comorbid": (0, 1, 1), "high-all": (1, 1, 1)}
ORDER = list(BITS)
ph["octant"] = pd.Categorical(ph.octant, categories=ORDER, ordered=True)
N_BY = ph.octant.value_counts().reindex(ORDER)


def save(fig, name, caption):
    fig.text(0.005, -0.012, caption, fontsize=7, color=viz.INK_MUTED, ha="left",
             va="top", wrap=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote plots/{name}.pdf/.png")


def bit_glyph(bits):
    """Filled/open squares for (c1, c2, c3): filled = above that axis's median.

    Set in the tick label itself rather than drawn as patches -- it cannot collide with
    the labels, and it survives being copied out of the figure as text.
    """
    return "".join("■" if b else "□" for b in bits)


# ==================================================== fig12 construction & occupancy
fig = plt.figure(figsize=(14.4, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.26)
rng = np.random.default_rng(0)
pairs = [(0, 1), (0, 2), (1, 2)]
for k, (i, j) in enumerate(pairs):
    ax = fig.add_subplot(gs[0, k])
    a, la = AXES[i]
    b, lb = AXES[j]
    jt = rng.normal(0, 0.035, (len(ph), 2))
    hb = ax.hexbin(ph[a] + jt[:, 0], ph[b] + jt[:, 1], gridsize=58, cmap=viz.SEQ_GREY,
                   bins="log", mincnt=1, linewidths=0)
    ax.axvline(CUTS[a], color=viz.CLASS_COLORS[1], lw=1.6, zorder=4)
    ax.axhline(CUTS[b], color=viz.CLASS_COLORS[1], lw=1.6, zorder=4)
    hi_a, hi_b = ph[a] > CUTS[a], ph[b] > CUTS[b]
    for xa, xb, xt, yt in ((False, False, 0.06, 0.06), (True, False, 0.94, 0.06),
                           (False, True, 0.06, 0.94), (True, True, 0.94, 0.94)):
        pct = ((hi_a == xa) & (hi_b == xb)).mean()
        ax.text(xt, yt, f"{pct:.0%}", transform=ax.transAxes, fontsize=9,
                fontweight="bold", color=viz.INK_2,
                ha="right" if xt > 0.5 else "left", va="top" if yt > 0.5 else "bottom")
    r = ph[[a, b]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel(la)
    ax.set_ylabel(lb)
    ax.set_title(f"Spearman r = {r:+.3f}", fontsize=9)
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(-0.25, 2.25)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

# ---- occupancy bars with bit glyphs
ax = fig.add_subplot(gs[1, :])
n_hi = np.array([sum(BITS[o]) for o in ORDER])
# colour = how many axes are high (magnitude), NOT identity -- one hue, light to dark
cols = [viz.SEQ_BLUE(0.18 + 0.26 * h) for h in n_hi]
y = np.arange(len(ORDER))
ax.barh(y, N_BY.to_numpy(), color=cols, height=0.62, zorder=3)
for k, o in enumerate(ORDER):
    n = int(N_BY[o])
    ax.text(n + 90, k, f"{n:,}  ({n/len(ph):.1%})", va="center", ha="left",
            fontsize=8, color=viz.INK_2)
ax.set_yticks(y)
ax.set_yticklabels([f"{bit_glyph(BITS[o])}  {o}" for o in ORDER], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("patients")
ax.set_xlim(0, N_BY.max() * 1.22)
ax.grid(axis="y", visible=False)
ax.set_title("Every patient is classified, and the eight arms are within 11% of "
             "each other in size", fontsize=10, pad=26)
ax.text(0.0, 1.10, "■□□ = above the cohort median on physiologic / symptom / "
        "comorbidity, in that order", transform=ax.transAxes, fontsize=7,
        color=viz.INK_MUTED, ha="left", va="bottom")

cb = fig.colorbar(hb, ax=fig.axes[:3], shrink=0.82, pad=0.012)
cb.set_label("patients per bin (log)", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
fig.suptitle("Octant phenotypes: a median split on each of the three independent axes",
             fontsize=12, fontweight="bold", x=0.005, ha="left")
save(fig, "fig12_octant_construction",
     "Top: the three score planes, with each axis cut at its own cohort median (orange "
     f"lines: score_c1 {CUTS['score_c1']:.3f}, score_c2 {CUTS['score_c2']:.3f}, score_c3 "
     f"{CUTS['score_c3']:.3f}). Because the three scores are mutually uncorrelated "
     "(|r| <= 0.04) the four quadrants of every plane hold close to 25% each, which is "
     "what makes the eight-cell cross worth taking. Bottom: the resulting arms, "
     f"{N_BY.min():,}-{N_BY.max():,} patients each, covering all {len(ph):,} patients in "
     "the shared cohort. The corner archetypes this replaces named only the extremes and "
     "left 75.4% unclassified, and no archetype contrast reached significance in the "
     "incidence models. The cut points are cohort-derived medians, not clinical "
     "thresholds, and do not transfer to another sample.")

# ============================================================== fig13 clinical signature
c1 = pd.read_csv(DATA / "c1_physiologic" / "c1_features_maximal.csv", dtype={"MRN": str})
c1 = c1[~c1.STUDY_DEVICE.isin(prep.C1_EXCLUDE_DEVICES)]
c2 = pd.read_parquet(WORK / "c2_model_scales.parquet")
c3 = pd.read_parquet(WORK / "c3_model.parquet")
c3a = pd.read_parquet(WORK / "c3_aux.parquet")[["MRN", "n_phecodes_all"]]

COND = ["278", "272", "401", "250", "300", "530"]
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

d = ph[["MRN", "octant"]].merge(
    c1[["MRN", "index_ahi", "ODI4PCT", "MINIMUM_SPO2", "MEAN_SPO2", "T90%"]],
    on="MRN", how="left")
d = d.merge(c2[["MRN", "ESS_total", "ISI_total", "FOSQ_impair", "PHQ2_total",
                "STOP_total"]], on="MRN", how="left")
d = d.merge(c3a, on="MRN", how="left").merge(c3[["MRN"] + COND], on="MRN", how="left")

rows, labels, blocks, decs = [], [], [], []
for blk, items in BLOCKS:
    for col, lbl, dec in items:
        rows.append(d.groupby("octant", observed=True)[col].mean().reindex(ORDER))
        labels.append(lbl)
        blocks.append(blk)
        decs.append(dec)
M = pd.DataFrame(rows, index=labels)
cohort = pd.Series([d[c].mean() for _, its in BLOCKS for c, _, _ in its], index=labels)
sd = pd.Series([d[c].std() for _, its in BLOCKS for c, _, _ in its], index=labels)
Z = M.sub(cohort, axis=0).div(sd.replace(0, np.nan), axis=0)

fig, ax = plt.subplots(figsize=(11.6, 0.42 * len(M) + 3.4))
lim = float(np.nanmax(np.abs(Z.to_numpy())))
im = ax.imshow(Z.to_numpy(), cmap=viz.DIV, aspect="auto", vmin=-lim, vmax=lim)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v, z = M.iloc[i, j], Z.iloc[i, j]
        ax.text(j, i, f"{v:,.{decs[i]}f}", ha="center", va="center", fontsize=7,
                color=viz.SURFACE if abs(z) > 0.62 * lim else viz.INK_2)
ax.set_xticks(range(M.shape[1]))
ax.set_xticklabels([f"{o.replace('-', '-\n')}\n{bit_glyph(BITS[o])}\nn={int(N_BY[o]):,}"
                    for o in M.columns], fontsize=7.2)
ax.set_yticks(range(len(M)))
ax.set_yticklabels(labels, fontsize=7.5)
for b in ("symptom", "comorbidity", "external"):
    ax.axhline(blocks.index(b) - 0.5, color=viz.INK_MUTED, lw=0.9)
for b in ("physiologic", "symptom", "comorbidity", "external"):
    idx = [i for i, x in enumerate(blocks) if x == b]
    ax.annotate(b.upper(), xy=(0, np.mean(idx)), xycoords=("axes fraction", "data"),
                xytext=(-118, 0), textcoords="offset points", fontsize=6.5,
                color=viz.INK_MUTED, ha="center", va="center", rotation=90)
ax.text(0.0, 1.032, "■□□ = above the cohort median on physiologic / symptom / "
        "comorbidity, in that order", transform=ax.transAxes, fontsize=7,
        color=viz.INK_MUTED, ha="left", va="bottom")
cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.02)
cb.set_label("SD from cohort mean", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
ax.set_title("Clinical signature of each octant phenotype", pad=26)
ax.grid(False)


def val(row, col):
    return M.loc[row, col]


save(fig, "fig13_octant_signature",
     "Cells are annotated in each variable's own units; colour is the standardised "
     "contrast against the whole shared cohort, and the three squares under each column "
     "say which axes that octant is above the median on. The two single-axis mismatch "
     "groups are the ones the cross was taken to find: hypoxemia-predominant carries ODI "
     f"{val('ODI 4% (events/h)','hypoxemia-predominant'):.1f} and minimum SpO2 "
     f"{val('minimum SpO2 (%)','hypoxemia-predominant'):.0f}% on an Epworth of "
     f"{val('Epworth total','hypoxemia-predominant'):.1f}, while symptom-predominant is "
     f"its mirror (ODI {val('ODI 4% (events/h)','symptom-predominant'):.1f}, Epworth "
     f"{val('Epworth total','symptom-predominant'):.1f}, ISI "
     f"{val('Insomnia Severity Index','symptom-predominant'):.1f}). AHI is external and "
     "was never modelled, yet it tracks the physiologic axis and not the other two "
     f"(mild-all {val('index AHI (events/h)','mild-all'):.0f} vs hypoxemia-predominant "
     f"{val('index AHI (events/h)','hypoxemia-predominant'):.0f} vs symptom-predominant "
     f"{val('index AHI (events/h)','symptom-predominant'):.0f} events/h). Contrasts are "
     "milder than the archetype figure because a median split is a milder cut than a "
     "corner rule -- that is the trade the octants make in exchange for classifying "
     f"all {len(ph):,} patients instead of 24.6% of them.")
M.round(3).to_csv(PH / "octant_clinical_signature.csv")
print(f"    wrote phenotypes/octant_clinical_signature.csv")
print("\ndone.")
