"""Stage 6 -- figures for the finalised classes.

Each domain is shown in the geometry its own model defines, because that is the
view whose axes mean something:

  c1  canonical discriminant projection + the model's own 95% covariance ellipses
  c2  ternary plot of the 3-class posterior simplex (classification uncertainty)
  c3  multiple correspondence analysis (the categorical analogue of PCA)

plus a cross-domain alluvial, and nonlinear embeddings (UMAP / t-SNE) used
strictly as a continuum-versus-discrete diagnostic. Those last ones are captioned
to say so: UMAP manufactures visually crisp islands even from structureless data,
and its between-cluster distances, blob sizes and densities carry no meaning, so it
can never be evidence that a class solution is real.

Writes vector PDF + raster PNG to results/plots/.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import viz  # noqa: E402
from common.config import RESULTS, SEED, WORK  # noqa: E402

warnings.filterwarnings("ignore")
viz.style()
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
EMBED_N = 40_000          # subsample for the nonlinear embeddings
CHOICES = json.loads((RESULTS / "K_CHOICES.json").read_text())
LABELS = json.loads((RESULTS / "CLASS_LABELS.json").read_text())
FINAL = json.loads((RESULTS / "FINAL_lca.json").read_text())["domains"]


def fit_index(domain, k):
    """Fitting-stage class index for final (size-ordered) label ``k``.

    The profile and validation CSVs from stages 01-03 are keyed to the order the
    mixture happened to converge in; the assignments and labels are keyed to
    descending class size. For c2 those two orderings differ by a swap and for c3
    by a full permutation, so joining them without this map silently attaches every
    class's numbers to the wrong name.
    """
    return int(FINAL[domain]["final_to_fit_index"][str(k)])


def save(fig, name, caption=None):
    if caption:
        fig.text(0.005, -0.012, caption, fontsize=7, color=viz.INK_MUTED,
                 ha="left", va="top", wrap=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote plots/{name}.pdf/.png")


def names_for(domain, K, short=False):
    """Full class names, or the short forms used where a full label would collide.

    The full name always appears in the legend, so a short direct label never costs
    the reader the identity -- it only avoids two labels landing on top of each other.
    """
    d = LABELS.get(domain, {})
    if short and "short" in d:
        return [d["short"].get(str(k), d.get(str(k), f"class {k}")) for k in range(K)]
    return [d.get(str(k), f"class {k}") for k in range(K)]


def load(domain, prefix, matrix):
    a = pd.read_csv(RESULTS / domain / f"{prefix}_final_assignments.csv",
                    dtype={"MRN": str})
    mat = pd.read_parquet(WORK / matrix)
    aux = pd.read_parquet(WORK / f"{prefix}_aux.parquet")
    assert (a.MRN.to_numpy() == mat.MRN.to_numpy()).all()
    return a, mat, aux


# ===================================================================== c1
print("[c1] discriminant projection")
K1 = CHOICES["c1_physiologic"]
N1 = names_for("c1_physiologic", K1)
P1 = viz.class_palette(K1)
a1, m1, x1 = load("c1_physiologic", "c1", "c1_model.parquet")
F1 = [c for c in m1.columns if c != "MRN"]
X1 = m1[F1].to_numpy(float)
lab1 = a1["class"].to_numpy()

with open(WORK / "c1_models.pkl", "rb") as fh:
    blob = pickle.load(fh)
gm = blob["primary"][K1]

# Complete the matrix under the fitted mixture before projecting. Only 26.5% of rows
# are observed on all 10 features, and NONE of them are WatchPAT -- the supine ratio
# is missing for 100% of that device -- so a projection fitted on complete cases would
# silently show one device and call it the cohort.
X1c = gm.complete(X1)
Y1, lda = viz.discriminant_axes(X1c, lab1, 2)
ok = ~np.isnan(Y1).any(axis=1)
print(f"    projected {ok.sum():,} of {len(X1):,} patients "
      f"(complete-case would have been {(~np.isnan(X1).any(axis=1)).sum():,})")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
# (a) classes with the model's own fitted covariance ellipses, projected
order = [fit_index("c1_physiologic", k) for k in range(K1)]
A = lda.scalings_[:, :2]
ax = axes[0]
viz.density_panel(ax, Y1[ok, 0], Y1[ok, 1], cmap=viz.SEQ_GREY, gridsize=150)
for k in range(K1):
    src = order[k]
    mu = (gm.means_[src] - lda.xbar_) @ A
    cov = A.T @ gm.covariances_[src] @ A
    viz.cov_ellipse(ax, mu, cov, level=0.95, facecolor="none", edgecolor=P1[k],
                    lw=2.0, zorder=5)
    viz.label_at(ax, mu, N1[k].replace("-", "-\n"), P1[k])
ax.set_title("a  Canonical discriminant space")
ax.set_xlabel("discriminant 1")
ax.set_ylabel("discriminant 2")

# (b) device share per bin. Two overplotted point clouds at this density resolve to
# whichever was drawn last, so the question "where does each device sit?" is answered
# with a binned share instead: diverging, because the two devices are opposite poles.
ax = axes[1]
dev = x1.STUDY_DEVICE.to_numpy()
is_wp = (dev == "WatchPAT").astype(float)
hb = ax.hexbin(Y1[ok, 0], Y1[ok, 1], C=is_wp[ok], reduce_C_function=np.mean,
               gridsize=110, cmap=viz.DIV, vmin=0, vmax=1, mincnt=25, linewidths=0)
cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("share WatchPAT  (0 = all NOX T3)", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
ax.set_title("b  Same space, device mix per bin")
ax.set_xlabel("discriminant 1")
ax.set_ylabel("")

# (c) mean AHI per bin -- sequential, one hue, with a scale legend
ax = axes[2]
ahi = x1.index_ahi.to_numpy()
hb = ax.hexbin(Y1[ok, 0], Y1[ok, 1], C=np.clip(ahi[ok], 0, 80),
               reduce_C_function=np.mean, gridsize=110, cmap=viz.SEQ_BLUE,
               mincnt=25, linewidths=0)
cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("mean index AHI (events/h)", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
ax.set_title("c  Same space, mean AHI per bin (never modelled)")
ax.set_xlabel("discriminant 1")
ax.set_ylabel("")
for ax in axes:
    ax.set_xlim(np.nanpercentile(Y1[:, 0], [0.2, 99.8]))
    ax.set_ylim(np.nanpercentile(Y1[:, 1], [0.2, 99.8]))
fig.suptitle("c1_physiologic — 3 latent profiles in the model's own geometry",
             fontsize=12, fontweight="bold", x=0.005, ha="left")
fig.tight_layout()
save(fig, "fig1_c1_discriminant",
     "Ellipses are the fitted 95% covariance of each profile projected into the "
     "discriminant plane, not a summary of assigned points — the overlap shown is the "
     "model's own. For a K-class Gaussian mixture all between-class structure lies in "
     "K-1=2 dimensions, so this view loses nothing. Missing entries were completed "
     "under the fitted mixture: only 26.5% of rows are observed on all 10 features and "
     "none of those are WatchPAT (the supine ratio is missing for 100% of that device), "
     "so a complete-case projection would show one device and call it the cohort. "
     "Panel b: Cramér's V(class, device) = 0.227 — part of the apparent severity axis "
     "is sensor physics, not physiology.")

# ---- c1 profile plot
print("[c1] profile plot")
prof = pd.read_csv(RESULTS / "c1_physiologic" / "c1_class_profiles.csv")
prof = prof[prof.K == K1]
piv = prof.pivot(index="indicator", columns="class", values="std_diff")
piv = piv.loc[piv.abs().max(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ypos = np.arange(len(piv))
ax.axvline(0, color=viz.INK_MUTED, lw=0.9, zorder=1)
for k in range(K1):
    ax.plot(piv[order[k]].to_numpy(), ypos, "o-", color=P1[k], lw=1.6, ms=5.5,
            label=N1[k], zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels(piv.index, fontsize=8)
ax.set_xlabel("standardised difference from the cohort mean (SD units)")
ax.set_title("c1_physiologic — profile of each latent class")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.5)
save(fig, "fig2_c1_profiles",
     "Values are on the original clinical scale, standardised for comparability. "
     "Higher ODI4PCT / T90% and lower SpO2 define the severe profile.")

# ===================================================================== c2
print("[c2] ternary posterior simplex")
K2 = CHOICES["c2_symptomatic"]
N2 = names_for("c2_symptomatic", K2)
P2 = viz.class_palette(K2)
a2, m2, x2 = load("c2_symptomatic", "c2", "c2_model_scales.parquet")
post2 = a2[[f"p_class{k}" for k in range(K2)]].to_numpy()
lab2 = a2["class"].to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
ax = axes[0]
V = viz.ternary_axes(ax, [n.replace("-", "-\n") for n in N2])
T = viz.to_ternary(post2, V)
sub = rng.choice(len(T), min(30_000, len(T)), replace=False)
for k in range(K2):
    s = sub[lab2[sub] == k]
    ax.scatter(T[s, 0], T[s, 1], s=2.2, c=P2[k], alpha=0.25, linewidths=0,
               rasterized=True, zorder=2)
ax.set_title("a  Posterior simplex — every patient")

ax = axes[1]
ax.hist([a2.max_posterior[lab2 == k] for k in range(K2)], bins=40, stacked=True,
        color=P2, label=N2, lw=0)
ax.set_xlabel("maximum posterior probability")
ax.set_ylabel("patients")
ax.set_title("b  Classification certainty")
ax.legend(loc="upper left")
ax.grid(axis="x", alpha=0)

# A stacked bar over ~12 intake-form patterns would put more than seven colour
# classes on screen, past the point where adjacent classes blur. The five dominant
# patterns (92.6% of the cohort) go in a labelled heatmap instead, with the rest
# folded into "other".
ax = axes[2]
fp = x2.form_pattern.astype(str)
top5 = fp.value_counts().head(5).index.tolist()
fp_g = np.where(fp.isin(top5), fp, "other")
tab = pd.crosstab(pd.Series(lab2, name="c"), pd.Series(fp_g), normalize="index")
tab = tab[[c for c in top5 if c in tab.columns] + (["other"] if "other" in tab else [])]
im = ax.imshow(tab.to_numpy(), cmap=viz.SEQ_BLUE, aspect="auto", vmin=0, vmax=1)
for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
        v = tab.to_numpy()[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color=viz.SURFACE if v > 0.55 else viz.INK_2)
ax.set_xticks(range(tab.shape[1]))
ax.set_xticklabels([f"form {i+1}" if c != "other" else "other"
                    for i, c in enumerate(tab.columns)], fontsize=8)
ax.set_yticks(range(K2))
ax.set_yticklabels([n.replace("-", "-\n") for n in N2], fontsize=8)
ax.set_title("c  Intake-form version by class (artifact check)")
ax.grid(False)
fig.suptitle("c2_symptomatic — 3 symptom classes, and how confidently patients are assigned",
             fontsize=12, fontweight="bold", x=0.005, ha="left")
fig.tight_layout()
save(fig, "fig3_c2_simplex",
     "Panel a: each point is one patient's posterior; a point at a vertex is a certain "
     "assignment and one near the centre is ambiguous. Relative entropy is 0.792, which "
     "is the spread you see. Panel c: Cramér's V(class, intake-form pattern) = 0.294 — "
     "the classes are not purely questionnaire-version artifacts, but the association "
     "is not negligible either.")

print("[c2] profile plot")
prof2 = pd.read_csv(RESULTS / "c2_symptomatic" / "c2_class_profiles.csv")
prof2 = prof2[prof2.K == K2]
piv2 = prof2.pivot(index="indicator", columns="class", values="std_diff")
piv2 = piv2.loc[piv2.abs().max(axis=1).sort_values(ascending=False).head(26).index]
piv2 = piv2.loc[piv2.abs().max(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(7.8, 6.4))
ypos = np.arange(len(piv2))
ax.axvline(0, color=viz.INK_MUTED, lw=0.9, zorder=1)
for k in range(K2):
    ax.plot(piv2[fit_index("c2_symptomatic", k)].to_numpy(), ypos, "o-",
            color=P2[k], lw=1.6, ms=5.0,
            label=N2[k], zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels(piv2.index, fontsize=7.5)
ax.set_xlabel("standardised difference from the cohort mean (SD units)")
ax.set_title("c2_symptomatic — profile of each latent class (top 26 indicators)")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.5)
save(fig, "fig4_c2_profiles",
     "Most continuous indicators are oriented so that higher = worse (FOSQ is reverse "
     "scored to an impairment score). SLEEPTIME is the exception and runs the other way "
     "— the high-symptom class sleeps ~1.1 h less — which is the line that crosses. The "
     "three classes are a severity ladder, yet external AHI is flat across them "
     "(eps^2 = 0.004).")

# ===================================================================== c3
print("[c3] multiple correspondence analysis")
K3 = CHOICES["c3_comorbidities"]
N3 = names_for("c3_comorbidities", K3)
P3 = viz.class_palette(K3)
a3, m3, x3 = load("c3_comorbidities", "c3", "c3_model.parquet")
F3 = [c for c in m3.columns if c != "MRN"]
B = m3[F3].to_numpy(np.int8)
lab3 = a3["class"].to_numpy()
meta3 = pd.read_csv(RESULTS / "c3_indicator_selection.csv", dtype={"root": str})
lab_of = dict(zip(meta3.root, meta3.label))
cat_of = dict(zip(meta3.root, meta3.category))

t0 = time.perf_counter()
row, col, inertia, _ = viz.mca(B, n_components=3, seed=SEED)
print(f"    MCA on {B.shape} in {time.perf_counter()-t0:.0f}s; "
      f"axis inertia {np.round(inertia[:3], 3)}")

fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2))
ax = axes[0]
viz.density_panel(ax, row[:, 0], row[:, 1], cmap=viz.SEQ_GREY, gridsize=150)
for k in range(K3):
    c = row[lab3 == k].mean(axis=0)[:2]
    ax.scatter(c[0], c[1], s=170, c=P3[k], edgecolor=viz.SURFACE, linewidth=2.0,
               zorder=6)
    viz.label_at(ax, c, N3[k].replace("-", "-\n"), P3[k])
ax.set_title("a  Patients in MCA space, with class centroids")
ax.set_xlabel(f"MCA axis 1 ({inertia[0]:.1%} inertia)")
ax.set_ylabel(f"MCA axis 2 ({inertia[1]:.1%} inertia)")

# variable level: the "=1" (present) coordinates, which is where the meaning is
ax = axes[1]
J = len(F3)
present = col[J:]
prev = B.mean(axis=0)
cats = pd.Series([cat_of.get(f) for f in F3]).fillna("other")
top = np.argsort(-np.abs(present[:, 0]) - np.abs(present[:, 1]))[:26]
ax.scatter(present[:, 0], present[:, 1], s=10 + 220 * prev, c=viz.INK_MUTED,
           alpha=0.35, linewidths=0)
for i in top:
    ax.scatter(present[i, 0], present[i, 1], s=10 + 220 * prev[i],
               c=viz.CLASS_COLORS[0], alpha=0.85, linewidths=0, zorder=4)
txt = [ax.annotate(str(lab_of.get(F3[i], F3[i]))[:34], (present[i, 0], present[i, 1]),
                   fontsize=6.4, color=viz.INK_2, zorder=5) for i in top]
try:
    from adjustText import adjust_text
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", color=viz.GRID, lw=0.5))
except Exception:
    pass
ax.set_title("b  Conditions in the same space (marker size = prevalence)")
ax.set_xlabel(f"MCA axis 1 ({inertia[0]:.1%} inertia)")
ax.set_ylabel(f"MCA axis 2 ({inertia[1]:.1%} inertia)")
fig.suptitle("c3_comorbidities — 4 classes by multiple correspondence analysis",
             fontsize=12, fontweight="bold", x=0.005, ha="left")
fig.tight_layout()
save(fig, "fig5_c3_mca",
     "MCA is correspondence analysis of the disjunctive coding — the categorical "
     "analogue of PCA, and the right linear ordination for binary indicators. Axis 1 "
     "orders classes by number of distinct codes (14 / 47 / 59 / 109), i.e. the leading "
     "dimension of this domain is healthcare contact rather than a disease pattern.")

print("[c3] class prevalence heatmap")
prof3 = pd.read_csv(RESULTS / "c3_comorbidities" / "c3_class_profiles.csv")
prof3 = prof3[prof3.K == K3]
pv = prof3.pivot_table(index="phenotype", columns="class", values="class_prevalence")
spread = (pv.max(axis=1) - pv.min(axis=1)).sort_values(ascending=False)
pv = pv.loc[spread.head(24).index]
pv = pv[[fit_index("c3_comorbidities", k) for k in range(K3)]]
fig, ax = plt.subplots(figsize=(6.6, 7.4))
im = ax.imshow(pv.to_numpy(), cmap=viz.SEQ_BLUE, aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(K3))
ax.set_xticklabels([n.replace("-", "-\n") for n in N3], fontsize=7.5)
ax.set_yticks(range(len(pv)))
ax.set_yticklabels([s[:46] for s in pv.index], fontsize=7)
for i in range(len(pv)):
    for j in range(K3):
        v = pv.to_numpy()[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                color=viz.SURFACE if v > 0.55 else viz.INK_2)
cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cb.set_label("prevalence within class", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
ax.set_title("c3 — conditions that most separate the classes")
ax.grid(False)
save(fig, "fig6_c3_prevalence",
     "The 24 conditions with the widest between-class spread. Cell values are "
     "within-class prevalence.")

# ============================================================== cross-domain
print("[cross] alluvial")
sh = pd.read_csv(RESULTS / "shared_cohort_class_assignments.csv", dtype={"MRN": str})
doms = ["c1_physiologic", "c2_symptomatic", "c3_comorbidities"]
codes, stage_labels = [], []
for d in doms:
    K = CHOICES[d]
    nm = names_for(d, K)
    idx = {n: i for i, n in enumerate(nm)}
    codes.append(sh[d].map(idx).to_numpy())
    stage_labels.append(nm)
fig, ax = plt.subplots(figsize=(12.4, 6.2))
viz.alluvial(ax, codes, stage_labels, viz.class_palette(CHOICES[doms[0]]))
for s, d in enumerate(doms):
    ax.text(np.linspace(0, 1, 3)[s], -0.055, d, ha="center", fontsize=9,
            fontweight="bold", color=viz.INK)
ax.set_title("Cross-domain flow on the shared cohort (n = 70,880)", pad=18)
save(fig, "fig7_cross_domain_alluvial",
     "Ribbons are coloured by c1 profile. Each c1 class splits into c2 and c3 in almost "
     "the same proportions as every other — that near-uniform splitting is what "
     "statistical independence looks like. Cramér's V: c1xc2 0.031, c1xc3 0.086, "
     "c2xc3 0.080.")

# ================================================== nonlinear embeddings
print("[embeddings] UMAP + t-SNE (continuum diagnostic)")
from sklearn.manifold import TSNE  # noqa: E402
import umap  # noqa: E402

S1 = names_for("c1_physiologic", K1, short=True)
S3 = names_for("c3_comorbidities", K3, short=True)

# the completed matrix again -- UMAP cannot take NaN, and dropping incomplete rows
# would embed NOX T3 alone. Embeddings are cached: they are the slow part and are a
# deterministic function of (data, seed), so re-styling a panel should not refit them.
CACHE = WORK / "embeddings.npz"
sub1 = rng.choice(len(X1c), EMBED_N, replace=False)
sub_t = rng.choice(EMBED_N, 15000, replace=False)
sub3 = rng.choice(len(B), EMBED_N, replace=False)
if CACHE.exists():
    z = np.load(CACHE)
    U1, T1, U3, sub1, sub_t, sub3 = (z["U1"], z["T1"], z["U3"], z["sub1"],
                                     z["sub_t"], z["sub3"])
    print(f"    loaded cached embeddings from {CACHE.name}")
else:
    t0 = time.perf_counter()
    U1 = umap.UMAP(n_neighbors=30, min_dist=0.1,
                   random_state=SEED).fit_transform(X1c[sub1])
    print(f"    c1 UMAP {time.perf_counter()-t0:.0f}s")
    t0 = time.perf_counter()
    T1 = TSNE(n_components=2, perplexity=40, init="pca",
              random_state=SEED).fit_transform(X1c[sub1][sub_t])
    print(f"    c1 t-SNE {time.perf_counter()-t0:.0f}s")
    t0 = time.perf_counter()
    U3 = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="jaccard",
                   random_state=SEED).fit_transform(B[sub3].astype(bool))
    print(f"    c3 UMAP (jaccard) {time.perf_counter()-t0:.0f}s")
    np.savez_compressed(CACHE, U1=U1, T1=T1, U3=U3, sub1=sub1, sub_t=sub_t, sub3=sub3)


def trim(ax, E, lo=0.4, hi=99.6, pad=0.03):
    """Clip to percentiles so a handful of detached outliers cannot set the scale."""
    x0, x1 = np.percentile(E[:, 0], [lo, hi])
    y0, y1 = np.percentile(E[:, 1], [lo, hi])
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)


def spread_labels(E, labels, K):
    """Label anchors pushed apart along the dominant axis so they do not collide."""
    out = []
    for k in range(K):
        pts = E[labels == k]
        out.append([np.median(pts[:, 0]), np.median(pts[:, 1])])
    out = np.array(out, float)
    span = np.ptp(E[:, 1]) or 1.0
    order_y = np.argsort(out[:, 1])
    for a, b in zip(order_y[:-1], order_y[1:]):
        if out[b, 1] - out[a, 1] < 0.14 * span:
            out[b, 1] = out[a, 1] + 0.14 * span
    return out

fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.9))
ax = axes[0]
anch = spread_labels(U1, lab1[sub1], K1)
for k in range(K1):
    s = lab1[sub1] == k
    ax.scatter(U1[s, 0], U1[s, 1], s=1.6, c=P1[k], alpha=0.35, linewidths=0,
               rasterized=True, label=N1[k])
    viz.label_at(ax, anch[k], S1[k], P1[k])
ax.set_title("a  c1 UMAP, by latent profile")
trim(ax, U1)
ax.legend(loc="upper left", markerscale=6, handletextpad=0.4, labelspacing=0.25,
          fontsize=7)

ax = axes[1]
sc = ax.scatter(U1[:, 0], U1[:, 1], c=np.clip(x1.index_ahi.to_numpy()[sub1], 0, 80),
                s=1.6, cmap=viz.SEQ_BLUE, alpha=0.6, linewidths=0, rasterized=True)
cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("index AHI", fontsize=8, color=viz.INK_2)
cb.outline.set_visible(False)
ax.set_title("b  c1 UMAP, by AHI")
trim(ax, U1)

ax = axes[2]
dsub = x1.STUDY_DEVICE.to_numpy()[sub1]
for d, colr in zip(["NOX T3", "WatchPAT"], [viz.CLASS_COLORS[0], viz.CLASS_COLORS[1]]):
    s = dsub == d
    ax.scatter(U1[s, 0], U1[s, 1], s=1.6, c=colr, alpha=0.3, linewidths=0,
               rasterized=True)
    viz.label_at(ax, [np.median(U1[s, 0]), np.median(U1[s, 1])], d, colr)
ax.set_title("c  c1 UMAP, by device")
trim(ax, U1)

ax = axes[3]
anch3 = spread_labels(U3, lab3[sub3], K3)
for k in range(K3):
    s = lab3[sub3] == k
    ax.scatter(U3[s, 0], U3[s, 1], s=1.6, c=P3[k], alpha=0.35, linewidths=0,
               rasterized=True, label=N3[k])
    viz.label_at(ax, anch3[k], S3[k], P3[k], fontsize=8)
ax.set_title("d  c3 UMAP (Jaccard), by latent class")
trim(ax, U3, 1.5, 98.5)
ax.legend(loc="upper left", markerscale=6, handletextpad=0.4, labelspacing=0.25,
          fontsize=7)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Nonlinear embeddings — a continuum-versus-discrete diagnostic, NOT validation",
             fontsize=12, fontweight="bold", x=0.005, ha="left")
fig.tight_layout()
save(fig, "fig8_umap_diagnostic",
     "READ THIS AS: does the cloud look like one continuous manifold with classes as "
     "smooth bands (a gradient that the model has cut), or like genuinely disjoint "
     "lobes (discrete subpopulations)? UMAP produces crisp separated islands even from "
     "structureless data, and its between-cluster distances, blob sizes and densities "
     f"carry no meaning — it cannot confirm a class solution. n = {EMBED_N:,} subsample.")

fig, ax = plt.subplots(figsize=(5.6, 5.2))
lab_t = lab1[sub1][sub_t]
for k in range(K1):
    s = lab_t == k
    ax.scatter(T1[s, 0], T1[s, 1], s=2.0, c=P1[k], alpha=0.35, linewidths=0,
               rasterized=True)
    viz.label_at(ax, [np.median(T1[s, 0]), np.median(T1[s, 1])], S1[k], P1[k])
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
trim(ax, T1)
ax.set_title("c1 t-SNE — independent cross-check of the same question")
save(fig, "fig9_c1_tsne",
     "Same caveats as UMAP. Shown because the two methods have different failure modes; "
     "agreement between them is weak evidence about continuity, not about class validity.")

print("\ndone.")
