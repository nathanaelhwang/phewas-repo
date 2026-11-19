"""Figure primitives for the latent class / latent profile results.

Colour is assigned by the job it does, not by taste:

* **class identity -> categorical.** Blue / orange / aqua (+ violet at K=4). This
  ordering was validated with the data-viz palette validator under ``--pairs all``
  (the rule for scatter and ordination forms, where any two classes can end up
  adjacent): worst colour-vision-deficient dE 9.2, worst normal-vision dE 16.3,
  both clear of the floors. The four-slot default (which puts yellow next to
  orange) fails the normal-vision floor at dE 13.7, so violet is used instead.
  Aqua sits below 3:1 contrast on a white surface, so the relief rule applies and
  every class carries a direct label -- identity is never colour-alone.
* **magnitude (AHI, comorbidity burden, point density) -> sequential**, one hue
  light-to-dark with a scale legend. No rainbow ramps.

Marks are thin, grids are hairline and solid, and text wears ink colours rather
than the series colour.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path

# --- categorical: validated slot order (light surface) ------------------------
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
# --- ink -----------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#ffffff"
GRID = "#e4e4e1"
# --- sequential: single hue, light -> dark, for magnitude ----------------------
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eaf1fb", "#2a78d6", "#16375f"])
SEQ_GREY = LinearSegmentedColormap.from_list("seq_grey", ["#f7f7f6", "#b9b8b3", "#4a4a47"])
SEQ_WARM = LinearSegmentedColormap.from_list("seq_warm", ["#fdeee6", "#eb6834", "#7d2f11"])
# --- diverging: warm/cool poles, neutral grey midpoint ------------------------
DIV = LinearSegmentedColormap.from_list("div", ["#16375f", "#2a78d6", "#eeeeec",
                                                "#eb6834", "#7d2f11"])

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "grid.alpha": 1.0,          # never dashed
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 140,
}


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)


def class_palette(K: int) -> list[str]:
    if K > len(CLASS_COLORS):
        raise ValueError(
            f"{K} classes exceeds the validated palette (4). Facet into small "
            "multiples or fold the tail rather than generating a new hue.")
    return CLASS_COLORS[:K]


def cov_ellipse(ax, mean, cov, level=0.95, **kw):
    """Draw the ``level`` probability contour of a bivariate Gaussian.

    Shows a latent profile's *fitted* covariance rather than a summary of its
    assigned points, so the overlap a reader sees is the model's own. The radius is
    the chi-square quantile on 2 degrees of freedom — for 95% that is 2.45 SD, not
    the 2 SD that a univariate intuition suggests.
    """
    from scipy.stats import chi2
    r = np.sqrt(chi2.ppf(level, df=2))
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * r * np.sqrt(np.maximum(vals, 0))
    e = Ellipse(xy=mean, width=w, height=h, angle=angle, **kw)
    ax.add_patch(e)
    return e


def density_panel(ax, x, y, cmap=SEQ_GREY, gridsize=140, mincnt=1, bins="log"):
    """Hexbin density instead of a scatter.

    At 40,000-240,000 points a scatter is a solid blob whose apparent shape is set
    by draw order and alpha; a density surface shows where the mass actually is.
    """
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=mincnt, bins=bins,
                   linewidths=0)
    return hb


def label_at(ax, xy, text, color, fontsize=8.5, halo=SURFACE):
    """Direct label with a surface halo so it stays legible over dense marks.

    Every class carries one of these: the validated palette puts aqua below 3:1
    contrast on a white surface, which triggers the relief rule -- identity must
    never rest on colour alone.
    """
    import matplotlib.patheffects as pe
    x, y = float(xy[0]), float(xy[1])
    t = ax.text(x, y, text, color=color, fontsize=fontsize, fontweight="bold",
                ha="center", va="center", zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=2.6, foreground=halo)])
    return t


def ternary_axes(ax, labels, color=INK_2):
    """Draw an equilateral 2-simplex frame for 3-class posterior probabilities."""
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    ax.plot(*np.vstack([V, V[:1]]).T, color=GRID, lw=1.0, zorder=1)
    for f in (0.25, 0.5, 0.75):                       # hairline internal guides
        for a, b, c_ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            p = V[a] + (V[b] - V[a]) * f
            q = V[a] + (V[c_] - V[a]) * f
            ax.plot([p[0], q[0]], [p[1], q[1]], color=GRID, lw=0.5, zorder=1)
    off = [(-0.045, -0.035), (0.045, -0.035), (0.0, 0.045)]
    for v, lab, o in zip(V, labels, off):
        ax.text(v[0] + o[0], v[1] + o[1], lab, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.11, np.sqrt(3) / 2 + 0.11)
    ax.set_aspect("equal")
    ax.axis("off")
    return V


def to_ternary(P, V):
    """Map rows of a 3-column probability matrix onto simplex coordinates."""
    return P @ V


def alluvial(ax, frames, labels_per_stage, colors, gap=0.012, ribbon_alpha=0.55):
    """Ribbon diagram of how one categorical partition maps onto the next.

    ``frames`` is a list of integer label arrays (one per stage), all the same
    length. Ribbons are coloured by the FIRST stage, so the reader can follow a
    starting class across the diagram; near-uniform splitting is what statistical
    independence between two partitions actually looks like.
    """
    n = len(frames[0])
    stages = len(frames)
    xs = np.linspace(0, 1, stages)
    bar_w = 0.035

    # node vertical extents per stage
    nodes = []
    for s, lab in enumerate(frames):
        ks = np.arange(lab.max() + 1)
        counts = np.array([(lab == k).sum() for k in ks], float)
        frac = counts / counts.sum()
        tot_gap = gap * (len(ks) - 1)
        heights = frac * (1 - tot_gap)
        tops = np.concatenate([[0.0], np.cumsum(heights + gap)[:-1]])
        nodes.append(list(zip(tops, tops + heights)))

    for s in range(stages - 1):
        a_lab, b_lab = frames[s], frames[s + 1]
        # running cursor within each node so ribbons stack without overlap
        cur_a = {k: nodes[s][k][0] for k in range(len(nodes[s]))}
        cur_b = {k: nodes[s + 1][k][0] for k in range(len(nodes[s + 1]))}
        pairs = [(i, j, int(((a_lab == i) & (b_lab == j)).sum()))
                 for i in range(len(nodes[s])) for j in range(len(nodes[s + 1]))]
        for i, j, cnt in sorted(pairs, key=lambda t: (t[0], t[1])):
            if cnt == 0:
                continue
            ha = cnt / n * (1 - gap * (len(nodes[s]) - 1))
            hb = cnt / n * (1 - gap * (len(nodes[s + 1]) - 1))
            y0a, y0b = cur_a[i], cur_b[j]
            cur_a[i] += ha
            cur_b[j] += hb
            x0, x1 = xs[s] + bar_w / 2, xs[s + 1] - bar_w / 2
            cx = (x0 + x1) / 2
            verts = [(x0, y0a), (cx, y0a), (cx, y0b), (x1, y0b),
                     (x1, y0b + hb), (cx, y0b + hb), (cx, y0a + ha), (x0, y0a + ha),
                     (x0, y0a)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor=colors[i],
                                   edgecolor="none", alpha=ribbon_alpha, zorder=2))

    for s in range(stages):
        for k, (lo, hi) in enumerate(nodes[s]):
            col = colors[k] if s == 0 else INK_2
            ax.add_patch(PathPatch(
                Path([(xs[s] - bar_w / 2, lo), (xs[s] + bar_w / 2, lo),
                      (xs[s] + bar_w / 2, hi), (xs[s] - bar_w / 2, hi),
                      (xs[s] - bar_w / 2, lo)]),
                facecolor=col, edgecolor="none", zorder=3))
            ax.text(xs[s] + bar_w / 2 + 0.012, (lo + hi) / 2,
                    f"{labels_per_stage[s][k]}  {(hi-lo)/(1-gap*(len(nodes[s])-1)):.0%}",
                    va="center", ha="left", fontsize=7.5, color=INK_2)
    ax.set_xlim(-0.05, 1.28)
    ax.set_ylim(1.03, -0.03)
    ax.axis("off")


def mca(Z_binary, n_components=4, seed=0):
    """Multiple correspondence analysis of a binary indicator matrix.

    MCA is correspondence analysis of the disjunctive (indicator) coding, i.e. the
    categorical analogue of PCA -- the right linear ordination for binary disease
    indicators. Both levels of every variable become columns, so a patient who
    *lacks* a condition contributes as much geometry as one who has it.

    Returns (row_coords, col_coords, explained_inertia, column_names).
    Implemented via a randomised SVD of the column-scaled sparse indicator matrix,
    dropping the trivial first axis, so 236k x 386 stays tractable.
    """
    from scipy import sparse
    from sklearn.utils.extmath import randomized_svd

    n, J = Z_binary.shape
    # disjunctive coding: [not-x, x] for every variable
    Z = sparse.hstack([sparse.csr_matrix(1 - Z_binary), sparse.csr_matrix(Z_binary)],
                      format="csr").astype(np.float64)
    total = Z.sum()
    c = np.asarray(Z.sum(axis=0)).ravel() / total        # column masses
    r = np.asarray(Z.sum(axis=1)).ravel() / total        # row masses (all equal here)
    Dc_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(c, 1e-12)))
    Dr_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(r, 1e-12)))
    M = Dr_inv_sqrt @ (Z / total) @ Dc_inv_sqrt
    U, S, Vt = randomized_svd(M, n_components=n_components + 1, random_state=seed)
    U, S, Vt = U[:, 1:], S[1:], Vt[1:]                   # drop the trivial axis
    row = Dr_inv_sqrt @ U * S
    col = Dc_inv_sqrt @ Vt.T * S
    inertia = S ** 2 / (S ** 2).sum() if S.sum() else S
    names = [f"{j}=0" for j in range(J)] + [f"{j}=1" for j in range(J)]
    return np.asarray(row), np.asarray(col), inertia, names


def discriminant_axes(X, labels, n_components=2):
    """Canonical (linear) discriminant projection of a labelled cloud.

    For a K-class Gaussian mixture the between-class structure lives in exactly
    K-1 dimensions, so at K=3 this 2-D view discards *nothing* about how the
    profiles separate -- unlike a nonlinear embedding, whose axes have no units and
    whose between-cluster distances carry no meaning.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    ok = ~np.isnan(X).any(axis=1)
    lda = LinearDiscriminantAnalysis(n_components=min(n_components,
                                                      len(np.unique(labels)) - 1))
    lda.fit(X[ok], labels[ok])
    Y = np.full((len(X), lda.scalings_.shape[1]), np.nan)
    Y[ok] = lda.transform(X[ok])
    return Y[:, :n_components], lda
