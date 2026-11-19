"""Cumulative-incidence curves for the octant phenotypes.

Unlike the archetype version (script 23, which had to plot descriptively because no
archetype contrast was significant), these plot *significant* contrasts: 15 one-vs-rest
tests survive Bonferroni at body-system level and 6 at phecode level, all at m4.

One panel per significant contrast, and the panel shows exactly the comparison the model
made -- the focal octant against the **pooled other seven**, not eight lines. Two reasons:

  * an eight-line chart cannot be made colourblind-safe. The validated 8-slot palette
    fails the all-pairs CVD check (worst pair dE 3.2 protan), and curves in a CIF plot
    cross arbitrarily, so the all-pairs rule is the one that applies. Faceting is the
    fix the guidance prescribes for exactly this case.
  * one-vs-rest *is* the estimand. Drawing the other seven separately would invite the
    reader to eyeball pairwise gaps that were never tested.

Colour therefore carries POLARITY, not identity: warm = hazard above the pooled rest,
cool = below, grey = the pooled rest itself. Two hues plus a neutral, the diverging job.

Estimator: Aalen-Johansen cumulative incidence with death as a competing risk, from the
tie-aware `_pheno_engine.aalen_johansen`. 87% of observations here sit on a tied time
(event dates are daily), and the earlier local implementation walked observations one at a
time, so it depended on the row order among ties and ran up to 0.039 percentage points off
-- see script 27, which audits both against lifelines. Curves are UNADJUSTED; the HR
annotated on each panel is the adjusted m4 estimate. Where those two disagree the
covariates are doing the work, which is worth seeing.

Usage:  python 25_octant_survival_plots.py --level system
        python 25_octant_survival_plots.py --level phecode
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent-class-analysis"
                       / "analysis"))
import incwas_utils as U  # noqa: E402
import _cox_engine as E  # noqa: E402
import _pheno_engine as P  # noqa: E402
from common import viz  # noqa: E402

viz.style()
matplotlib.rcParams["path.simplify"] = False
OUTDIR = P.OUT / "survival_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

UP, DOWN = viz.CLASS_COLORS[1], viz.CLASS_COLORS[0]      # warm = raised, cool = lowered
BITS = {"mild-all": (0, 0, 0), "hypoxemia-predominant": (1, 0, 0),
        "symptom-predominant": (0, 1, 0), "comorbidity-predominant": (0, 0, 1),
        "hypoxemic-symptomatic": (1, 1, 0), "hypoxemic-comorbid": (1, 0, 1),
        "symptomatic-comorbid": (0, 1, 1), "high-all": (1, 1, 1)}


def glyph(o):
    return "".join("■" if b else "□" for b in BITS[o])


aalen_johansen = P.aalen_johansen     # tie-aware; see _pheno_engine and script 27's audit


def panel(ax, surv, octant, hr, lo, hi, p, title):
    """Focal octant vs the pooled other seven."""
    foc = surv[surv["octant"] == octant]
    rest = surv[surv["octant"] != octant]
    c = UP if hr > 1 else DOWN
    grid_r = P.event_time_grid(rest["time"], rest["event"])
    grid = P.event_time_grid(foc["time"], foc["event"])
    cif_r = aalen_johansen(rest["time"], rest["event"], rest["is_death"], grid_r)
    cif_f = aalen_johansen(foc["time"], foc["event"], foc["is_death"], grid)
    ax.plot(grid_r, 100 * cif_r, color=viz.INK_MUTED, lw=2.0, zorder=2,
            drawstyle="steps-post")
    ax.plot(grid, 100 * cif_f, color=c, lw=2.2, zorder=3, drawstyle="steps-post")
    label_time = grid[-1] * 0.60
    label_index = np.searchsorted(grid, label_time, side="right") - 1
    viz.label_at(ax, (label_time, 100 * cif_f[label_index]),
                 glyph(octant), c, fontsize=9)
    # keep the two end labels from stacking when the curves finish close together
    yf, yr = 100 * cif_f[-1], 100 * cif_r[-1]
    sep = 0.13 * max(yf, yr, 1e-9)
    if abs(yf - yr) < sep:
        push = (sep - abs(yf - yr)) / 2
        s = 1 if yf >= yr else -1
        yf, yr = yf + s * push, yr - s * push
    ax.text(grid[-1] * 1.02, yf, f"{octant}\n{len(foc):,} pts, "
            f"{int(foc['event'].sum()):,} ev", color=c, fontsize=6.6, va="center",
            ha="left", fontweight="bold")
    ax.text(grid[-1] * 1.02, yr, f"other seven\n{len(rest):,} pts, "
            f"{int(rest['event'].sum()):,} ev", color=viz.INK_MUTED, fontsize=6.6,
            va="center", ha="left")
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks(np.arange(0, grid[-1] + 0.01, 0.5))     # no ticks in the label margin
    ax.text(0.03, 0.955, f"adj. HR {hr:.2f} ({lo:.2f}–{hi:.2f})   p = {p:.1e}",
            transform=ax.transAxes, fontsize=7.4, color=c, fontweight="bold",
            va="top", ha="left")
    ax.set_xlabel("years since index")
    ax.set_ylabel("cumulative incidence (%)")
    ax.set_xlim(0, grid[-1] * 1.42)
    ax.margins(y=0.14)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["system", "phecode"], default="system")
    a = ap.parse_args()

    r = pd.read_csv(P.OUT / f"results_octant_{a.level}.csv",
                    dtype={"outcome": str})
    sig = r[(r.model == "m4") & r.contrast.str.startswith("one_vs_rest")
            & r.sig_bon].sort_values("p").copy()
    sig["octant"] = sig.contrast.str.split(":").str[1]
    print(f"{len(sig)} Bonferroni-significant one-vs-rest contrasts at m4 ({a.level})")

    d = P.load_exposure(("c1_physiologic", "c2_symptomatic", "c3_comorbidities"),
                        with_archetype=False)
    d = P.assign_octants(d)
    E._PREP, E._ID2IX = (d, {}), {v: i for i, v in enumerate(d["id"].to_numpy())}
    prev, excl = E.prev_index(), E.exclude_map()
    ev_all = pd.read_parquet(U.EVENTS)
    ev_all["phecode"] = ev_all["phecode"].astype(str)
    ev_all = ev_all[ev_all["id"].isin(set(d["id"]))]
    ev_all = ev_all[~ev_all["phecode"].map(P.is_circular_outcome)]
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)

    surv_cache, rows = {}, []
    n = len(sig)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, sig.itertuples()):
        key = t.outcome
        if key not in surv_cache:
            surv_cache[key] = (
                P.system_events(d, ev_all, dim, key, rule="anynew")
                if a.level == "system"
                else E.build_survival(key, ev_all[ev_all["phecode"] == key], d,
                                      prev, excl))
        surv = surv_cache[key]
        name = key if a.level == "system" else f"{getattr(t, 'phenotype', key)}"
        title = (f"{name}\n{int(surv['event'].sum()):,} first events / "
                 f"{len(surv):,} at risk")
        panel(ax, surv, t.octant, t.effect, t.ci_low, t.ci_high, t.p, title)
        foc = surv[surv["octant"] == t.octant]
        rest = surv[surv["octant"] != t.octant]
        rows.append({
            "level": a.level, "outcome": key, "name": name, "octant": t.octant,
            "hr_m4": round(t.effect, 3), "ci_low": round(t.ci_low, 3),
            "ci_high": round(t.ci_high, 3), "p": t.p,
            "n_focal": len(foc), "ev_focal": int(foc["event"].sum()),
            "n_rest": len(rest), "ev_rest": int(rest["event"].sum()),
            "cif3_focal_pct": round(100 * aalen_johansen(
                foc["time"], foc["event"], foc["is_death"], [3.0])[-1], 2),
            "cif3_rest_pct": round(100 * aalen_johansen(
                rest["time"], rest["event"], rest["is_death"], [3.0])[-1], 2)})
    for ax in axes[n:]:
        ax.axis("off")

    lvl = "body system" if a.level == "system" else "phecode"
    fig.suptitle(f"Octant phenotypes vs the pooled rest — every panel is Bonferroni-"
                 f"significant at m4 ({lvl} level)",
                 fontsize=13, fontweight="bold", x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0.052 if nrow > 2 else 0.10, 1, 0.985])
    fig.text(0.005, (0.036 if nrow > 2 else 0.075),
             "Aalen-Johansen cumulative incidence with death as a competing risk. Each "
             "panel shows the contrast the Cox model tested: the named octant (coloured) "
             "against the pooled other seven (grey). Colour carries direction only — warm "
             "= hazard raised, cool = lowered — and the ■□□ glyph repeats which of "
             "physiologic / symptom / comorbidity that octant is above the median on. "
             "Curves are UNADJUSTED; the annotated hazard ratio is the m4 estimate "
             "(age spline, sex, race/ethnicity, BMI, insurance, study year, prior "
             "phecode count, encounter rate). Where the drawn gap and the adjusted HR "
             "point differently, the covariates are carrying the difference. "
             + ("At body-system level the outcome is the first NEW phecode in that "
                "system, so everyone is at risk and the estimand is a rate of "
                "diagnostic accrual, not first-ever disease onset. " if a.level ==
                "system" else "") +
             "327.x and 347.x are excluded throughout as circular with the sleep-study "
             "index. Absolute separations are small — a Bonferroni-significant HR of "
             "1.2-1.5 on a 3-year cumulative incidence of a few per cent is a few "
             "percentage points of absolute risk, which is what these panels are for.",
             fontsize=7.5, color=viz.INK_MUTED, ha="left", va="top", wrap=True)
    stem = f"octant_cif_{a.level}"
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / f"{stem}_3yr.csv", index=False)
    print(f"wrote {OUTDIR}/{stem}.pdf/.png  and  {stem}_3yr.csv")
    print(out[["name", "octant", "hr_m4", "cif3_focal_pct", "cif3_rest_pct",
               "ev_focal"]].to_string(index=False))


if __name__ == "__main__":
    main()
