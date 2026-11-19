"""Cumulative-incidence curves stratified by the six clinical phenotypes.

**These are descriptive.** No archetype contrast is Bonferroni- or FDR-significant at m4,
at phecode level (0 of 2,208 tests) or body-system level (0 of 108). Only 6 of 2,208
phecode tests reach even nominal p<0.05, against ~110 expected by chance — the archetype
arms are small (447-2,030 patients) with 1.42 y mean follow-up, and the ridge penalty
shrinks a rare binary dummy far harder than a continuous score (variance 0.017-0.074 vs
0.547), attenuating the archetype coefficients by roughly 65%.

So the outcomes plotted here are **not** selected on archetype significance — that set is
empty. They are the outcomes that are m4 **Bonferroni**-significant for the continuous
score axes (the powered representation of the same phenotypes) *and* clear the
negative-control ascertainment floor. The figures then show what the six phenotypes
actually look like on those outcomes, so a reader can see the separation — or the lack of
it — rather than infer it from a null p-value.

Estimator: Aalen-Johansen cumulative incidence with death as a competing risk. A
Kaplan-Meier 1-S curve would overstate absolute risk here because mortality is not
negligible over the follow-up window.
"""
from __future__ import annotations

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

# Six lines: the validated categorical order, checked on the adjacent pairlist that
# applies to line charts (worst CVD dE 9.1, worst normal-vision dE 19.6). Three of the
# six fall below 3:1 contrast on white, so every curve is also directly labelled.
ARCH = ["triple-low", "symptomatic-without-hypoxemia", "comorbid-without-hypoxemia",
        "severe-but-asymptomatic", "severe-symptomatic-otherwise-well", "triple-high"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SHORT = {"triple-low": "triple-low", "symptomatic-without-hypoxemia": "sympt-no-hypox",
         "comorbid-without-hypoxemia": "comorbid-no-hypox",
         "severe-but-asymptomatic": "severe-asympt",
         "severe-symptomatic-otherwise-well": "severe-sympt-well",
         "triple-high": "triple-high"}

# m4 Bonferroni-significant for a score axis AND clearing the NCO floor.
OUTCOMES = [
    ("250.2", "Type 2 diabetes", "c1"), ("278.1", "Obesity", "c1"),
    ("509.1", "Respiratory failure", "c1"), ("428.2", "Heart failure", "c1"),
    ("250.41", "Impaired fasting glucose", "c1"), ("782.3", "Edema", "c1"),
    ("440.9", "Atherosclerosis of aorta", "c3"), ("798", "Malaise and fatigue", "c3"),
    ("512.7", "Shortness of breath", "c3"),
]


def aalen_johansen(t, ev, death, grid):
    """Cumulative incidence of the event treating death as a competing risk."""
    o = np.argsort(t)
    t, ev, death = np.asarray(t)[o], np.asarray(ev)[o], np.asarray(death)[o]
    n = len(t)
    S, cif, out, at = 1.0, 0.0, [], n
    j = 0
    for g in grid:
        while j < n and t[j] <= g:
            d_e, d_d = int(ev[j] == 1), int(death[j] == 1 and ev[j] == 0)
            if at > 0 and (d_e or d_d):
                cif += S * d_e / at
                S *= (1 - (d_e + d_d) / at)
            at -= 1
            j += 1
        out.append(cif)
    return np.array(out)


def main() -> None:
    d = P.load_exposure(("c1_physiologic", "c2_symptomatic", "c3_comorbidities"),
                        with_archetype=True)
    E._PREP, E._ID2IX = (d, {}), {v: i for i, v in enumerate(d["id"].to_numpy())}
    prev, excl = E.prev_index(), E.exclude_map()
    ev_all = pd.read_parquet(U.EVENTS)
    ev_all["phecode"] = ev_all["phecode"].astype(str)
    ev_all = ev_all[ev_all["id"].isin(set(d["id"]))]

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 13.2))
    summary = []
    for ax, (phe, name, axis) in zip(axes.ravel(), OUTCOMES):
        dropped = []
        surv = E.build_survival(phe, ev_all[ev_all["phecode"] == phe], d, prev, excl)
        # grey reference band: everyone at risk, regardless of archetype
        grid = P.event_time_grid(surv["time"], surv["event"])
        ref = aalen_johansen(surv["time"], surv["event"], surv["is_death"], grid)
        ax.plot(grid, 100 * ref, color=viz.INK_MUTED, lw=2.4, alpha=0.55, zorder=1,
                drawstyle="steps-post")
        ax.text(3.02, 100 * ref[-1], "all", color=viz.INK_MUTED, fontsize=7,
                va="center", ha="left")
        for a, c in zip(ARCH, COLORS):
            s = surv[surv["archetype"] == a]
            n_ev = int(s["event"].sum())
            if len(s) < 100 or n_ev < 5:
                dropped.append(f"{SHORT[a]} ({n_ev} ev)")
                continue
            grid = P.event_time_grid(s["time"], s["event"])
            cif = aalen_johansen(s["time"], s["event"], s["is_death"], grid)
            # a curve resting on <20 events is drawn thin and dashed: the step size is
            # then bigger than any difference the curve could be showing
            thin = n_ev < 20
            ax.plot(grid, 100 * cif, color=c, lw=1.2 if thin else 1.9,
                    ls=(0, (4, 2)) if thin else "-", zorder=3, drawstyle="steps-post")
            ax.text(3.02, 100 * cif[-1], f"{SHORT[a]} ({n_ev})", color=c, fontsize=6.8,
                    va="center", ha="left", fontweight="bold")
            summary.append({"phecode": phe, "outcome": name, "archetype": a,
                            "n": len(s), "events": int(s["event"].sum()),
                            "cif_3yr_pct": round(100 * cif[-1], 2)})
        ax.set_title(f"{name}\n{int(surv['event'].sum()):,} events / "
                     f"{len(surv):,} at risk   (signal on {axis})", fontsize=9)
        ax.set_xlabel("years since index")
        ax.set_ylabel("cumulative incidence (%)")
        ax.set_xlim(0, 4.05)
        ax.margins(y=0.12)
        if dropped:
            ax.text(0.02, 0.97, "too few events to draw: " + ", ".join(dropped),
                    transform=ax.transAxes, fontsize=6, color=viz.INK_MUTED,
                    va="top", ha="left")
    fig.suptitle("Cumulative incidence by clinical phenotype — DESCRIPTIVE, "
                 "no archetype contrast is significant",
                 fontsize=13, fontweight="bold", x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.text(0.005, 0.028,
             "Aalen-Johansen cumulative incidence, death treated as a competing risk. "
             "Outcomes are those that are m4 BONFERRONI-significant for the continuous "
             "score axes AND clear the negative-control ascertainment floor — the set of "
             "outcomes significant for the six ARCHETYPES is empty (0 of 2,208 phecode "
             "tests and 0 of 108 body-system tests at m4, Bonferroni or FDR). Curves are "
             "unadjusted; the grey line is all at-risk patients. Numbers in brackets are "
             "that phenotype's EVENT count — most are under 40, and curves below 20 events "
             "are drawn dashed because one event moves them more than any real difference "
             "would. The visually dramatic gaps (triple-high heart failure, for instance) "
             "rest on single-digit event counts and are precisely what the null Cox models "
             "say is indistinguishable from noise. Read these as whether the phenotypes "
             "separate at all, not as effect estimates.",
             fontsize=7.5, color=viz.INK_MUTED, ha="left", va="top", wrap=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"phenotype_cif_panels.{ext}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary).to_csv(OUTDIR / "phenotype_cif_3yr.csv", index=False)
    print(f"wrote {OUTDIR}/phenotype_cif_panels.pdf/.png")
    s = pd.DataFrame(summary)
    print("\n3-year cumulative incidence (%) by phenotype:")
    print(s.pivot(index="outcome", columns="archetype",
                  values="cif_3yr_pct").to_string())


if __name__ == "__main__":
    main()
