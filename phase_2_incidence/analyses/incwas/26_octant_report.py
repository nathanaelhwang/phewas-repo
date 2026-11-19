"""Append the octant results to PHENOTYPE_INCWAS.md.

Script 21 wrote the report for the score axes and the corner archetypes. The archetypes
produced nothing, and the octants replaced them; this appends their sections rather than
rewriting the file, so the archetype failure stays on the record as the reason the octants
exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import _pheno_engine as P

REPORT = P.OUT / "PHENOTYPE_INCWAS.md"
COLS = ["outcome", "n_events", "n_in_group", "effect", "ci_low", "ci_high", "p"]
HEAD = {"effect": "HR", "n_in_group": "n_in_octant"}


def table(df, extra=()):
    c = [x for x in (list(extra) + COLS) if x in df]
    t = df[c].copy()
    for x in ("effect", "ci_low", "ci_high"):
        if x in t:
            t[x] = t[x].round(3)
    t["p"] = t["p"].map(lambda v: f"{v:.2e}")
    for x in ("n_in_group",):
        if x in t:
            t[x] = t[x].astype("Int64")
    return t.rename(columns=HEAD).to_markdown(index=False)


def main() -> None:
    cuts = P.octant_cuts()
    out = ["\n## Octant phenotypes — the replacement for the corner archetypes\n"]
    out.append(
        "Each gradient score is cut at its own cohort median (score_c1 "
        f"{cuts['score_c1']:.4f}, score_c2 {cuts['score_c2']:.4f}, score_c3 "
        f"{cuts['score_c3']:.4f}), giving eight cells that classify **every** patient "
        "(arms 8,349–9,249 of 70,880 in the shared cohort; 2,737–3,527 of 25,380 on the "
        "3-domain incwas spine) rather than the 24.6% the corner rules reached. The cut "
        "points are cohort medians, not clinical thresholds, and do not transfer.\n")
    out.append(
        "**Testing is hierarchical.** Stage 1 is a single 7-df joint Wald test per outcome "
        "— *do the eight phenotypes differ at all here?* — corrected across outcomes on its "
        "own. Stage 2 runs one-vs-rest (each phenotype against the pooled other seven, the "
        "comparison a taxonomy is actually for) only for outcomes whose omnibus clears "
        "p<0.05. At phecode level the gate cuts stage 2 from 2,944 tests to 120, which is "
        "the single biggest lever on what survives multiplicity.\n")
    out.append(
        "**Ridge note.** `_cox_engine` uses penalizer=0.1, which shrinks a rare binary "
        "dummy far harder than a continuous covariate because the L2 penalty is scale-free "
        "and the variances are not (archetype dummies 0.017–0.074 vs 0.547 for a score, "
        "giving ~65% attenuation). Octants sit at ~0.109. These models therefore fit at "
        "penalizer=0.01 and fall back to 0.1 only if that fails; the penalty used is "
        "recorded per row.\n")

    for lvl, title in (("system", "Body-system level (`anynew` rule)"),
                       ("phecode", "Phecode level")):
        r = pd.read_csv(P.OUT / f"results_octant_{lvl}.csv", dtype={"outcome": str})
        om = r[(r.contrast == "omnibus") & (r.model == "m4")]
        ovr = r[r.contrast.str.startswith("one_vs_rest") & (r.model == "m4")].copy()
        ovr["octant"] = ovr.contrast.str.split(":").str[1]
        out.append(f"\n### {title}\n")
        out.append(f"Omnibus, m4: **{int(om.sig_fdr.sum())} of {len(om)}** outcomes "
                   f"FDR-significant ({int(om.sig_bon.sum())} Bonferroni). Gated "
                   f"one-vs-rest: {len(ovr)} tests, **{int(ovr.sig_fdr.sum())} FDR** "
                   f"({int(ovr.sig_bon.sum())} Bonferroni).\n")
        sig = ovr[ovr.sig_fdr].sort_values("p")
        if len(sig):
            extra = ["octant"] + (["phenotype"] if lvl == "phecode" else [])
            out.append(table(sig, extra) + "\n")

    out.append("\n### Reading these\n")
    out.append(
        "- The one-vs-rest contrasts are drawn in "
        "`survival_plots/octant_cif_system.png` and `octant_cif_phecode.png` — one panel "
        "per Bonferroni-significant contrast, focal octant against the pooled other seven, "
        "Aalen-Johansen with death competing. In all 15 body-system panels the unadjusted "
        "3-year CIF gap points the same direction as the adjusted HR, so nothing here "
        "depends on adjustment reversing a raw difference.\n"
        "- **Absolute separations are small.** The largest body-system gap is about 6 "
        "percentage points of 3-year cumulative incidence (symptom-predominant → mental "
        "disorders, 14.1% vs 8.5%). The largest HR in the study, generalized anxiety "
        "disorder at 1.91, rests on 34 focal events and carries the widest CI in the set.\n"
        "- **The mental-disorders signal is not PHQ-2 circularity.** c2 includes a 2-item "
        "depression screen, but restricting to mood/anxiety phecodes (HR 1.56) versus all "
        "other mental phecodes (1.49) gives essentially the same estimate — the effect is "
        "not concentrated in the codes the instrument could have written.\n"
        "- **At body-system level the outcome is the first NEW phecode in that system**, "
        "so everyone is at risk and the estimand is a rate of diagnostic accrual, not "
        "first-ever disease onset. The strict system-wide washout was rejected because it "
        "deletes the exposure (0.2% of high-burden patients retained for respiratory).\n"
        "- **Octants trade contrast for coverage.** Mean between-arm score contrast falls "
        "from 1.91 (corners) to 1.07, but √n × contrast is a wash (55.8 vs 57.0) and every "
        "patient is now in an arm. The corners produced 0 significant results anywhere; "
        "the octants produce 17 FDR / 15 Bonferroni at system level.\n")

    REPORT.open("a", encoding="utf-8").write("\n".join(out))
    print(f"appended octant sections to {REPORT}")


if __name__ == "__main__":
    main()
