# %% [markdown]
# Step 05 - Postprocess the full incidence Cox scan into human-readable summaries.
#
# From analysis/results_cox.csv:
#   summary/cox_results_combined.csv  - the full long table (copy, canonical location)
#   summary/adjustment_ladder.csv     - severe-vs-none HR across m1->m4 per phecode (attenuation)
#   summary/dose_response.csv         - m4 HR by severity (mild/mod/severe) + trend per phecode
#   summary/README.md                 - counts, FDR-significant tallies, top positive/negative hits
# DE-IDENTIFIED (effect sizes + counts only).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import incwas_utils as U


def main() -> None:
    U.ensure_dirs()
    d = pd.read_csv(U.ANALYSIS / "results_cox.csv", dtype={"phecode": str})
    d.to_csv(U.SUMMARY / "cox_results_combined.csv", index=False)

    # ---- adjustment ladder: severe-vs-none HR across models --------------------
    sv = d[d.contrast == "severe_vs_none"]
    ladder = sv.pivot_table(index=["phecode", "phenotype", "category"], columns="model",
                            values="effect", aggfunc="first")
    ladder = ladder.reindex(columns=["m1", "m2", "m3", "m4"])
    ev = sv.groupby("phecode")["n_events"].first()
    ladder["n_events"] = ladder.index.get_level_values("phecode").map(ev)
    ladder.reset_index().to_csv(U.SUMMARY / "adjustment_ladder.csv", index=False)

    # ---- dose-response: m4 HR by severity + trend ------------------------------
    m4 = d[d.model == "m4"]
    dr = m4.pivot_table(index=["phecode", "phenotype", "category"], columns="contrast",
                        values="effect", aggfunc="first")
    for c in ["mild_vs_none", "moderate_vs_none", "severe_vs_none", "trend", "ahi_ge5", "ahi_ge15"]:
        if c not in dr.columns:
            dr[c] = np.nan
    trend_p = m4[m4.contrast == "trend"].set_index("phecode")["p"]
    dr["trend_p"] = dr.index.get_level_values("phecode").map(trend_p)
    sev_fdr = m4[m4.contrast == "severe_vs_none"].set_index("phecode")["sig_fdr"]
    dr["severe_sig_fdr"] = dr.index.get_level_values("phecode").map(sev_fdr)
    dr = dr[["mild_vs_none", "moderate_vs_none", "severe_vs_none", "trend", "trend_p",
             "ahi_ge5", "ahi_ge15", "severe_sig_fdr"]]
    dr.reset_index().to_csv(U.SUMMARY / "dose_response.csv", index=False)

    # ---- README ----------------------------------------------------------------
    n_phe = d["phecode"].nunique()
    L = ["# incwas — incidence Cox PheWAS results\n",
         f"- Phecodes tested (>= {U.MIN_EVENTS} incident events): **{n_phe:,}**",
         f"- Total model fits (rows): {len(d):,}",
         f"- Exposure: OSA severity (ref None); cause-specific Cox; m1-m4 ladder; harmonized washout; "
         f"prefix-matched phecode v1.2; administrative cutoff {U.ADATE_CUTOFF}.\n",
         "## FDR-significant phecodes by contrast (m4)\n",
         "| contrast | FDR-sig | Bonferroni-sig |", "|---|--:|--:|"]
    for c in ["severe_vs_none", "moderate_vs_none", "mild_vs_none", "trend", "ahi_ge5", "ahi_ge15"]:
        sub = d[(d.contrast == c) & (d.model == "m4")]
        L.append(f"| {c} | {int(sub.sig_fdr.sum()):,} | {int(sub.sig_bon.sum()):,} |")

    def top(direction, n=20):
        h = m4[(m4.contrast == "severe_vs_none") & m4.sig_fdr & m4.effect.notna()].copy()
        h = h[h.effect > 1] if direction == "risk" else h[h.effect < 1]
        h = h.sort_values("effect", ascending=(direction != "risk")).head(n)
        return h

    for title, dirn in [("Top incident-RISK (HR>1) — severe vs none, m4, FDR-sig", "risk"),
                        ("Top incident-PROTECTIVE (HR<1) — severe vs none, m4, FDR-sig", "prot")]:
        L.append(f"\n## {title}\n")
        L.append("| phecode | phenotype | HR (95% CI) | p | n_events | PH? |")
        L.append("|---|---|--:|--:|--:|:--:|")
        for r in top(dirn).itertuples():
            ph = "!" if (np.isfinite(r.ph_p) and r.ph_p < 0.05) else ""
            L.append(f"| {r.phecode} | {str(r.phenotype)[:40]} | {r.effect:.2f} "
                     f"({r.ci_low:.2f}-{r.ci_high:.2f}) | {r.p:.1e} | {int(r.n_events):,} | {ph} |")

    (U.SUMMARY / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote summary tables + README to {U.SUMMARY}")
    print(f"phecodes: {n_phe:,}; severe-vs-none m4 FDR-sig: "
          f"{int(m4[(m4.contrast=='severe_vs_none')].sig_fdr.sum()):,}")
    U.log_milestone(f"Step 05 postprocess: {n_phe:,} phecodes; adjustment ladder + dose-response + README.")


if __name__ == "__main__":
    main()
