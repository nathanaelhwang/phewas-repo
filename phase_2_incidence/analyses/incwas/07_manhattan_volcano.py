# %% [markdown]
# Step 07 - Phenome-wide Manhattan + volcano for the incidence Cox scan (severe-vs-none, m4).
#
# Manhattan: phecodes ordered by category on x, -log10(p) on y, colored by category, with BH-FDR
# and Bonferroni threshold lines; top hits labeled. Volcano: ln(HR) vs -log10(p), colored by
# direction x significance. DE-IDENTIFIED (aggregate association statistics only).
#
# Usage: python 07_manhattan_volcano.py
# Emits plots/manhattan_severe_m4.png, plots/volcano_severe_m4.png.

# %%
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import incwas_utils as U

CONTRAST, MODEL = "severe_vs_none", "m4"


def main() -> None:
    U.ensure_dirs()
    d = pd.read_csv(U.ANALYSIS / "results_cox.csv", dtype={"phecode": str})
    d = d[(d.contrast == CONTRAST) & (d.model == MODEL) & d.p.notna() & d.effect.notna()].copy()
    # drop the circular OSA-recoding control (sleep apnea 327.3*): severe OSA trivially predicts an OSA
    # code, and its off-scale p-value compresses the readable range. Insomnia (327.4) etc. are kept.
    d = d[~d["phecode"].astype(str).str.startswith("327.3")]
    d["category"] = d["category"].fillna("other")
    d = d.sort_values(["category", "phecode"]).reset_index(drop=True)
    d["x"] = np.arange(len(d))
    d["nlp"] = -np.log10(d["p"].clip(lower=1e-300))
    n = len(d)
    fdr_line = -np.log10(0.05 * d["sig_fdr"].sum() / n) if d["sig_fdr"].any() else None
    bonf = -np.log10(0.05 / n)

    cats = list(dict.fromkeys(d["category"]))
    cmap = {c: plt.cm.tab20(i % 20) for i, c in enumerate(cats)}

    # ---- Manhattan ----
    fig, ax = plt.subplots(figsize=(14, 6))
    ticks, labels = [], []
    for c in cats:
        s = d[d.category == c]
        ax.scatter(s["x"], s["nlp"], s=10, color=cmap[c], edgecolors="none")
        ticks.append(s["x"].mean())
        labels.append(c if len(str(c)) < 16 else str(c)[:14] + "..")
    ax.axhline(bonf, color="red", ls="--", lw=0.8, label=f"Bonferroni (p<{0.05/n:.1e})")
    if fdr_line is not None:
        ax.axhline(fdr_line, color="blue", ls=":", lw=0.8, label="BH-FDR 0.05")
    for r in d.sort_values("nlp", ascending=False).head(18).itertuples():
        ax.annotate(str(r.phenotype)[:24], (r.x, r.nlp), fontsize=6,
                    xytext=(0, 3), textcoords="offset points", rotation=25)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"Incidence PheWAS — OSA severe vs none (m4-adjusted Cox), {n:,} phecodes")
    ax.legend(fontsize=8, loc="upper right")
    ax.margins(x=0.01)
    fig.savefig(U.PLOTS / "manhattan_severe_m4.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Volcano ----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    lnhr = np.log(d["effect"])
    up = d.sig_fdr & (d.effect > 1)
    dn = d.sig_fdr & (d.effect < 1)
    ns = ~d.sig_fdr
    ax.scatter(lnhr[ns], d["nlp"][ns], s=8, color="#cccccc", label="NS")
    ax.scatter(lnhr[up], d["nlp"][up], s=12, color="#b2182b", label="FDR risk (HR>1)")
    ax.scatter(lnhr[dn], d["nlp"][dn], s=12, color="#2166ac", label="FDR protective (HR<1)")
    ax.axvline(0, color="k", lw=0.6)
    if fdr_line is not None:
        ax.axhline(fdr_line, color="blue", ls=":", lw=0.8)
    for r in d[d.sig_fdr].reindex(d[d.sig_fdr]["nlp"].sort_values(ascending=False).index).head(20).itertuples():
        ax.annotate(str(r.phenotype)[:22], (np.log(r.effect), r.nlp), fontsize=6,
                    xytext=(3, 0), textcoords="offset points")
    ax.set_xlabel("ln(hazard ratio) — severe vs none")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"Incidence PheWAS volcano — OSA severe vs none (m4)")
    ax.legend(fontsize=8)
    fig.savefig(U.PLOTS / "volcano_severe_m4.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote manhattan + volcano ({n:,} phecodes) to {U.PLOTS}")
    U.log_milestone(f"Step 07 plots: phenome-wide Manhattan + volcano (severe-vs-none, m4), {n:,} phecodes.")


if __name__ == "__main__":
    main()
