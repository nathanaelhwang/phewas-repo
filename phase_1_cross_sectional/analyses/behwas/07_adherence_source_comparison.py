# %% [markdown]
# Step 07 - Adherence source comparison (standalone side-analysis).
#
# Question: is it worth deriving medication adherence from the raw medication.csv fills instead
# of trusting the pre-computed medication_adherence.csv? We compare, per patient, over the same
# STUDY_DATE-anchored 1-year window:
#   REF  = mean per-class PDC from medication_adherence.csv (features/adherence.parquet).
#   MINE = a fills-derived MPR proxy from MedWAS's distinct_fills.parquet: per (id, gpi4),
#          LEAST(1, sum(days_supply)/365); then averaged across classes per patient.
# We report coverage overlap, Pearson/Spearman correlation, and a Bland-Altman agreement summary,
# then give a switch/stay recommendation.
#
# Depends on MedWAS step 01 output (results/medwas_results/fills/distinct_fills.parquet). If it is
# missing, the script explains how to produce it and exits without error.
#
# Emits analysis/adherence_source_comparison.md + analysis/plots/adherence_source_comparison.png
# DE-IDENTIFIED aggregates only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import behwas_utils as U


def main() -> None:
    U.ensure_dirs()
    fills = U.FILLS_PARQUET
    adh_feat = U.FEATURES_DIR / "adherence.parquet"
    report = U.ANALYSIS / "adherence_source_comparison.md"

    if not fills.exists():
        msg = (f"MedWAS distinct_fills.parquet not found at {fills}. "
               f"Run analyses/medwas/01_extract_fills.py first, then re-run this step.")
        print(msg)
        report.write_text(f"# Adherence source comparison\n\n**Skipped:** {msg}\n", encoding="utf-8")
        U.log_milestone(f"Step 07 adherence comparison SKIPPED: {msg}")
        return
    if not adh_feat.exists():
        print(f"Missing {adh_feat}; run 03_extract_adherence.py first.")
        return

    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)
    F = fills.as_posix()
    A = adh_feat.as_posix()

    # MINE: fills-derived MPR proxy over the 1yr window
    mine = con.sql(f"""
        WITH c AS (SELECT id, CAST(STUDY_DATE AS DATE) idx FROM read_parquet('{U.COHORT}')),
             f AS (SELECT id, gpi4, fill_date, TRY_CAST(rxsup AS DOUBLE) rxsup
                   FROM read_parquet('{F}')),
             w AS (SELECT f.id, f.gpi4, SUM(f.rxsup) AS days_sup
                   FROM f JOIN c USING (id)
                   WHERE f.fill_date >= (c.idx - to_years(1))::DATE AND f.fill_date < c.idx
                   GROUP BY f.id, f.gpi4),
             cls AS (SELECT id, LEAST(1.0, days_sup / 365.0) AS mpr FROM w WHERE days_sup IS NOT NULL)
        SELECT id, avg(mpr) AS my_pdc, count(*) AS n_classes FROM cls GROUP BY id
    """).df()

    # REF: pre-computed per-class PDC over the 1yr window
    ref = con.sql(f"""
        WITH c AS (SELECT id, CAST(STUDY_DATE AS DATE) idx FROM read_parquet('{U.COHORT}')),
             a AS (SELECT id, CAST(med_start_date AS DATE) AS msd, TRY_CAST(adherence AS DOUBLE) AS adh
                   FROM read_parquet('{A}'))
        SELECT a.id, avg(a.adh) AS ref_pdc, count(*) AS n_ref
        FROM a JOIN c USING (id)
        WHERE a.msd >= (c.idx - to_years(1))::DATE AND a.msd < c.idx
        GROUP BY a.id
    """).df()

    both = mine.merge(ref, on="id", how="inner")
    n_mine, n_ref, n_both = len(mine), len(ref), len(both)
    x, y = both["ref_pdc"].to_numpy(), both["my_pdc"].to_numpy()
    diff = y - x
    if n_both > 2:
        pear_r = float(np.corrcoef(x, y)[0, 1])
        spear_r = float(stats.spearmanr(x, y)[0])  # index [0] is version-robust
    else:
        pear_r = spear_r = np.nan
    md = float(np.mean(diff)) if n_both else np.nan
    sd = float(np.std(diff, ddof=1)) if n_both > 1 else np.nan
    loa = (md - 1.96 * sd, md + 1.96 * sd)

    # ---- plots: scatter + Bland-Altman -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x, y, s=6, alpha=0.2, linewidths=0)
    axes[0].plot([0, 1], [0, 1], "r--", lw=0.8)
    axes[0].set_xlabel("REF PDC (medication_adherence.csv)")
    axes[0].set_ylabel("MINE MPR (medication.csv fills)")
    axes[0].set_title(f"1yr window agreement (n={n_both:,})")
    mean_ab = (x + y) / 2
    axes[1].scatter(mean_ab, diff, s=6, alpha=0.2, linewidths=0)
    for yv, ls in [(md, "-"), (loa[0], ":"), (loa[1], ":")]:
        axes[1].axhline(yv, color="red", ls=ls, lw=0.8)
    axes[1].set_xlabel("mean(REF, MINE)")
    axes[1].set_ylabel("MINE - REF")
    axes[1].set_title(f"Bland-Altman (bias {md:+.3f}, LoA [{loa[0]:+.3f}, {loa[1]:+.3f}])")
    png = U.PLOTS / "adherence_source_comparison.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- recommendation heuristic ----------------------------------------------
    concordant = (np.isfinite(spear_r) and spear_r >= 0.60 and abs(md) <= 0.10)
    rec = ("**Stay with `medication_adherence.csv`.** The pre-computed PDC concords well with a "
           "fills-derived MPR, so re-deriving from the 33 GB medication.csv is not worth the cost."
           if concordant else
           "**Consider deriving PDC from `medication.csv`.** The two sources disagree materially "
           "(low correlation and/or a non-trivial Bland-Altman bias), so the pre-computed definition "
           "may not match a transparent PDC and a bespoke derivation is worth the extra work.")

    lines = [
        "# Adherence source comparison (1-year window)\n",
        "Compares the pre-computed per-patient PDC (`medication_adherence.csv`) against a "
        "fills-derived MPR proxy (`LEAST(1, sum(days_supply)/365)` per class, averaged) computed "
        "from MedWAS's `distinct_fills.parquet`, over the same STUDY_DATE-anchored 1-year window.\n",
        "## Coverage",
        f"- patients with a fills-derived value (MINE): {n_mine:,}",
        f"- patients with a pre-computed value (REF):   {n_ref:,}",
        f"- patients in BOTH (compared):                {n_both:,}\n",
        "## Agreement",
        f"- Pearson r:  {pear_r:.3f}",
        f"- Spearman rho: {spear_r:.3f}",
        f"- Bland-Altman bias (MINE - REF): {md:+.3f}  (SD {sd:.3f})",
        f"- 95% limits of agreement: [{loa[0]:+.3f}, {loa[1]:+.3f}]",
        f"- REF PDC median: {np.median(x):.3f}; MINE MPR median: {np.median(y):.3f}\n",
        "## Recommendation",
        rec,
        f"\n(See plot: {png.name}. Caveat: MINE is an MPR proxy from days_supply, not a true "
        "gap-based PDC; treat as an order-of-magnitude concordance check.)\n",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"n_both={n_both:,}  Spearman={spear_r:.3f}  bias={md:+.3f}")
    print(f"Wrote {report}")
    print(f"Wrote {png}")
    U.log_milestone(
        f"Step 07 adherence comparison: n={n_both:,}, Spearman {spear_r:.3f}, bias {md:+.3f} "
        f"-> {'stay with pre-computed' if concordant else 'consider medication.csv'}.")


if __name__ == "__main__":
    main()
