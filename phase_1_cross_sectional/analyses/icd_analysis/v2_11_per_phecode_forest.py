# %% [markdown]
# v2 Step 11 - Per-phecode forest plots comparing the published M4 (L1 conditional)
# severe_vs_none effect against each reweighted sensitivity estimate, on the log OR scale.
# One figure per (phecode, method): M4 vs IPTW, and M4 vs ATO.
#   - feature set : phecodes Bonferroni-significant in M4 severe_vs_none
#   - ranking     : M4 p-value ascending, ties broken by larger |log OR| (v2_09 convention)
#   - top 20 -> plots/correlation/, remainder -> plots/correlation/significant other/
# ATO confidence intervals are not stored in the v2_08 CSV, so both weight schemes are
# recomputed here via ipw.marginal_or_sandwich (verified to reproduce the saved ORs/CIs).
#
# Usage: python v2_11_per_phecode_forest.py

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U
import v2_08_ipw_severe_vs_none as ipw           # reuse load_design + marginal_or_sandwich

OUT = U.V2_RESULTS / "sensitivity_ipw"
PLOTS = OUT / "plots" / "correlation"
OTHER = PLOTS / "significant other"
COMBINED = U.V2_SUMMARY / "phewas_results_combined.csv"
TOPN = 20

METHODS = [("ipw", "IPTW", "stabilized, truncated", "#1f77b4"),
           ("ato", "ATO", "overlap weights", "#2a9d5c")]


def recompute_weighted() -> dict[str, pd.DataFrame]:
    """Marginal OR + 95% CI per phecode under IPTW and ATO weights."""
    grp, fm, codes, cov = ipw.load_design(U.V2_PYPHEWAS, None)
    g = grp["genotype"].to_numpy(np.float64)
    ps = np.clip(smf.logit("genotype ~ " + " + ".join(cov), data=grp).fit(disp=0, maxiter=200)
                 .predict(grp).to_numpy(), 1e-6, 1 - 1e-6)
    p_bar = g.mean()
    sw = g * (p_bar / ps) + (1 - g) * ((1 - p_bar) / (1 - ps))
    lo, hi = np.quantile(sw, ipw.TRUNC)
    sw = np.clip(sw, lo, hi)                                  # PRIMARY: stabilized IPTW
    ato = g * (1 - ps) + (1 - g) * ps                         # SECONDARY: overlap weights

    out = {}
    for key, w in [("ipw", sw), ("ato", ato)]:
        r = ipw.marginal_or_sandwich(fm, g, w)
        r["phecode"] = codes
        out[key] = r[["phecode", "OR", "OR_low", "OR_high", "p", "valid", "unstable", "n_carriers"]]
    return out


def build_table() -> pd.DataFrame:
    """M4 Bonferroni hits with M4 + IPTW + ATO log ORs and CIs, ranked by importance."""
    comb = pd.read_csv(COMBINED, dtype={"PheWAS Code": str})
    m4 = comb[(comb.contrast == "severe_vs_none") & (comb.model == "m4_bmi")
              & (comb.sig_bon == True)].copy()
    m4 = m4.rename(columns={"PheWAS Code": "phecode", "PheWAS Name": "phecode_name",
                            "category_string": "category", "p-val": "p_m4",
                            "OR": "OR_m4", "OR_low": "OR_m4_low", "OR_high": "OR_m4_high"})
    d = m4[["phecode", "phecode_name", "category", "p_m4", "OR_m4", "OR_m4_low", "OR_m4_high"]]

    w = recompute_weighted()
    for key, _, _, _ in METHODS:
        d = d.merge(w[key].rename(columns={"OR": f"OR_{key}", "OR_low": f"OR_{key}_low",
                                           "OR_high": f"OR_{key}_high", "p": f"p_{key}",
                                           "unstable": f"unstable_{key}"})
                    .drop(columns=["valid"]),
                    on="phecode", how="left", suffixes=("", f"_{key}"))

    or_cols = ["OR_m4", "OR_m4_low", "OR_m4_high"] + \
              [f"OR_{k}{s}" for k, _, _, _ in METHODS for s in ("", "_low", "_high")]
    bad = d[or_cols].isna().any(axis=1) | (d[or_cols] <= 0).any(axis=1)
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} phecodes with missing/non-positive OR or CI")
        d = d[~bad]
    for key, _, _, _ in METHODS:
        if d[f"unstable_{key}"].any():
            print(f"NOTE: {int(d[f'unstable_{key}'].sum())} phecodes flagged unstable under {key}")

    for c in or_cols:
        d["log_" + c] = np.log(d[c])
    d = (d.sort_values(["p_m4", "log_OR_m4"], key=lambda s: s if s.name == "p_m4" else -s.abs())
           .reset_index(drop=True))
    d["rank"] = np.arange(1, len(d) + 1)
    return d


def forest(row: pd.Series, key: str, short: str, sub: str, color: str, path: Path) -> None:
    x_m4, lo_m4, hi_m4 = row.log_OR_m4, row.log_OR_m4_low, row.log_OR_m4_high
    x_me, lo_me, hi_me = row[f"log_OR_{key}"], row[f"log_OR_{key}_low"], row[f"log_OR_{key}_high"]
    delta = x_me - x_m4
    overlap = (lo_m4 <= hi_me) and (lo_me <= hi_m4)

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ys = [1, 0]
    for y, (x, lo, hi, c) in zip(ys, [(x_m4, lo_m4, hi_m4, "#444444"),
                                      (x_me, lo_me, hi_me, color)]):
        ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="o", color=c, ecolor=c,
                    elinewidth=1.6, capsize=4, markersize=6)
        ax.annotate(f"{x:+.3f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=c)

    ax.axvline(0, ls="--", color="grey", lw=1, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(["M4\n(L1 conditional)", f"{short}\n({sub})"], fontsize=8.5)
    for tick, c in zip(ax.get_yticklabels(), ["#444444", color]):
        tick.set_color(c)
    ax.set_ylim(-0.7, 1.7)

    span = [lo_m4, hi_m4, lo_me, hi_me, 0.0]
    pad = 0.18 * (max(span) - min(span) + 1e-6)
    ax.set_xlim(min(span) - pad, max(span) + pad)
    ax.set_xlabel("log OR (severe vs none)   —   dashed line = no effect")

    name = row.phecode_name if len(str(row.phecode_name)) <= 46 else str(row.phecode_name)[:43] + "..."
    ax.set_title(f"[{row['rank']}] {row.phecode}  {name}\n"
                 f"{row.category}   |   {int(row.n_carriers):,} carriers", fontsize=10)

    note = (f"$\\Delta$ = {delta:+.3f}   CIs overlap: {'yes' if overlap else 'NO'}\n"
            f"OR  M4 {row.OR_m4:.3f}   {short} {row[f'OR_{key}']:.3f}")
    ax.text(0.985, 0.04, note, transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    OTHER.mkdir(parents=True, exist_ok=True)
    d = build_table()
    print(f"{len(d)} M4 Bonferroni-significant phecodes -> top {TOPN} in main folder, "
          f"{max(0, len(d) - TOPN)} in '{OTHER.name}'")

    n = 0
    for row in d.itertuples(index=False):
        row = pd.Series(row._asdict())
        dest = PLOTS if row["rank"] <= TOPN else OTHER
        safe = str(row.phecode).replace("/", "_").replace("\\", "_")
        for key, short, sub, color in METHODS:
            forest(row, key, short, sub, color, dest / f"{row['rank']:03d}_{safe}_{key}.png")
            n += 1
    print(f"wrote {n} forest plots")

    idx = PLOTS / "per_phecode_index.csv"
    keep = ["rank", "phecode", "phecode_name", "category", "n_carriers", "p_m4",
            "OR_m4", "OR_m4_low", "OR_m4_high"]
    for key, _, _, _ in METHODS:
        keep += [f"OR_{key}", f"OR_{key}_low", f"OR_{key}_high", f"p_{key}"]
    out = d[keep].copy()
    for key, _, _, _ in METHODS:
        out[f"delta_log_or_{key}"] = d[f"log_OR_{key}"] - d["log_OR_m4"]
        out[f"ci_overlap_{key}"] = ((d.log_OR_m4 <= d[f"log_OR_{key}_high"])
                                    & (d[f"log_OR_{key}_low"] <= d.log_OR_m4_high))
    out["folder"] = np.where(out["rank"] <= TOPN, ".", OTHER.name)
    out.to_csv(idx, index=False)
    print(f"wrote {idx}")
    for key, short, _, _ in METHODS:
        print(f"  {short}: CIs overlap M4 for {int(out[f'ci_overlap_{key}'].sum())}/{len(out)} phecodes")


if __name__ == "__main__":
    main()
