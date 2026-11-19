# %% [markdown]
# Step 06 - BehWAS forest plots (the deliverable), adapted from
# analyses/icd_analysis/v2_05_filtered_plots.py `forest()`.
#
# One row per behavioral outcome x window, effect + 95% CI whiskers, reference line at the null.
# Because the effect scale is still an open decision, we render BOTH:
#   - OR forests   (binary outcome forms; log x-axis; reference at OR=1)
#   - beta forests (continuous outcome forms on the rank-inverse-normal scale; reference at 0)
# for the two headline contrasts (severe_vs_none, trend) at the M4-adjusted model, plus an
# adjustment-ladder (M1/M3/M4) OR forest for severe_vs_none. Windowed outcomes contribute a
# [1yr] and [5yr] row each; lifetime outcomes (no-show/off-work) contribute a single [lifetime] row.
#
# Emits analysis/plots/forest_*.png + analysis/tables/forest_values.csv. DE-IDENTIFIED only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import behwas_utils as U

HEADLINE = [("severe_vs_none", "m4"), ("trend", "m4")]
MODEL_COLORS = {"m1": "0.6", "m3": "steelblue", "m4": "firebrick"}


def _clean(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.dropna(subset=["effect", "ci_low", "ci_high"]).copy()
    return sub[sub["unstable"].fillna("") == ""]


def forest(sub: pd.DataFrame, title: str, path: Path, scale: str) -> None:
    sub = _clean(sub)
    if sub.empty:
        return
    sub = sub.sort_values("effect")
    labels = (sub["outcome"] + " [" + sub["window"] + "]").tolist()
    y = np.arange(len(sub))
    colors = np.where(sub["sig_fdr"].to_numpy(), "firebrick", "0.5")

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(sub) + 1.5))
    ax.errorbar(sub["effect"], y,
                xerr=[sub["effect"] - sub["ci_low"], sub["ci_high"] - sub["effect"]],
                fmt="none", ecolor="0.6", capsize=2, lw=1, zorder=1)
    ax.scatter(sub["effect"], y, s=28, c=colors, zorder=2)
    ref = 1.0 if scale == "ratio" else 0.0
    ax.axvline(ref, color="black", lw=0.7, ls="--")
    if scale == "ratio":
        ax.set_xscale("log")
        ax.set_xlabel("Odds Ratio (95% CI)")
    else:
        ax.set_xlabel("beta (95% CI, rank-inverse-normal scale)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def forest_ladder(sub: pd.DataFrame, title: str, path: Path) -> None:
    """OR forest with M1/M3/M4 shown as offset colored points per outcome."""
    sub = _clean(sub[sub["model"].isin(["m1", "m3", "m4"])])
    if sub.empty:
        return
    order = (sub.query("model=='m4'").sort_values("effect")["outcome"].tolist()
             or sub["outcome"].unique().tolist())
    pos = {o: i for i, o in enumerate(order)}
    off = {"m1": -0.22, "m3": 0.0, "m4": 0.22}

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.5))
    for model in ["m1", "m3", "m4"]:
        g = sub[sub["model"] == model]
        yy = [pos[o] + off[model] for o in g["outcome"] if o in pos]
        gg = g[g["outcome"].isin(pos)]
        ax.errorbar(gg["effect"], yy,
                    xerr=[gg["effect"] - gg["ci_low"], gg["ci_high"] - gg["effect"]],
                    fmt="o", ms=4, color=MODEL_COLORS[model], ecolor=MODEL_COLORS[model],
                    capsize=2, lw=1, label=model.upper())
    ax.axvline(1.0, color="black", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o + " [" + sub[sub.outcome == o]["window"].iloc[0] + "]" for o in order], fontsize=8)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    U.PLOTS.mkdir(parents=True, exist_ok=True)
    U.TABLES.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(U.ANALYSIS / "results_all.csv")

    plotted = []
    for contrast, model in HEADLINE:
        for scale, etype in [("ratio", "OR"), ("beta", "beta")]:
            sub = res.query("contrast == @contrast and model == @model and effect_type == @etype")
            tag = "OR" if scale == "ratio" else "beta"
            forest(sub, f"BehWAS {tag} — {contrast} ({model.upper()})",
                   U.PLOTS / f"forest_{tag}_{contrast}_{model}.png", scale)
            plotted.append(_clean(sub).assign(scale=scale, panel=f"{contrast}_{model}"))

    # adjustment ladder (M1/M3/M4) for the headline binary contrast
    ladder = res.query("contrast == 'severe_vs_none' and effect_type == 'OR'")
    forest_ladder(ladder, "BehWAS OR — severe_vs_none (M1/M3/M4)",
                  U.PLOTS / "forest_OR_severe_vs_none_ladder.png")

    if plotted:
        pd.concat(plotted, ignore_index=True).to_csv(U.TABLES / "forest_values.csv", index=False)

    print(f"Wrote forest plots -> {U.PLOTS}")
    print(f"Wrote forest values -> {U.TABLES / 'forest_values.csv'}")
    U.log_milestone("Step 06 forest plots: OR + beta forests for severe_vs_none & trend (M4) "
                    "plus an M1/M3/M4 ladder written to analysis/plots/.")


if __name__ == "__main__":
    main()
