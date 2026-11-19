# %% [markdown]
# v2 Step 5 - Readable plots: regenerate Manhattan / Volcano / Forest for each of the 20 runs,
# labeling ONLY the top-20 POSITIVE (OR>1) associations (ranked by p-value), with repelled,
# non-overlapping labels. Writes results/icd_analysis_v2_results/filtered_plots/.

# %%
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from adjustText import adjust_text
from pyPheWAS.pyPhewasCorev2 import cat_colors

import utils as U

WORK = U.V2_PYPHEWAS
OUT = U.V2_RESULTS / "filtered_plots"
TOPN = 20


def load(name):
    df = pd.read_csv(WORK / f"regressions_{name}.csv", skiprows=1, dtype={"PheWAS Code": str})
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.dropna(subset=["p-val", "beta"]).copy()
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["neglog10p"] = -np.log10(df["p-val"].clip(lower=1e-300))
    df["OR"] = np.exp(df["beta"].clip(-30, 30))
    ci = df["Conf-interval beta"].astype(str).str.strip("[]").str.split(",", expand=True)
    df["OR_low"] = np.exp(pd.to_numeric(ci[0], errors="coerce").clip(-30, 30))
    df["OR_high"] = np.exp(pd.to_numeric(ci[1], errors="coerce").clip(-30, 30))
    df["unstable"] = (df["beta"].abs() > 10) | (pd.to_numeric(df["std_error"], errors="coerce") > 10)
    df["cat"] = df["category_string"].fillna("other")
    return df.dropna(subset=["beta", "neglog10p"])


def thresholds(p):
    p = np.asarray(p, float); p = p[np.isfinite(p)]
    bon = 0.05 / len(p)
    sn = np.sort(p); ok = sn <= 0.05 * np.arange(1, len(sn) + 1) / len(sn)
    fdr = sn[ok].max() if ok.any() else np.nan
    return bon, fdr


def top_positive(df):
    return df[(df["beta"] > 0) & (~df["unstable"])].nsmallest(TOPN, "p-val")


def manhattan(df, name):
    d = df.sort_values(["cat", "PheWAS Code"]).reset_index(drop=True)
    d["x"] = np.arange(len(d))
    bon, fdr = thresholds(d["p-val"].values)
    fig, ax = plt.subplots(figsize=(17, 7))
    for cat, g in d.groupby("cat"):
        c = cat_colors.get(cat, "gray")
        pos, neg = g[g["beta"] > 0], g[g["beta"] <= 0]
        ax.scatter(pos["x"], pos["neglog10p"], s=11, color=c, marker="^", alpha=0.75, linewidths=0)
        ax.scatter(neg["x"], neg["neglog10p"], s=8, color=c, marker="v", alpha=0.30, linewidths=0)
    if np.isfinite(bon):
        ax.axhline(-np.log10(bon), color="red", ls="--", lw=0.9, label="Bonferroni 0.05")
    if np.isfinite(fdr) and fdr > 0:
        ax.axhline(-np.log10(fdr), color="navy", ls=":", lw=0.9, label="FDR 0.05")
    top = top_positive(d)
    texts = [ax.text(r["x"], r["neglog10p"], r["PheWAS Name"], fontsize=8) for _, r in top.iterrows()]
    adjust_text(texts, ax=ax, only_move={"text": "y", "static": "y"},
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5), expand=(1.1, 1.4))
    ax.set_xticks([]); ax.set_xlabel("PheCodes (grouped by category)")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"{name} — Manhattan  (top {TOPN} positive associations labeled; ▲ OR>1, ▼ OR<1)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / f"{name}_Manhattan.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def volcano(df, name):
    bon, fdr = thresholds(df["p-val"].values)
    fig, ax = plt.subplots(figsize=(11, 8))
    neg, pos = df[df["beta"] <= 0], df[df["beta"] > 0]
    ax.scatter(neg["beta"], neg["neglog10p"], s=9, color="steelblue", alpha=0.35, linewidths=0, label="OR<1")
    ax.scatter(pos["beta"], pos["neglog10p"], s=9, color="firebrick", alpha=0.45, linewidths=0, label="OR>1")
    ax.axvline(0, color="black", lw=0.6)
    if np.isfinite(bon):
        ax.axhline(-np.log10(bon), color="red", ls="--", lw=0.9)
    if np.isfinite(fdr) and fdr > 0:
        ax.axhline(-np.log10(fdr), color="navy", ls=":", lw=0.9)
    top = top_positive(df)
    texts = [ax.text(r["beta"], r["neglog10p"], r["PheWAS Name"], fontsize=8) for _, r in top.iterrows()]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5), expand=(1.15, 1.5))
    ax.set_xlabel("log Odds Ratio (beta)"); ax.set_ylabel("-log10(p)")
    ax.set_title(f"{name} — Volcano  (top {TOPN} positive associations labeled)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / f"{name}_Volcano.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def forest(df, name):
    top = top_positive(df).sort_values("OR")
    if not len(top):
        return
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(top) + 1.5))
    y = np.arange(len(top))
    ax.errorbar(top["OR"], y, xerr=[top["OR"] - top["OR_low"], top["OR_high"] - top["OR"]],
                fmt="o", color="firebrick", ecolor="0.5", capsize=2, ms=4, lw=1)
    ax.axvline(1, color="black", lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(top["PheWAS Name"], fontsize=8)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title(f"{name} — top {TOPN} positive associations", fontsize=10)
    fig.savefig(OUT / f"{name}_Forest.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r["name"] for r in json.loads((WORK / "model_spec.json").read_text())["runs"]]
    for name in runs:
        d = load(name)
        manhattan(d, name); volcano(d, name); forest(d, name)
        print(f"  {name}: {len(d)} phecodes, {int((d['beta']>0).sum())} positive -> 3 plots")
    print(f"\nWrote {len(runs)*3} plots -> {OUT}")
    U.log_milestone(f"v2 Step 5 filtered plots: top-{TOPN}-positive labels, {len(runs)*3} PNGs.")


if __name__ == "__main__":
    main()
