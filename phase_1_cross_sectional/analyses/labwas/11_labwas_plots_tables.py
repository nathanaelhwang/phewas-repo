# %% [markdown]
# Step 11 - LabWAS Manhattan/volcano plots + summary tables (mirrors icd v2_05/v2_06).
# Reads analysis/results_{1yr,5yr}.csv + loinc_category.csv. Writes analysis/plots/,
# analysis/tables/, analysis/SUMMARY.md. Category-colored Manhattan per
# (target x window x contrast x model) for the key contrasts/models.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import labwas_utils as U

ANALYSIS = U.RESULTS / "analysis"
PLOTS = ANALYSIS / "plots"
TABLES = ANALYSIS / "tables"
CATS = ["allergy", "blood", "cancer", "cardiovascular", "coagulation", "endocrine",
        "immune", "infectious", "kidney", "liver", "metabolic", "nutrition/vitamins",
        "ob/gyn", "other", "toxicology/pharmacology", "urinary"]
COLORS = {c: plt.cm.tab20(i / len(CATS)) for i, c in enumerate(CATS)}
KEY_CONTRASTS = ["omnibus", "trend"]
KEY_MODELS = ["unadjusted", "m4"]
VOLC_CONTRASTS = ["severe_vs_none", "trend"]
VOLC_MODELS = ["unadjusted", "m4"]


def manhattan(sub: pd.DataFrame, name: pd.DataFrame, title: str, path: Path) -> None:
    d = sub.dropna(subset=["p"]).copy()
    if d.empty:
        return
    d["cat"] = d["category"].fillna("other")
    d = d.sort_values(["cat", "loinc_code"]).reset_index(drop=True)
    d["x"] = np.arange(len(d))
    d["nl"] = d["neglog10p"]
    fig, ax = plt.subplots(figsize=(15, 6))
    for cat, g in d.groupby("cat"):
        up = g["effect"].fillna(0) > (1 if (g["effect_type"].iloc[0] == "IRR") else 0)
        ax.scatter(g["x"][up], g["nl"][up], s=22, marker="^", color=COLORS.get(cat, "gray"), label=cat)
        ax.scatter(g["x"][~up], g["nl"][~up], s=22, marker="v", color=COLORS.get(cat, "gray"))
    ax.axhline(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)          # Bonferroni
    fdr = d.loc[d["sig_fdr"], "p"].max() if d["sig_fdr"].any() else np.nan       # BH boundary
    if np.isfinite(fdr):
        ax.axhline(-np.log10(fdr), ls=":", color="navy", lw=0.8)
    for _, r in d.nlargest(12, "nl").iterrows():
        ax.annotate(str(name.get(r["loinc_code"], r["loinc_code"]))[:22], (r["x"], r["nl"]),
                    fontsize=6, rotation=30, ha="left", va="bottom")
    ax.set_ylabel("-log10(p)"); ax.set_xlabel("analyte (grouped by category)")
    ax.set_title(title); ax.set_xticks([])
    ax.legend(ncol=8, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def volcano(sub: pd.DataFrame, name: dict, title: str, path: Path) -> None:
    d = sub.dropna(subset=["p", "effect"]).copy()
    d = d[np.isfinite(d["effect"])]
    if d.empty:
        return
    d["cat"] = d["category"].fillna("other")
    et = d["effect_type"].iloc[0]
    if et in ("IRR", "OR"):              # log2(ratio) so no-effect sits at x=0
        d = d[d["effect"] > 0]
        d["x"] = np.log2(d["effect"]); xlabel = f"log2({'rate ratio' if et == 'IRR' else 'odds ratio'})"
    else:
        d["x"] = d["effect"]; xlabel = {"beta": "beta (rank-normal SD)", "spearman_rho": "Spearman rho",
                                        "rank_biserial": "rank-biserial r", "ca_z_r": "trend (z/sqrt N)"}.get(et, et)
    if d.empty:
        return
    d["nl"] = d["neglog10p"]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    for cat, g in d.groupby("cat"):
        ax.scatter(g["x"], g["nl"], s=26, color=COLORS.get(cat, "gray"), label=cat, alpha=0.85)
    ax.axvline(0, color="gray", lw=0.7)
    ax.axhline(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)          # Bonferroni
    fdr = d.loc[d["sig_fdr"], "p"].max() if d["sig_fdr"].any() else np.nan       # BH boundary
    if np.isfinite(fdr):
        ax.axhline(-np.log10(fdr), ls=":", color="navy", lw=0.8)
    top = d.nlargest(30, "nl")                                                   # label top 30 by significance
    texts = [ax.text(r["x"], r["nl"], str(name.get(r["loinc_code"], r["loinc_code"]))[:26], fontsize=6)
             for _, r in top.iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
    except Exception:
        pass
    ax.set_xlabel(xlabel); ax.set_ylabel("-log10(p)"); ax.set_title(title)
    ax.legend(ncol=8, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    res = pd.concat([pd.read_csv(ANALYSIS / f"results_{w}.csv") for w in ("1yr", "5yr")], ignore_index=True)
    prop = pd.concat([pd.read_csv(ANALYSIS / f"results_propensity_{w}.csv") for w in ("1yr", "5yr")], ignore_index=True)
    name = dict(zip(pd.read_csv(U.RESULTS / "loinc_category.csv")["loinc_code"],
                    pd.read_csv(U.RESULTS / "loinc_category.csv")["loinc_long_common_name"]))

    # ---- Manhattan plots (key contrasts x models) ----
    for w in ("1yr", "5yr"):
        for t in ["order_rate", "median", "mean"]:
            for con in KEY_CONTRASTS:
                for mod in KEY_MODELS:
                    sub = res[(res.window == w) & (res.target == t) & (res.contrast == con) & (res.model == mod)]
                    manhattan(sub, name, f"LabWAS {t} | {w} | {con} | {mod}",
                              PLOTS / f"manhattan_{t}_{w}_{con}_{mod}.png")
            # Part 2 (order_rate) & value volcanoes: severe_vs_none & trend, unadjusted & M4
            for con in VOLC_CONTRASTS:
                for mod in VOLC_MODELS:
                    sub = res[(res.window == w) & (res.target == t) & (res.contrast == con) & (res.model == mod)]
                    volcano(sub, name, f"LabWAS {t} | {w} | {con} | {mod}",
                            PLOTS / f"volcano_{t}_{w}_{con}_{mod}.png")
        # Part 1 (ordering-propensity) volcanoes — one set per window (not per-target)
        for con in VOLC_CONTRASTS:
            for mod in VOLC_MODELS:
                psub = prop[(prop.window == w) & (prop.contrast == con) & (prop.model == mod)]
                volcano(psub, name, f"ordering-propensity (Part 1) | {w} | {con} | {mod}",
                        PLOTS / f"volcano_propensity_{w}_{con}_{mod}.png")

    # ---- adjustment table: trend effect+p across models, per (target, window) ----
    for w in ("1yr", "5yr"):
        tr = res[(res.window == w) & (res.contrast == "trend")]
        wide = tr.pivot_table(index=["target", "loinc_code", "category"], columns="model",
                              values=["effect", "p", "sig_fdr"], aggfunc="first")
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide.reset_index().to_csv(TABLES / f"adjustment_trend_{w}.csv", index=False)

    # ---- top hits: omnibus M4 FDR-sig, sorted by p, per (target, window) ----
    lines = ["# LabWAS results summary\n"]
    for w in ("1yr", "5yr"):
        for t in ["order_rate", "median", "mean"]:
            om = res[(res.window == w) & (res.target == t) & (res.contrast == "omnibus") & (res.model == "m4")]
            om = om.sort_values("p")
            om.assign(name=om["loinc_code"].map(name)).to_csv(TABLES / f"tophits_omnibus_m4_{t}_{w}.csv", index=False)
            nsig = int(om["sig_fdr"].sum())
            lines.append(f"## {t} | {w} — omnibus (M4 adjusted): {nsig} FDR-significant of {len(om)}")
            for _, r in om[om.sig_fdr].head(12).iterrows():
                lines.append(f"- {r['category']:<22} {r['loinc_code']:<9} {str(name.get(r['loinc_code'],''))[:44]}  p={r['p']:.2e}")
            lines.append("")
    # FDR-sig counts by family
    lines.append("## FDR-significant counts by (target, contrast, model)")
    tab = (res[res.sig_fdr].groupby(["window", "target", "contrast", "model"]).size()
           .rename("n_fdr_sig").reset_index())
    lines.append(tab.to_string(index=False))
    (ANALYSIS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {sum(1 for _ in PLOTS.glob('*.png'))} Manhattan plots, tables, and SUMMARY.md")
    U.log_milestone("Step 11 LabWAS plots+tables: Manhattan/adjustment/top-hits + SUMMARY.md written.")


if __name__ == "__main__":
    main()
