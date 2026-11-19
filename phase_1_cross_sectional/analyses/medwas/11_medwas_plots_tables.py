# %% [markdown]
# Step 11 - MedWAS Manhattan plots, adjustment / top-hit tables, and SUMMARY.md.
#
# Mirrors labwas/11: results merged with gpi_category.csv; Manhattan sorted + colored by
# super-heading, markers up/down by OR>1, Bonferroni + BH-FDR lines, top hits labeled.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import medwas_utils as U

PLOTS = U.ANALYSIS / "plots"
TABLES = U.ANALYSIS / "tables"
# (contrast, model) combos to render Manhattans for
KEY = [("trend", "unadjusted"), ("trend", "m4"),
       ("severe_vs_none", "unadjusted"), ("severe_vs_none", "m4"),
       ("omnibus", "m4")]


def manhattan(d: pd.DataFrame, title: str, path: Path) -> None:
    d = d[np.isfinite(d["p"])].copy()
    if d.empty:
        return
    d["cat"] = d["category"].fillna("other")
    d = d.sort_values(["cat", "code"]).reset_index(drop=True)
    d["x"] = np.arange(len(d))
    cats = sorted(d["cat"].unique())
    colors = {c: plt.cm.tab20(i / max(len(cats), 1)) for i, c in enumerate(cats)}
    up = d["effect"] > 1 if d["effect_type"].iloc[0] == "OR" else d["effect"] > 0

    fig, ax = plt.subplots(figsize=(15, 6))
    for c, g in d.groupby("cat"):
        gu = g[up.loc[g.index]]; gd = g[~up.loc[g.index]]
        ax.scatter(gu["x"], gu["neglog10p"], s=16, color=colors[c], marker="^", label=c)
        ax.scatter(gd["x"], gd["neglog10p"], s=16, color=colors[c], marker="v")
    m = len(d)
    ax.axhline(-np.log10(0.05 / m), ls="--", c="red", lw=0.8, label="Bonferroni")
    ax.axhline(-np.log10(0.05), ls=":", c="gray", lw=0.8)
    for _, r in d.nlargest(12, "neglog10p").iterrows():
        nm = r.get("drug_class_name")
        nm = str(nm).strip() if pd.notna(nm) and str(nm).strip() else str(r["code"])
        ax.annotate(nm[:28], (r["x"], r["neglog10p"]), fontsize=6,
                    rotation=30, ha="left", va="bottom")
    ax.set_ylabel("-log10(p)"); ax.set_title(title)
    ax.set_xticks([]); ax.legend(fontsize=6, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(U.RESULTS / "gpi_category.csv", dtype={"code": str})
    lab = cat.set_index("code")["drug_class_name"].to_dict()

    lines = ["# MedWAS summary\n", f"Engine binary logistic; windows {[w for w,_ in U.WINDOWS]}.\n"]
    for label, _ in U.WINDOWS:
        res = pd.read_csv(U.ANALYSIS / f"results_{label}.csv", dtype={"code": str})
        res = res.merge(cat[["code", "category", "drug_class_name"]], on="code", how="left")

        for contrast, model in KEY:
            sub = res.query("contrast == @contrast and model == @model")
            manhattan(sub, f"MedWAS {label} {contrast} {model}",
                      PLOTS / f"manhattan_{label}_{contrast}_{model}.png")

        # adjustment table: severe_vs_none OR across models
        adj = (res.query("contrast=='severe_vs_none' and model in ['m1','m3','m4']")
               .pivot_table(index="code", columns="model", values="effect"))
        adj.insert(0, "drug_class", adj.index.map(lambda g: lab.get(g, "")))
        adj.to_csv(TABLES / f"adjustment_severe_vs_none_{label}.csv")

        # top hits: m4 trend, FDR-significant
        top = (res.query("contrast=='trend' and model=='m4' and sig_fdr")
               .assign(drug_class=lambda d: d["code"].map(lambda g: lab.get(g, "")))
               .sort_values("neglog10p", ascending=False)
               [["code", "drug_class", "category", "effect", "ci_low", "ci_high", "p", "sig_bon"]])
        top.to_csv(TABLES / f"tophits_trend_m4_{label}.csv", index=False)

        n_tested = res["code"].nunique()
        fdr_m4 = res.query("model=='m4' and sig_fdr")["contrast"].value_counts().to_dict()
        lines.append(f"## {label}")
        lines.append(f"- GPI-4 classes tested: {n_tested}")
        lines.append(f"- m4 FDR-significant counts by contrast: {fdr_m4}")
        lines.append(f"- top m4-trend hits (FDR): {len(top)}; see tables/tophits_trend_m4_{label}.csv\n")

    (U.ANALYSIS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote plots ->", PLOTS)
    print("Wrote tables ->", TABLES)
    print("Wrote", U.ANALYSIS / "SUMMARY.md")
    U.log_milestone("Step 11 plots/tables/SUMMARY written.")


if __name__ == "__main__":
    main()
