# %% [markdown]
# Step 10 - UtilWAS Manhattan + volcano plots (mirrors labwas/11, icd v2_05).
# Phecode PheWAS: per (combo x window x contrast x part=presence|count) category-colored Manhattan
# (^/v = OR/IRR >1 / <1, Bonferroni + FDR lines, top-12 disease labels) and volcano
# (x = log2(OR|IRR), y = -log10 p, top-30 labeled). Specialty: volcano + Manhattan per (family x window).
# Focus contrasts {omnibus, trend, severe_vs_none} x models {unadjusted, m4}. -> analysis/plots/.
# DE-IDENTIFIED (disease/specialty labels + stats only).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utilwas_utils as U

PLOTS = U.ANALYSIS / "plots"
RD = dict(keep_default_na=False, na_values=[""])


def _load(kind):
    return pd.concat([pd.read_csv(U.ANALYSIS / f"results_{kind}_{w}.csv",
                     dtype={"feature": str, "phecode": str, "combo": str}, **RD)
                     for w, _ in U.WINDOWS], ignore_index=True)


def _nl(p):
    return -np.log10(np.clip(pd.to_numeric(p, errors="coerce"), 1e-300, 1.0))


def manhattan(d, label_col, cat_col, title, path, colors):
    d = d.copy()
    d["p"] = pd.to_numeric(d["p"], errors="coerce")
    d["effect"] = pd.to_numeric(d["effect"], errors="coerce")
    d = d.dropna(subset=["p"])
    if len(d) < 2:
        return
    d["cat"] = d[cat_col].fillna("other")
    d = d.sort_values(["cat", label_col]).reset_index(drop=True)
    d["x"] = np.arange(len(d))
    d["nl"] = _nl(d["p"])
    ratio = d["effect_type"].iloc[0] in ("IRR", "OR")
    fig, ax = plt.subplots(figsize=(15, 6))
    for cat, g in d.groupby("cat"):
        up = g["effect"].fillna(1 if ratio else 0) > (1 if ratio else 0)
        ax.scatter(g["x"][up], g["nl"][up], s=22, marker="^", color=colors.get(cat, "gray"), label=cat)
        ax.scatter(g["x"][~up], g["nl"][~up], s=22, marker="v", color=colors.get(cat, "gray"))
    ax.axhline(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)
    if d["sig_fdr"].any():
        ax.axhline(_nl(d.loc[d["sig_fdr"], "p"].max()), ls=":", color="navy", lw=0.8)
    for _, r in d.nlargest(12, "nl").iterrows():
        ax.annotate(str(r[label_col])[:24], (r["x"], r["nl"]), fontsize=6, rotation=30, ha="left", va="bottom")
    ax.set_ylabel("-log10(p)"); ax.set_xlabel("feature (grouped by category)")
    ax.set_title(title); ax.set_xticks([])
    ax.legend(ncol=8, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def volcano(d, label_col, cat_col, title, path, colors):
    d = d.copy()
    d["p"] = pd.to_numeric(d["p"], errors="coerce")
    d["effect"] = pd.to_numeric(d["effect"], errors="coerce")
    d = d.dropna(subset=["p", "effect"])
    d = d[np.isfinite(d["effect"])]
    if len(d) < 2:
        return
    d["cat"] = d[cat_col].fillna("other")
    et = d["effect_type"].iloc[0]
    if et in ("IRR", "OR"):
        d = d[d["effect"] > 0]
        d["x"] = np.log2(d["effect"]); xlabel = f"log2({'rate ratio' if et == 'IRR' else 'odds ratio'})"
    else:
        d["x"] = d["effect"]; xlabel = {"spearman_rho": "Spearman rho", "ca_z": "trend z",
                                        "rank_biserial": "rank-biserial r"}.get(et, et)
    if len(d) < 2:
        return
    d["nl"] = _nl(d["p"])
    fig, ax = plt.subplots(figsize=(9.5, 8))
    for cat, g in d.groupby("cat"):
        ax.scatter(g["x"], g["nl"], s=26, color=colors.get(cat, "gray"), label=cat, alpha=0.85)
    ax.axvline(0, color="gray", lw=0.7)
    ax.axhline(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)
    if d["sig_fdr"].any():
        ax.axhline(_nl(d.loc[d["sig_fdr"], "p"].max()), ls=":", color="navy", lw=0.8)
    texts = [ax.text(r["x"], r["nl"], str(r[label_col])[:26], fontsize=6) for _, r in d.nlargest(30, "nl").iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
    except Exception:
        pass
    ax.set_xlabel(xlabel); ax.set_ylabel("-log10(p)"); ax.set_title(title)
    ax.legend(ncol=8, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    phe = _load("phecode")
    spec = _load("specialty")
    for df in (phe, spec):
        df["sig_fdr"] = df["sig_fdr"].astype(str).str.lower().isin(["true", "1"])

    # color map: phecode categories + specialty families
    cats = sorted(set(phe["category"].fillna("other").unique()) | set(spec["family"].unique()))
    colors = {c: plt.cm.tab20(i % 20) for i, c in enumerate(cats)}

    n = 0
    # ---- phecode plots ----
    for combo in sorted(phe["combo"].unique()):
        cdir = PLOTS / "phecode" / combo
        cdir.mkdir(parents=True, exist_ok=True)
        for w, _ in U.WINDOWS:
            base = phe[(phe.combo == combo) & (phe.window == w)]
            for part in ["presence", "count_among_present"]:
                pd_ = base[base.part == part]
                pk = "OR" if part == "presence" else "IRR"
                for con in ["omnibus", "trend"]:
                    for mod in ["unadjusted", "m4"]:
                        sub = pd_[(pd_.contrast == con) & (pd_.model == mod)]
                        if len(sub) >= 2:
                            manhattan(sub, "phenotype", "category",
                                      f"{combo} {part}({pk}) | {w} | {con} | {mod}",
                                      cdir / f"manhattan_{part}_{w}_{con}_{mod}.png", colors)
                            n += 1
                for con in ["severe_vs_none", "trend"]:
                    for mod in ["unadjusted", "m4"]:
                        sub = pd_[(pd_.contrast == con) & (pd_.model == mod)]
                        if len(sub) >= 2:
                            volcano(sub, "phenotype", "category",
                                    f"{combo} {part}({pk}) | {w} | {con} | {mod}",
                                    cdir / f"volcano_{part}_{w}_{con}_{mod}.png", colors)
                            n += 1

    # ---- specialty plots ----
    sdir = PLOTS / "specialty"
    sdir.mkdir(parents=True, exist_ok=True)
    for fam in sorted(spec["family"].unique()):
        for w, _ in U.WINDOWS:
            base = spec[(spec.family == fam) & (spec.window == w)]
            for con in ["trend", "severe_vs_none"]:
                for mod in ["unadjusted", "m4"]:
                    sub = base[(base.contrast == con) & (base.model == mod)]
                    if len(sub) >= 2:
                        volcano(sub, "feature", "family", f"{fam} encounters/yr | {w} | {con} | {mod}",
                                sdir / f"volcano_{fam}_{w}_{con}_{mod}.png", colors)
                        manhattan(sub, "feature", "family", f"{fam} encounters/yr | {w} | {con} | {mod}",
                                  sdir / f"manhattan_{fam}_{w}_{con}_{mod}.png", colors)
                        n += 2

    total = sum(1 for _ in PLOTS.rglob("*.png"))
    print(f"generated {n} plot calls; {total} PNGs under {PLOTS}")
    U.log_milestone(f"Step 10 plots: {total} Manhattan/volcano PNGs (phecode per-combo + specialty).")


if __name__ == "__main__":
    main()
