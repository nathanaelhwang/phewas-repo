# %% [markdown]
# ProcWAS Step 04 - Volcano + Manhattan plots (Part 1 = Procedure type, Part 2 = Body system),
# adjustment / top-hit tables, and SUMMARY.md. Both Parts read the single association run and
# differ only in the category used to group/colour points. Mirrors utilwas/10 + medwas/11.
# Volcano: x = log2(IRR), y = -log10 p, coloured by category, top-30 labelled.
# Manhattan: features grouped/coloured by category, ^/v = IRR >/< 1, Bonferroni + FDR lines,
#            top-12 labelled. Focus contrasts {omnibus, trend, severe_vs_none} x {unadjusted, m4}.
# DE-IDENTIFIED (procedure labels + stats only).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import procwas_utils as U

RD = dict(keep_default_na=False, na_values=[""])
PARTS = [("part1_procedure_type", "procedure_type"), ("part2_body_system", "body_system")]


def _nl(p):
    return -np.log10(np.clip(pd.to_numeric(p, errors="coerce"), 1e-300, 1.0))


def _prep(d, cat_col):
    d = d.copy()
    d["p"] = pd.to_numeric(d["p"], errors="coerce")
    d["effect"] = pd.to_numeric(d["effect"], errors="coerce")
    d["cat"] = d[cat_col].fillna("other")
    return d


def manhattan(d, cat_col, title, path, colors):
    d = _prep(d, cat_col).dropna(subset=["p"])
    if len(d) < 2:
        return
    d = d.sort_values(["cat", "feature"]).reset_index(drop=True)
    d["x"] = np.arange(len(d)); d["nl"] = _nl(d["p"])
    fig, ax = plt.subplots(figsize=(15, 6))
    for cat, g in d.groupby("cat"):
        up = g["effect"].fillna(1) > 1
        ax.scatter(g["x"][up], g["nl"][up], s=22, marker="^", color=colors.get(cat, "gray"), label=cat)
        ax.scatter(g["x"][~up], g["nl"][~up], s=22, marker="v", color=colors.get(cat, "gray"))
    ax.axhline(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)
    if d["sig_fdr"].any():
        ax.axhline(_nl(d.loc[d["sig_fdr"], "p"].max()), ls=":", color="navy", lw=0.8)
    for _, r in d.nlargest(12, "nl").iterrows():
        ax.annotate(str(r["feature"])[:26], (r["x"], r["nl"]), fontsize=6, rotation=30, ha="left", va="bottom")
    ax.set_ylabel("-log10(p)"); ax.set_xlabel("procedure (grouped by category)")
    ax.set_title(title); ax.set_xticks([])
    ax.legend(ncol=8, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def volcano(d, cat_col, title, path, colors):
    d = _prep(d, cat_col).dropna(subset=["p", "effect"])
    d = d[np.isfinite(d["effect"])]
    et = d["effect_type"].iloc[0] if len(d) else ""
    if et in ("IRR", "OR"):
        d = d[d["effect"] > 0]
        d["x"] = np.log2(d["effect"]); xlabel = "log2(rate ratio)"
    else:
        d["x"] = d["effect"]
        xlabel = {"spearman_rho": "Spearman rho", "rank_biserial": "rank-biserial r"}.get(et, et)
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
    texts = [ax.text(r["x"], r["nl"], str(r["feature"])[:26], fontsize=6) for _, r in d.nlargest(30, "nl").iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
    except Exception:
        pass
    ax.set_xlabel(xlabel); ax.set_ylabel("-log10(p)"); ax.set_title(title)
    ax.legend(ncol=6, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> None:
    U.ensure_dirs()
    res = pd.concat([pd.read_csv(U.ANALYSIS / f"results_{w}.csv", **RD) for w, _ in U.WINDOWS], ignore_index=True)
    res["sig_fdr"] = res["sig_fdr"].astype(str).str.lower().isin(["true", "1"])
    tot = pd.concat([pd.read_csv(U.ANALYSIS / f"results_total_{w}.csv", **RD) for w, _ in U.WINDOWS], ignore_index=True)

    n = 0
    for part_dir, cat_col in PARTS:
        pdir = U.PLOTS / part_dir
        pdir.mkdir(parents=True, exist_ok=True)
        cats = sorted(res[cat_col].fillna("other").unique())
        colors = {c: plt.cm.tab20(i % 20) for i, c in enumerate(cats)}
        for w, _ in U.WINDOWS:
            base = res[res.window == w]
            for con in ["severe_vs_none", "trend"]:
                for mod in ["unadjusted", "m4"]:
                    sub = base[(base.contrast == con) & (base.model == mod)]
                    volcano(sub, cat_col, f"ProcWAS proc/yr | {w} | {con} | {mod} | by {cat_col}",
                            pdir / f"volcano_{w}_{con}_{mod}.png", colors); n += 1
            for con in ["omnibus", "trend", "severe_vs_none"]:
                for mod in ["unadjusted", "m4"]:
                    sub = base[(base.contrast == con) & (base.model == mod)]
                    manhattan(sub, cat_col, f"ProcWAS proc/yr | {w} | {con} | {mod} | by {cat_col}",
                              pdir / f"manhattan_{w}_{con}_{mod}.png", colors); n += 1

    # ---- tables + SUMMARY ----
    lines = ["# ProcWAS summary (procedures/yr; Poisson-rate IRR)\n",
             f"Windows {[w for w, _ in U.WINDOWS]}; models unadjusted/m1/m2/m3/m4; 7 OSA contrasts; "
             f">={U.MIN_PATIENTS}-patient floor. Part 1 groups by Procedure type, Part 2 by Body system.\n"]
    for w, _ in U.WINDOWS:
        rw = res[res.window == w]
        adj = (rw.query("contrast=='severe_vs_none' and model in ['m1','m2','m3','m4']")
               .pivot_table(index="feature", columns="model", values="effect"))
        adj.to_csv(U.TABLES / f"adjustment_severe_vs_none_{w}.csv")
        top = (rw.query("contrast=='trend' and model=='m4' and sig_fdr")
               .sort_values("neglog10p", ascending=False)
               [["feature", "procedure_type", "body_system", "effect", "ci_low", "ci_high", "p", "sig_bon"]])
        top.to_csv(U.TABLES / f"tophits_trend_m4_{w}.csv", index=False)

        fdr_m4 = rw.query("model=='m4' and sig_fdr")["contrast"].value_counts().to_dict()
        th = tot[(tot.window == w) & (tot.model == "m4")]
        def _e(con):
            r = th[th.contrast == con]
            return f"IRR={r['effect'].iloc[0]:.3f} (p={r['p'].iloc[0]:.1e})" if len(r) else "na"
        lines += [f"## {w}",
                  f"- descriptors tested: {rw['feature'].nunique()}",
                  f"- m4 FDR-significant by contrast: {fdr_m4}",
                  f"- m4 top trend hits (FDR): {len(top)} -> tables/tophits_trend_m4_{w}.csv",
                  f"- ALL-procedures/yr vs severity (m4): severe_vs_none {_e('severe_vs_none')}; trend {_e('trend')}\n"]

    (U.ANALYSIS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    total_png = sum(1 for _ in U.PLOTS.rglob("*.png"))
    print(f"generated {n} plot calls; {total_png} PNGs under {U.PLOTS}")
    print("wrote tables ->", U.TABLES, "and", U.ANALYSIS / "SUMMARY.md")
    U.log_milestone(f"Step 04 plots/tables: {total_png} volcano/Manhattan PNGs (Part1 procedure_type + "
                    f"Part2 body_system) + adjustment/tophit tables + SUMMARY.")


if __name__ == "__main__":
    main()
