# %% [markdown]
# QWAS Step 03 - Manhattan / volcano / forest plots, tables and SUMMARY.md.
#
# The OR phenome (binary prevalence form) and the beta phenome (INT-continuous form) are
# plotted separately because they were FDR-corrected separately; features are grouped and
# coloured by questionnaire domain. Referral-driving airway items (snoring / witnessed
# apnea / gasping / snoring-apnea chief complaint) are ringed in the volcanoes and starred
# in the tables -- their association with severity is partly ascertainment.
# Mirrors procwas/04 + medwas/11.
#
# -> analysis/plots/{manhattan,volcano,forest}_*.png
#    analysis/tables/{adjustment_severe_vs_none,tophits_trend_m4,composites_by_severity}.csv
#    analysis/SUMMARY.md
# DE-IDENTIFIED (item labels + statistics only).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qwas_utils as U

RD = dict(keep_default_na=False, na_values=[""])
# (form, label, adjusted effect_type, null, volcano x-axis) -- the adjusted effect type is
# what the FDR family and the forest/ladder plots key on.
FAMILIES = [("binary", "binary prevalence", "OR", 1.0, "log2(odds ratio)"),
            ("continuous", "continuous (INT)", "beta", 0.0, "beta (INT scale)")]
COMPOSITES = ["ESS_TOTAL", "ISI_TOTAL", "PHQ2_TOTAL", "FOSQ_MEAN"]
# null value per effect type (the unadjusted screen reports its own effect sizes)
NULL_OF = {"OR": 1.0}
# effect types that carry a direction (sign / above-below the null); the omnibus Wald and
# chi-square statistics do not, so those Manhattans use a single marker and get no volcano.
DIRECTIONAL = {"OR", "beta", "spearman_rho", "rank_biserial", "ca_z"}


def _nl(p):
    """-log10(p) with p=0 clipped to the float floor (the strongest hits underflow)."""
    return -np.log10(np.clip(pd.to_numeric(p, errors="coerce"), 1e-300, 1.0))


def _prep(res: pd.DataFrame) -> pd.DataFrame:
    d = res.copy()
    for c in ("p", "effect", "se", "ci_low", "ci_high"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("sig_fdr", "sig_bon", "referral_item"):
        d[c] = d[c].astype(str).str.lower().isin(["true", "1"])
    d["nl"] = _nl(d["p"])
    d["domain"] = d["domain"].fillna("Other")
    return d


def colors_for(domains) -> dict:
    return {c: plt.cm.tab10(i % 10) for i, c in enumerate(domains)}


def manhattan(d, title, path, colors):
    d = d.dropna(subset=["p"])
    if len(d) < 2:
        return
    etype = d["effect_type"].iloc[0]
    null = NULL_OF.get(etype, 0.0)
    directional = etype in DIRECTIONAL
    d = d.assign(_o=d["domain"].map({c: i for i, c in enumerate(U.DOMAIN_ORDER)}).fillna(99))
    d = d.sort_values(["_o", "feature"]).reset_index(drop=True)
    d["x"] = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(15, 6))
    for dom, g in d.groupby("domain", sort=False):
        if not directional:
            ax.scatter(g["x"], g["nl"], s=26, marker="o", color=colors.get(dom, "gray"), label=dom)
            continue
        up = g["effect"].fillna(null) > null
        ax.scatter(g["x"][up], g["nl"][up], s=26, marker="^", color=colors.get(dom, "gray"), label=dom)
        ax.scatter(g["x"][~up], g["nl"][~up], s=26, marker="v", color=colors.get(dom, "gray"))
    ax.axhline(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)
    if d["sig_fdr"].any():
        ax.axhline(_nl(d.loc[d["sig_fdr"], "p"].max()), ls=":", color="navy", lw=0.8)
    for _, r in d.nlargest(12, "nl").iterrows():
        ax.annotate(str(r["feature"])[:24], (r["x"], r["nl"]), fontsize=6.5,
                    rotation=30, ha="left", va="bottom")
    ax.set_ylabel("-log10(p)"); ax.set_xlabel("questionnaire item (grouped by domain)")
    ax.set_title(title); ax.set_xticks([])
    ax.legend(ncol=10, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def volcano(d, xlabel, title, path, colors):
    d = d.dropna(subset=["p", "effect"])
    d = d[np.isfinite(d["effect"])]
    if len(d) < 2:
        return
    etype = d["effect_type"].iloc[0]
    if etype not in DIRECTIONAL:
        return
    d = d.assign(x=np.log2(d["effect"].clip(lower=1e-9)) if etype == "OR" else d["effect"])
    xlabel = {"spearman_rho": "Spearman rho", "rank_biserial": "rank-biserial r",
              "ca_z": "Cochran-Armitage Z"}.get(etype, xlabel)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    for dom, g in d.groupby("domain"):
        ax.scatter(g["x"], g["nl"], s=30, color=colors.get(dom, "gray"), label=dom, alpha=0.85)
    ref = d[d["referral_item"]]
    if len(ref):
        ax.scatter(ref["x"], ref["nl"], s=120, facecolors="none", edgecolors="black",
                   lw=1.0, label="referral-driving")
    ax.axvline(0, color="gray", lw=0.7)
    ax.axhline(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)
    if d["sig_fdr"].any():
        ax.axhline(_nl(d.loc[d["sig_fdr"], "p"].max()), ls=":", color="navy", lw=0.8)
    texts = [ax.text(r["x"], r["nl"], str(r["feature"])[:24], fontsize=6)
             for _, r in d.nlargest(25, "nl").iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
    except Exception:
        pass
    ax.set_xlabel(xlabel); ax.set_ylabel("-log10(p)"); ax.set_title(title)
    ax.legend(ncol=6, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def forest(d, etype, null, xlabel, title, path, colors, height_per_row=0.22):
    """Horizontal effect+CI plot, features grouped by domain (top = first domain)."""
    d = d.dropna(subset=["effect", "ci_low", "ci_high"])
    d = d[np.isfinite(d["effect"]) & np.isfinite(d["ci_low"]) & np.isfinite(d["ci_high"])]
    if not len(d):
        return
    d = d.assign(_o=d["domain"].map({c: i for i, c in enumerate(U.DOMAIN_ORDER)}).fillna(99))
    d = d.sort_values(["_o", "effect"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, height_per_row * len(d) + 1.6)))
    for dom, g in d.groupby("domain", sort=False):
        yy = y[g.index]
        eff, lo, hi = (g["effect"].to_numpy(float), g["ci_low"].to_numpy(float),
                       g["ci_high"].to_numpy(float))
        ax.errorbar(eff, yy, xerr=[eff - lo, hi - eff],
                    fmt="o", ms=3.5, lw=0, elinewidth=1.0, capsize=1.8,
                    color=colors.get(dom, "gray"), label=dom)
    ax.axvline(null, color="gray", lw=0.8, ls="--")
    labels = [f"{f}{' *' if r else ''}" for f, r in zip(d["feature"], d["referral_item"])]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    if etype == "OR":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
    ax.legend(ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.03 - 6 / len(d)))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def forest_ladder(d, etype, null, xlabel, title, path):
    """Composite scores across the m1->m4 adjustment ladder (confounding read-out)."""
    models = ["m1", "m2", "m3", "m4"]
    d = d[d["feature"].isin(COMPOSITES) & d["model"].isin(models)]
    if not len(d):
        return
    cols = {m: plt.cm.viridis(i / 3) for i, m in enumerate(models)}
    feats = [f for f in COMPOSITES if f in set(d["feature"])]
    fig, ax = plt.subplots(figsize=(7.5, 1.1 * len(feats) + 1.8))
    for fi, feat in enumerate(feats):
        for mi, m in enumerate(models):
            r = d[(d["feature"] == feat) & (d["model"] == m)]
            if not len(r):
                continue
            eff, lo, hi = (float(r["effect"].iloc[0]), float(r["ci_low"].iloc[0]),
                           float(r["ci_high"].iloc[0]))
            yy = fi + (1.5 - mi) * 0.18
            ax.errorbar([eff], [yy], xerr=[[eff - lo], [hi - eff]],
                        fmt="o", ms=5, lw=0, elinewidth=1.4, capsize=2.5, color=cols[m],
                        label=m if fi == 0 else None)
    ax.axvline(null, color="gray", lw=0.8, ls="--")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=9)
    if etype == "OR":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
    ax.legend(title="model", fontsize=7, title_fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    U.ensure_dirs()
    res = _prep(pd.read_csv(U.ANALYSIS / "results.csv", **RD))
    descr = pd.read_csv(U.ANALYSIS / "descriptives.csv", **RD)
    colors = colors_for(U.DOMAIN_ORDER)
    for old in U.PLOTS.glob("*.png"):      # this script owns the dir; drop stale reruns
        old.unlink()
    n_png = 0

    for form, form_lab, etype, null, xlabel in FAMILIES:
        fam = res[res["form"] == form]
        # Manhattan / volcano span the unadjusted screen and the fully adjusted model, so the
        # effect type varies by cell (crude OR vs adjusted OR, Spearman rho vs beta, ...).
        for con in ["omnibus", "trend", "severe_vs_none"]:
            for mod in ["unadjusted", "m4"]:
                sub = fam[(fam["contrast"] == con) & (fam["model"] == mod)]
                if not len(sub):
                    continue
                manhattan(sub, f"QWAS | {form_lab} | {con} | {mod}",
                          U.PLOTS / f"manhattan_{form}_{con}_{mod}.png", colors); n_png += 1
                if con != "omnibus":
                    volcano(sub, xlabel, f"QWAS | {form_lab} | {con} | {mod}",
                            U.PLOTS / f"volcano_{form}_{con}_{mod}.png", colors); n_png += 1
        # forests are effect-scale plots -> adjusted models only
        adj = fam[fam["effect_type"] == etype]
        for con in ["trend", "severe_vs_none"]:
            sub = adj[(adj["contrast"] == con) & (adj["model"] == "m4")]
            forest(sub, etype, null, xlabel, f"QWAS | {form_lab} | {con} | m4 (* referral-driving)",
                   U.PLOTS / f"forest_{form}_{con}_m4.png", colors); n_png += 1
        forest_ladder(adj[adj["contrast"] == "severe_vs_none"], etype, null, xlabel,
                      f"QWAS instrument scores | severe vs none | {form_lab} | m1->m4",
                      U.PLOTS / f"forest_composites_{form}.png"); n_png += 1

    # ---- tables ----
    adj = (res[res["contrast"].isin(["severe_vs_none", "trend"])
               & res["model"].isin(["m1", "m2", "m3", "m4"])]
           .pivot_table(index=["feature", "domain", "form", "contrast"],
                        columns="model", values="effect"))
    adj.to_csv(U.TABLES / "adjustment_ladder.csv")

    top = (res[(res["contrast"] == "trend") & (res["model"] == "m4") & res["sig_fdr"]]
           .sort_values("nl", ascending=False)
           [["feature", "domain", "form", "effect_type", "effect", "ci_low", "ci_high",
             "p", "n", "sig_bon", "referral_item"]])
    top.to_csv(U.TABLES / "tophits_trend_m4.csv", index=False)

    comp = descr[descr["feature"].isin(COMPOSITES)].pivot_table(
        index=["feature", "form"], columns="severity", values=["n", "mean", "median", "prevalence"])
    comp.to_csv(U.TABLES / "composites_by_severity.csv")

    # ---- SUMMARY ----
    n_feat = res["feature"].nunique()
    n_sample = len(pd.read_parquet(U.TARGETS_DIR / "targets.parquet", columns=["id"]))
    lines = [
        "# QWAS summary — sleep-questionnaire items vs OSA severity\n",
        f"Prevalence (cross-sectional) scan of **{n_feat} questionnaire features** over "
        f"**{n_sample:,} index-anchored patients** (closest questionnaire on/before the "
        f"sleep-study date; per-item complete-case n runs {int(res['n'].min()):,}–"
        f"{int(res['n'].max()):,}). Exposure, contrasts and covariates match every other WAS stage: "
        "7 contrasts (omnibus / trend / mild|moderate|severe-vs-none / AHI>=5 / AHI>=15) under "
        "unadjusted + m1 crude / m2 +demographics+index-year / m3 +smoking+income+obs_window / "
        "m4 +BMI. Binary items -> logistic OR; ordinal/continuous items -> OLS on the "
        "rank-inverse-normal transform (beta). BH-FDR + Bonferroni within each "
        "(contrast x model x effect_type) family. Analysis is complete-case per item, so n "
        "varies by feature.\n",
        f"- total tests: **{len(res):,}**",
    ]
    piv = (res[res["model"].isin(["m1", "m2", "m3", "m4"])]
           .pivot_table(index="model", columns="effect_type", values="sig_fdr", aggfunc="sum"))
    lines += ["\n## FDR-significant adjusted tests by model\n",
              piv.astype(int).to_markdown(), ""]

    for _form, form_lab, etype, _null, _x in FAMILIES:
        sub = res[(res["contrast"] == "trend") & (res["model"] == "m4")
                  & (res["effect_type"] == etype)]
        hits = sub[sub["sig_fdr"]].sort_values("nl", ascending=False).head(12)
        lines += [f"## Strongest severity-trend hits — m4, {form_lab}\n",
                  f"{sub['sig_fdr'].sum()} of {len(sub)} features FDR-significant.\n",
                  "| feature | domain | effect | 95% CI | p | n |", "|---|---|---|---|---|---|"]
        for _, r in hits.iterrows():
            star = " *" if r["referral_item"] else ""
            lines.append(f"| {r['feature']}{star} | {r['domain']} | {r['effect']:.3f} | "
                         f"{r['ci_low']:.3f}–{r['ci_high']:.3f} | {r['p']:.1e} | {int(r['n']):,} |")
        lines.append("")

    lines += ["## Instrument scores, severe vs none\n",
              "| score | form | m1 | m2 | m3 | m4 |", "|---|---|---|---|---|---|"]
    for feat in COMPOSITES:
        for form in ["continuous", "binary"]:
            r = res[(res["feature"] == feat) & (res["form"] == form)
                    & (res["contrast"] == "severe_vs_none") & res["model"].isin(["m1", "m2", "m3", "m4"])]
            if not len(r):
                continue
            e = r.set_index("model")["effect"]
            lines.append(f"| {feat} | {form} | " + " | ".join(
                f"{e.get(m, float('nan')):.3f}" for m in ["m1", "m2", "m3", "m4"]) + " |")
    lines += [
        "\n## Caveats\n",
        "- **Ascertainment.** The reference group (None, AHI<5) is not the general population — "
        "it is symptomatic patients referred for a sleep study who turned out not to have OSA, so "
        "it is enriched for insomnia and non-apnoeic sleep complaints. This compresses (and for "
        "some items reverses) contrasts against None, the same caveat carried by the ICD, Lab, "
        "Med and Proc stages.",
        "- **Referral-driving items** (starred: " + ", ".join(sorted(U.REFERRAL_ITEMS)) + ") are the "
        "symptoms that get a patient sent for a sleep study in the first place; their large "
        "effects are partly the referral pathway rather than a dose-response to AHI.",
        "- **BMI is bracketed, not resolved**: m3 vs m4 spans the confounder-vs-mediator range.",
        "- **Complete-case per item**: item non-response is unknown, not zero. Fill rates run "
        "from ~59% to ~94% of the analytic sample; features below "
        f"{U.MIN_PATIENTS} answered patients (or {U.MIN_EVENTS} in the minority class for a binary "
        "item) were dropped before testing.",
        "- FOSQ 0 (\"difficulty for other reasons\") is treated as not-applicable per standard "
        "FOSQ scoring, so FOSQ items run 1–4 with **higher = better function** — a negative beta "
        "means worse function with increasing severity.",
        "- The strongest items underflow to p = 0 in double precision; plots clip -log10(p) at "
        "300, so points on that ceiling are ranked arbitrarily among themselves.",
    ]
    (U.ANALYSIS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    total = sum(1 for _ in U.PLOTS.rglob("*.png"))
    print(f"generated {n_png} plot calls; {total} PNGs under {U.PLOTS}")
    print("wrote tables ->", U.TABLES, "and", U.ANALYSIS / "SUMMARY.md")
    U.log_milestone(f"Step 03 plots/tables: {total} Manhattan/volcano/forest PNGs (OR + beta "
                    f"phenomes, domain-coloured) + adjustment/tophit/composite tables + SUMMARY.")


if __name__ == "__main__":
    main()
