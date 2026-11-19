# %% [markdown]
# MeasWAS Step 04 - figures, tables and SUMMARY.md for the cardiopulmonary MEASUREMENT arm.
#
# Reads analysis/results_5yr.csv (1,260 rows = 36 features x 7 contrasts x 5 models) and
# analysis/descriptives_5yr.csv (144 rows = 36 features x 4 severity strata).
#
# The outcome here is CONTINUOUS (rank-inverse-normal beta in SD units), not a rate ratio, so
# procwas/04_plots.py's log2(IRR) axis does not apply. The template is labwas/11 - same house
# look (volcano, Bonferroni dashed red + BH dotted navy, adjustText labels, tab-coloured
# categories), retargeted at a raw beta.
#
# ECHO AND PFT ARE NEVER POOLED IN A FIGURE. They are different arms with different coverage
# (echo ~14% of the cohort in-window, PFT ~3.6%) and different multiplicity denominators
# (22 vs 14); one shared panel would invite reading them as a single phenome.
#
# Outputs (all under analysis/):
#   plots/volcano_{echo,pft}_{severe_vs_none,trend}_{unadjusted,m4}_5yr.png   [8]
#   plots/forest_{echo,pft}_severe_vs_none_m4_5yr.png                        [2]  <- headline
#   plots/gradient_{echo,pft}_top_trend_5yr.png                              [2]  raw clinical units
#   plots/ladder_severe_vs_none_m1_m4_5yr.png                                [1]  m3->m4 = BMI step
#   tables/tophits_{trend,severe_vs_none}_m4_5yr.csv
#   tables/adjustment_ladder_m4fdr_5yr.csv
#   SUMMARY.md
# DE-IDENTIFIED: feature labels, counts and statistics only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import measwas_utils as U

W = U.WINDOW_LABEL
SEV_ORDER = ["None", "Mild", "Moderate", "Severe"]
CONTRASTS = ["severe_vs_none", "trend"]
MODELS = ["unadjusted", "m4"]
LADDER = ["m1", "m2", "m3", "m4"]

# Clinical concept GROUPS - a legible colour axis. measwas_utils' per-feature `concept` strings
# are too granular (34 distinct values) for a legend, so they are rolled up here.
GROUP = {
    # ---- echo ----
    "LVEF_pct": "Systolic function", "ef_value": "Systolic function",
    "mv_ea_ratio": "Diastolic function", "e_prime_septal_cms": "Diastolic function",
    "E_over_e_prime_septal": "Diastolic function", "mv_dec_time": "Diastolic function",
    "ivrt": "Diastolic function",
    "LV_mass_ASE_g": "LV mass / remodelling", "RWT": "LV mass / remodelling",
    "lv_diastolic_dimension": "LV chamber size", "lv_systolic_dimension": "LV chamber size",
    "lv_edv_index": "LV chamber size",
    "la_systolic_dimension": "LA size", "la_volume_index": "LA size",
    "TR_grad_mmHg": "Pulmonary hypertension / right heart",
    "estimated_rap": "Pulmonary hypertension / right heart",
    "rv_diastolic_dimension": "Pulmonary hypertension / right heart",
    "ao_root": "Aorta", "ao_ascending": "Aorta",
    "AV_Vmax_cms": "Aortic valve",
    "heart_rate_rest": "Haemodynamic state", "syst_bp_rest_supine": "Haemodynamic state",
    # ---- pft (spirometry) ----
    "FEV1_FVC_pre": "Airflow obstruction (ratio)", "FEV1_FVC_post": "Airflow obstruction (ratio)",
    "FEV1_PRE": "Expiratory flow (FEV1)", "FEV1_POST": "Expiratory flow (FEV1)",
    "FEV1_PRE_PREDICTED": "Expiratory flow (FEV1)", "FEV1_POST_PREDICTED": "Expiratory flow (FEV1)",
    "FVC_PRE": "Lung volume (FVC)", "FVC_POST": "Lung volume (FVC)",
    "PEF_POST": "Peak flow", "PEF_PRE_PREDICTED": "Peak flow", "PEF_POST_PREDICTED": "Peak flow",
    "FEF25_75_PRE": "Small-airway flow", "FEF25_75_POST": "Small-airway flow",
    "PEF_REF": "Reference value (anthropometric)",
}

# Clinical units, for the raw-scale severity-gradient figure.
UNITS = {
    "LVEF_pct": "%", "ef_value": "%", "lv_diastolic_dimension": "cm",
    "lv_systolic_dimension": "cm", "LV_mass_ASE_g": "g", "RWT": "ratio",
    "lv_edv_index": "mL/m2", "la_systolic_dimension": "cm", "la_volume_index": "mL/m2",
    "TR_grad_mmHg": "mmHg", "estimated_rap": "mmHg", "rv_diastolic_dimension": "cm",
    "mv_ea_ratio": "ratio", "e_prime_septal_cms": "cm/s", "E_over_e_prime_septal": "ratio",
    "mv_dec_time": "ms", "ivrt": "ms", "ao_root": "cm", "ao_ascending": "cm",
    "AV_Vmax_cms": "cm/s", "heart_rate_rest": "bpm", "syst_bp_rest_supine": "mmHg",
    "FEV1_FVC_pre": "%", "FEV1_FVC_post": "%", "FEV1_PRE": "L", "FEV1_POST": "L",
    "FVC_PRE": "L", "FVC_POST": "L", "PEF_POST": "L/s", "PEF_REF": "L/s",
    "PEF_PRE_PREDICTED": "fraction", "PEF_POST_PREDICTED": "fraction",
    "FEV1_PRE_PREDICTED": "fraction", "FEV1_POST_PREDICTED": "fraction",
    "FEF25_75_PRE": "L/s", "FEF25_75_POST": "L/s",
}

XLAB = {"beta": "beta (SD of rank-normal outcome)", "spearman_rho": "Spearman rho",
        "rank_biserial": "rank-biserial r"}
DOMAIN_LAB = {"echo": "ECHO", "pft": "PFT (EXPLORATORY, ~3.6% coverage)"}


def colours(features):
    cats = sorted({GROUP[f] for f in features})
    return {c: plt.cm.tab10(i % 10) for i, c in enumerate(cats)}


def _thresholds(d, ax, horizontal=True):
    """Bonferroni (dashed red) at 0.05/n_features and the realised BH boundary (dotted navy)."""
    line = ax.axhline if horizontal else ax.axvline
    line(-np.log10(0.05 / len(d)), ls="--", color="red", lw=0.8)
    if d["sig_fdr"].any():
        line(-np.log10(d.loc[d["sig_fdr"], "p"].max()), ls=":", color="navy", lw=0.8)


# %%
def volcano(d, title, path, cmap):
    d = d.dropna(subset=["p", "effect"]).copy()
    if len(d) < 2:
        return
    et = d["effect_type"].iloc[0]
    d["grp"] = d["feature"].map(GROUP)
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    for g, gg in d.groupby("grp"):
        ax.scatter(gg["effect"], gg["neglog10p"], s=40, color=cmap[g], label=g, alpha=0.85,
                   edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="gray", lw=0.7)
    _thresholds(d, ax)
    texts = [ax.text(r["effect"], r["neglog10p"], str(r["feature"])[:26], fontsize=6.5)
             for _, r in d.nlargest(15, "neglog10p").iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
    except Exception:
        pass
    ax.set_xlabel(XLAB.get(et, et)); ax.set_ylabel("-log10(p)"); ax.set_title(title, fontsize=10)
    ax.legend(ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def forest(d, title, path, cmap):
    """m4 severe_vs_none beta with 95% CI, sorted by effect. Filled = FDR-sig, hollow = not."""
    d = d.dropna(subset=["effect"]).sort_values("effect").reset_index(drop=True)
    if d.empty:
        return
    d["grp"] = d["feature"].map(GROUP)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(d) + 2.6))
    ax.axvline(0, color="gray", lw=0.9, zorder=0)
    for i, r in d.iterrows():
        c = cmap[r["grp"]]
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=c, lw=1.9, solid_capstyle="round")
        ax.scatter([r["effect"]], [i], s=58, color=c if r["sig_fdr"] else "white",
                   edgecolor=c, linewidth=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['feature']}" for _, r in d.iterrows()], fontsize=8)
    xr = float(np.nanmax(d["ci_high"]))
    pad = 0.10 * (float(np.nanmax(d["ci_high"])) - float(np.nanmin(d["ci_low"])) + 1e-9)
    for i, r in d.iterrows():
        ax.text(xr + pad * 0.35, i, f"n={r['n']:,}", fontsize=6.5, va="center", color="0.35")
    ax.set_xlim(float(np.nanmin(d["ci_low"])) - pad, xr + pad * 2.2)
    ax.set_ylim(-1, len(d))
    ax.set_xlabel("beta, severe vs none (SD of rank-normal outcome), model m4")
    ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[g], label=g)
               for g in sorted(d["grp"].unique())]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", color="0.3",
                           label="not FDR-significant")]
    ax.legend(handles=handles, ncol=3, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.055 - 1.2 / len(d)))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def gradient(dsc, feats, title, path, ncol=3):
    """Mean raw value +/- 95% CI across None/Mild/Moderate/Severe, in clinical units."""
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.7 * nrow), squeeze=False)
    x = np.arange(4)
    for k, f in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        g = dsc[dsc.feature == f].set_index("severity").reindex(SEV_ORDER)
        hw = 1.96 * g["sd"] / np.sqrt(g["n"])
        ax.errorbar(x, g["mean"], yerr=hw, marker="o", ms=5, lw=1.6, capsize=3, color="#1f4e79")
        ax.set_xticks(x); ax.set_xticklabels(SEV_ORDER, fontsize=7.5, rotation=20)
        ax.set_title(f"{f}  [{UNITS.get(f, '')}]\nn={int(g['n'].sum()):,}", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    for k in range(len(feats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(path, dpi=130); plt.close(fig)


def _spread(ys, gap):
    """Push a set of label y-positions apart by at least `gap`, preserving their order."""
    order = np.argsort(-np.asarray(ys))
    out = np.array(ys, dtype=float)
    prev = None
    for i in order:
        if prev is not None and prev - out[i] < gap:
            out[i] = prev - gap
        prev = out[i]
    return out


def ladder_plot(lad, path, cmaps):
    """beta across m1->m2->m3->m4 for the features FDR-significant at m4. The m3->m4 step is
    the BMI adjustment; BMI is plausibly a MEDIATOR of OSA -> cardiac remodelling, so the
    attenuation there is a substantive result, not a nuisance correction."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0), sharey=False)
    for ax, dom in zip(axes, ["echo", "pft"]):
        sub = lad[lad.domain == dom]
        cmap = cmaps[dom]
        ends = []
        for f, g in sub.groupby("feature"):
            g = g.set_index("model").reindex(LADDER)
            c = cmap[GROUP[f]]
            ax.plot(range(4), g["effect"], marker="o", ms=4, lw=1.5, color=c, alpha=0.9)
            ends.append((float(g["effect"].iloc[-1]), f, c))
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvspan(2, 3, color="orange", alpha=0.10)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["m1\ncrude", "m2\n+demog/year", "m3\n+smoke/income", "m4\n+BMI"],
                           fontsize=8)
        ax.set_xlim(-0.15, 4.6)
        ax.set_ylabel("beta, severe vs none (SD units)")
        ax.set_title(f"{DOMAIN_LAB[dom]} - m4 FDR-significant features (n={sub['feature'].nunique()})\n"
                     f"shaded band = the BMI step (m3 -> m4)", fontsize=9.5)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        lo, hi = ax.get_ylim()
        for (y, f, c), ly in zip(ends, _spread([e[0] for e in ends], 0.032 * (hi - lo))):
            ax.annotate(f[:26], xy=(3.0, y), xytext=(3.35, ly), fontsize=7, va="center", color=c,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.6))
    fig.suptitle("MeasWAS adjustment ladder | severe vs none | 5yr pre-index window", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=130); plt.close(fig)


# %%
def summary_md(res, dsc, agree, lad, n_png) -> str:
    def top(dom, con, k=8):
        s = res.query("domain==@dom and contrast==@con and model=='m4'").sort_values("p").head(k)
        return "\n".join(
            f"| {r['feature']} | {r['concept'][:44]} | {r['n']:,} | {r['effect']:+.4f} "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p']:.2e} | "
            f"{'FDR+BON' if r['sig_bon'] else ('FDR' if r['sig_fdr'] else '-')} |"
            for _, r in s.iterrows())

    def nsig(dom, con, model="m4"):
        s = res.query("domain==@dom and contrast==@con and model==@model")
        return int(s["sig_fdr"].sum()), int(s["sig_bon"].sum()), len(s)

    a = agree.iloc[0]
    hdr = "| feature | concept | n | beta [95% CI] | p | sig |\n|---|---|---|---|---|---|"
    L = []
    A = L.append

    A("# MeasWAS - the cardiopulmonary MEASUREMENT arm of ProcWAS\n")
    A("**What this is.** ProcWAS proper asks how OFTEN a procedure happens (Poisson "
      "procedures/yr over the full cohort, non-performers = a structural zero). This arm asks a "
      "different question: **what the procedure MEASURED.** Echocardiogram parameters "
      "(`echo.csv` + `ef.csv`) and spirometry values (`pft.csv`) are regressed on OSA severity "
      "over the same 222,109-patient v2 cohort spine and a single 5-year pre-index window "
      "`[index - 5yr, index)`. Outcomes are rank-inverse-normal transformed per feature, so every "
      "`beta` below is in **SD units of the rank-normal outcome** - it is NOT a ratio and must "
      "not be exponentiated.\n")

    A("## Read these four caveats before any number\n")
    A("**1. The PFT arm is EXPLORATORY.** Only ~3.6% of the cohort has a PFT in the 5yr window "
      "(7,875 patients retained; tested features n = 5,152-7,874). `pft.csv` ends 2019-03 while "
      "cohort index dates run to 2025-03, so late-index patients are structurally unable to "
      "contribute, and coverage is confounded with index year (13.2% for 2016 index dates vs 4.7% "
      "for 2024). `C(study_year)` is in every adjusted model for exactly this reason, but "
      "adjustment is not the same as coverage. Treat PFT findings as hypothesis-generating.\n")
    A("**2. This is a COMPLETE-CASE analysis among the measured.** A patient contributes to a "
      "feature only if they have a non-null gated value for it in the window. Zero-filling "
      "(the ProcWAS reflex) would be catastrophic here - a patient with no echo has an UNKNOWN "
      "LVEF, not an LVEF of 0. So **every estimate is conditional on having been tested**, and "
      "testing is not severity-neutral. Echo rates by severity are None 12.6%, Mild 14.4%, "
      "Moderate 14.9%, Severe 13.3% - **non-monotonic**, peaking at Moderate and falling back at "
      "Severe, echoing this project's inverse-utilization finding (severe OSA has ~18% FEWER "
      "encounters). The selection into the analysed sample is therefore not a simple monotone "
      "function of the exposure, and no result here can be read as a population-marginal effect.\n")
    A("**3. Two independent LVEF sources, deliberately not pooled.** `echo.csv` supplies "
      "`LVEF_pct` (coalesced calculated/estimated LVEF) and `ef.csv` supplies `ef_value`, a "
      "separate reading stream with far better coverage (211,503 patients, 1995-2025; n=46,223 "
      "in-window vs 22,457 for the echo-side value). They are tested as two features. Agreement "
      f"on the {int(a['n_patients_both']):,} patients with both: Pearson r = {a['pearson_r']:.4f}, "
      f"Spearman rho = {a['spearman_rho']:.3f}, mean difference (ef - echo) = "
      f"{a['mean_diff_ef_minus_echo']:+.2f} points (SD {a['sd_diff']:.2f}), median |difference| = "
      f"{a['median_abs_diff']:.2f}, 95% limits of agreement [{a['loa_lower']:.2f}, "
      f"{a['loa_upper']:.2f}], {a['pct_within_5_points']:.1f}% within 5 points. Correlated but not "
      "interchangeable: the ~2.8-point offset and +/-13-point limits of agreement mean a "
      "single-source LVEF conclusion should be checked against the other. Both sources give the "
      "same direction and both are FDR+Bonferroni significant, which is the useful part.\n")
    A("**4. Multiplicity is applied WITHIN domain.** FDR (Benjamini-Hochberg) and Bonferroni are "
      "computed per `(contrast, model)` separately inside `echo` (denominator 22) and `pft` "
      "(denominator 14). Echo and PFT are never pooled in a figure or a denominator - different "
      "arms, different coverage, different power.\n")

    A("## Design\n")
    A("- **Cohort**: 222,109-patient ICD-v2 spine; window `[index - 5yr, index)`; one value per "
      "patient per feature (mean of that patient's in-window studies, after unit repair + gating).")
    A(f"- **Phenome**: {res.query('domain==\"echo\"')['feature'].nunique()} echo features + "
      f"{res.query('domain==\"pft\"')['feature'].nunique()} spirometry features = "
      f"{res['feature'].nunique()}. Floor: >= {U.MIN_PATIENTS} complete cases and >= 20 per "
      "severity stratum.")
    A("- **Contrasts (7)**: omnibus, trend, mild/moderate/severe-vs-none, ahi_ge5, ahi_ge15.")
    A("- **Models (5)**: `unadjusted` nonparametric screen (Kruskal-Wallis epsilon^2 / Spearman "
      "trend / Mann-Whitney rank-biserial), then nested OLS `m1` crude, `m2` +age/sex/race/"
      "`C(study_year)`/obs_window, `m3` +smoking+income, `m4` +BMI.")
    A("- **`C(study_year)` is mandatory**, not optional: echo parameter availability drifts hard "
      "across the study era (e.g. `la_volume_index` 0.0% -> 25.6% -> 46.7% of echoes at "
      "2010/2015/2020) and a patient's in-window studies are mechanically tied to their index year.")
    A("- **BMI is a MEDIATOR candidate, not just a confounder.** OSA -> obesity -> LV remodelling "
      "is a live causal path, so `m4` (which conditions on BMI) is plausibly an OVER-adjusted "
      "estimate for the remodelling features. `m3` vs `m4` is reported side by side in the "
      "adjustment ladder for precisely this reason; do not read `m4` as automatically 'the best "
      "model'.")
    A("- `syst_bp_rest_supine` is a **positive control** (the OSA-hypertension link is textbook) "
      "and is deliberately kept OUT of the covariate ladder for every other feature, because it "
      "sits on the causal path from OSA to cardiac remodelling.\n")

    A("## Headline figure\n")
    A("`plots/forest_echo_severe_vs_none_m4_5yr.png` - all 22 echo features, m4 severe-vs-none "
      "beta with 95% CI, sorted by effect, coloured by clinical concept, filled marker = "
      "FDR-significant, per-feature n annotated at the right. `forest_pft_...` is the PFT "
      "counterpart. Supporting: `volcano_{echo,pft}_{severe_vs_none,trend}_{unadjusted,m4}_5yr.png` "
      "(8), `gradient_{echo,pft}_top_trend_5yr.png` (raw clinical units by severity stratum), "
      "`ladder_severe_vs_none_m1_m4_5yr.png`.\n")

    A("## Headline - ECHO\n")
    f_s, b_s, t_s = nsig("echo", "severe_vs_none"); f_t, b_t, _ = nsig("echo", "trend")
    A(f"m4 severe-vs-none: **{f_s}/{t_s} FDR-significant** ({b_s} also Bonferroni). "
      f"m4 trend: **{f_t}/{t_s} FDR-significant** ({b_t} Bonferroni).\n")
    A("Top 8 by m4 severe-vs-none p:\n")
    A(hdr); A(top("echo", "severe_vs_none"))
    A("")
    A("The signal is coherent cardiac remodelling, not noise: LV mass up (`LV_mass_ASE_g` "
      "beta=+0.210, p=6.0e-37), concentric geometry up (`RWT`), septal e' down and E/e' up "
      "(impaired relaxation with higher filling pressure), LVEF down in BOTH independent sources, "
      "aortic root and ascending aorta larger, resting heart rate higher. The positive control "
      "fires as required (`syst_bp_rest_supine` beta=+0.191, p=1.2e-10). Pulmonary hypertension "
      "(`TR_grad_mmHg`, beta=+0.074, p=4.9e-04) is significant but an order of magnitude weaker "
      "than the LV-mass signal - the a-priori headline feature is not the strongest hit. Nulls: "
      "`lv_edv_index`, `AV_Vmax_cms`, `rv_diastolic_dimension`, `la_volume_index`, `ivrt` (the "
      "designated first cut, n=3,577).\n")

    A("## Headline - PFT (EXPLORATORY)\n")
    f_s, b_s, t_s = nsig("pft", "severe_vs_none"); f_t, b_t, _ = nsig("pft", "trend")
    A(f"m4 severe-vs-none: **{f_s}/{t_s} FDR-significant** ({b_s} Bonferroni). "
      f"m4 trend: **{f_t}/{t_s} FDR-significant** ({b_t} Bonferroni). EXPLORATORY - see caveat 1.\n")
    A(hdr); A(top("pft", "severe_vs_none", 6))
    A("")
    A("The direction is the surprise and deserves scepticism: `FEV1_FVC_pre` is HIGHER with "
      "increasing OSA severity (beta=+0.191, p=1.7e-09), as is `FEF25_75_PRE`. A higher FEV1/FVC "
      "ratio is *less* obstruction. The physiologic reading is restriction rather than "
      "obstruction - obesity-related restrictive mechanics compress FVC more than FEV1 and so "
      "RAISE the ratio - which is consistent with `FVC_POST` trending negative. But the alternative "
      "reading is referral selection: who gets a PFT at all differs by severity, coverage is 3.6%, "
      "and the file ends six years before the last index date. `PEF_REF` is a predicted value "
      "computed from age/sex/height, not a measurement of the patient's lungs; its significance "
      "(p=1.9e-03) is an anthropometric artefact and should be read as a negative control that "
      "partially fails.\n")
    A("**The PFT signal is CREATED BY ADJUSTMENT, and that is the single biggest reason to "
      "distrust it.** In the unadjusted nonparametric screen and in crude m1, the ratio features "
      "are flat: `FEV1_FVC_pre` m1 beta = +0.011 (p=0.74), `FEV1_FVC_post` m1 beta = -0.020 "
      "(p=0.62). Adding age/sex/race/`C(study_year)`/obs_window at m2 moves them to +0.224 and "
      "+0.195, where they stay. The entire effect is the m1 -> m2 step. That is *consistent* with "
      "genuine confounding by age and by the index-year coverage gradient - severe-OSA patients "
      "with a PFT are systematically different in age and era - but a result that exists only "
      "after conditioning on the same variables that drive who is in the sample is exactly the "
      "shape a selection artefact takes. The two features whose crude effect is real and then "
      "SHRINKS under adjustment (`PEF_REF` m1 +0.246 -> m4 +0.047; `PEF_POST` m1 +0.184 -> m4 "
      "+0.085) are the anthropometric ones, which is the expected pattern. Do not report any PFT "
      "number without the m1 column beside it.\n")
    A("`plots/gradient_pft_top_trend_5yr.png` makes this visible on the raw scale: the PFT means "
      "are nearly flat across severity and in places non-monotonic (`FEF25_75_PRE` FALLS from "
      "None to Mild before rising; `FEF25_75_POST` ends lower at Severe than at None), and the "
      "total range across all four strata is well under one within-stratum SD. Contrast "
      "`plots/gradient_echo_top_trend_5yr.png`, where the same figure shows clean monotone "
      "staircases with non-overlapping 95% CIs (LV mass 163 -> 177 -> 192 -> 209 g; LVEF 62.7 -> "
      "61.8 -> 60.8 -> 59.7%; septal e' 8.32 -> 7.66 -> 7.39 -> 7.16 cm/s). The echo arm does not "
      "need adjustment to show its gradient; the PFT arm does.\n")

    A("## Adjustment ladder (m1 -> m4)\n")
    A(f"`tables/adjustment_ladder_m4fdr_5yr.csv` + `plots/ladder_severe_vs_none_m1_m4_5yr.png` "
      f"trace beta across the four nested models for the "
      f"{lad['feature'].nunique()} features FDR-significant at m4 (severe-vs-none). The BMI step "
      "(m3 -> m4) is shaded. Median attenuation from m3 to m4 is "
      f"{100 * lad.attrs['med_bmi_atten']:.1f}% of the m3 beta (echo "
      f"{100 * lad.attrs['med_bmi_atten_echo']:.1f}%, pft "
      f"{100 * lad.attrs['med_bmi_atten_pft']:.1f}%; the percentage is left blank for the "
      f"{lad.attrs['n_signflip']} feature(s) whose beta flips sign or sits within noise of zero at "
      "m3, where a ratio is an artefact rather than an attenuation). If BMI is a mediator of OSA -> LV "
      "remodelling, that attenuation is part of the effect being removed, not confounding being "
      "controlled; m3 is then the more interpretable total-effect estimate.\n")
    A("The BMI step is not a rounding correction. `LV_mass_ASE_g` goes m3 +0.523 -> m4 +0.210 "
      "(60% of the m3 beta), `la_systolic_dimension` +0.361 -> +0.102 (72%), "
      "`lv_diastolic_dimension` +0.209 -> +0.046 (78%), `RWT` +0.229 -> +0.122 (47%). Chamber-size "
      "features attenuate most, which is what a body-size mediator should do. The remaining m4 "
      "effect is what survives conditioning on BMI, i.e. a lower bound on the OSA-remodelling "
      "association if BMI is on the causal path; the m3 estimate is the upper bound if BMI is a "
      "pure confounder. Both are in the ladder table - pick deliberately.\n")

    A("## Data provenance - the echo.csv misdiagnosis\n")
    A("The project data dictionary (2026-06-27) records `echo.csv` as **structurally corrupted and "
      "unusable**. That verdict is **wrong** and this module supersedes it. `echo.csv` is a "
      "non-RFC-4180 dialect: the exporter quotes a field only when it contains a comma or a quote, "
      "never when it contains a newline, so the RECORD terminator is a bare LF while newlines "
      "embedded in free-text report bodies are CRLF (2.4M of them) or a lone CR (74). Splitting on "
      "bare LF, re-joining the 28 segments that do not start with the `MRN,procedure_id` signature, "
      "and neutralising the embedded newlines recovers **284,031 / 284,031 records at 228 fields "
      "each - 100%**. Encoding is latin-1/cp1252.\n")
    A("The reader of record is `echo_records()` / `echo_fields()` in "
      "`analyses/procwas/measwas/measwas_utils.py`. Two things must never be done to this file:\n")
    A("- **Never** `pandas.read_csv` the raw file - it silently returns 926,702 NaN-padded junk "
      "rows under every engine / `on_bad_lines` setting, with no warning.")
    A("- **Never** `duckdb read_csv(..., ignore_errors=true)` - the 61.6% it keeps is confounded by "
      "facility (46-80% retention varies by site) and by year.\n")
    A("PHI and free-text report bodies (`U.ECHO_PHI_COLS`) are dropped at read time and are never "
      "materialised to disk. Every output in this module is a count, a percentage or a "
      "distribution.\n")

    A("## Files\n")
    A(f"- `results_{W}.csv` - {len(res):,} rows = 36 features x 7 contrasts x 5 models")
    A(f"- `descriptives_{W}.csv` - {len(dsc)} rows = 36 features x 4 severity strata, RAW "
      "clinical units")
    A("- `ef_source_agreement.csv`, `echo_feature_completeness.csv`, `echo_column_completeness.csv`")
    A(f"- `plots/` - {n_png} PNGs; `tables/tophits_{{trend,severe_vs_none}}_m4_{W}.csv`, "
      f"`tables/adjustment_ladder_m4fdr_{W}.csv`")
    return "\n".join(L) + "\n"


# %%
def main() -> None:
    U.ensure_dirs()
    res = pd.read_csv(U.ANALYSIS / f"results_{W}.csv")
    # 'None' is a severity LEVEL here, not a missing value.
    dsc = pd.read_csv(U.ANALYSIS / f"descriptives_{W}.csv", keep_default_na=False, na_values=[""])
    agree = pd.read_csv(U.ANALYSIS / "ef_source_agreement.csv")
    print(f"results {res.shape} | descriptives {dsc.shape}", flush=True)

    unmapped = sorted(set(res["feature"]) - set(GROUP))
    if unmapped:
        raise SystemExit(f"features missing from GROUP: {unmapped}")

    n_png = 0
    cmaps = {d: colours(sorted(res.loc[res.domain == d, "feature"].unique()))
             for d in ("echo", "pft")}
    for dom in ["echo", "pft"]:
        base = res[res.domain == dom]
        cmap = cmaps[dom]

        # ---- 1. volcanoes ----
        for con in CONTRASTS:
            for mod in MODELS:
                sub = base[(base.contrast == con) & (base.model == mod)]
                volcano(sub, f"MeasWAS {DOMAIN_LAB[dom]} | {W} | {con} | {mod}",
                        U.PLOTS / f"volcano_{dom}_{con}_{mod}_{W}.png", cmap); n_png += 1

        # ---- 2. forest (headline) ----
        sub = base[(base.contrast == "severe_vs_none") & (base.model == "m4")]
        forest(sub, f"MeasWAS {DOMAIN_LAB[dom]} | severe vs none | m4 | {W} pre-index window\n"
                    "filled = FDR-significant within domain; bars = 95% CI",
               U.PLOTS / f"forest_{dom}_severe_vs_none_m4_{W}.png", cmap); n_png += 1

        # ---- 3. severity gradient in RAW clinical units, top features by m4 trend p ----
        k = 9 if dom == "echo" else 6
        feats = (base[(base.contrast == "trend") & (base.model == "m4")]
                 .nsmallest(k, "p")["feature"].tolist())
        gradient(dsc[dsc.domain == dom], feats,
                 f"MeasWAS {DOMAIN_LAB[dom]} | top {k} by m4 trend p | mean raw value +/- 95% CI "
                 f"by OSA severity | {W}",
                 U.PLOTS / f"gradient_{dom}_top_trend_{W}.png"); n_png += 1

    # ---- 4a. top-hit tables ----
    keep = ["domain", "feature", "concept", "n", "effect", "se", "ci_low", "ci_high", "p",
            "neglog10p", "sig_fdr", "sig_bon"]
    for con in CONTRASTS:
        (res[(res.contrast == con) & (res.model == "m4")].sort_values("p")[keep]
         .to_csv(U.TABLES / f"tophits_{con}_m4_{W}.csv", index=False))

    # ---- 4b. adjustment ladder for the m4-FDR-significant features ----
    sv = res[res.contrast == "severe_vs_none"]
    sig_feats = set(sv[(sv.model == "m4") & sv.sig_fdr]["feature"])
    lad = sv[sv.feature.isin(sig_feats) & sv.model.isin(LADDER)].copy()
    wide = lad.pivot_table(index=["domain", "feature", "concept", "n"], columns="model",
                           values=["effect", "p"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["delta_m3_to_m4"] = wide["effect_m4"] - wide["effect_m3"]

    def _atten(a, b):
        """% of `a` removed by moving to `b`. Undefined (NaN) when the sign flips or `a` is
        within noise of 0 - a ratio there is an artefact, not an attenuation."""
        r = 100 * (1 - b / a)
        return r.where((np.sign(a) == np.sign(b)) & (a.abs() > 0.02))

    wide["pct_attenuation_bmi"] = _atten(wide["effect_m3"], wide["effect_m4"])
    wide["pct_attenuation_m1_to_m4"] = _atten(wide["effect_m1"], wide["effect_m4"])
    wide = wide.sort_values("p_m4")[
        ["domain", "feature", "concept", "n"]
        + [f"effect_{m}" for m in LADDER] + [f"p_{m}" for m in LADDER]
        + ["delta_m3_to_m4", "pct_attenuation_bmi", "pct_attenuation_m1_to_m4"]]
    wide.to_csv(U.TABLES / f"adjustment_ladder_m4fdr_{W}.csv", index=False)

    lad.attrs["med_bmi_atten"] = float(wide["pct_attenuation_bmi"].median() / 100)
    lad.attrs["n_signflip"] = int(wide["pct_attenuation_bmi"].isna().sum())
    for d in ("echo", "pft"):
        lad.attrs[f"med_bmi_atten_{d}"] = float(
            wide.loc[wide.domain == d, "pct_attenuation_bmi"].median() / 100)
    ladder_plot(lad, U.PLOTS / f"ladder_severe_vs_none_m1_m4_{W}.png", cmaps); n_png += 1

    # ---- 5. SUMMARY.md ----
    (U.ANALYSIS / "SUMMARY.md").write_text(summary_md(res, dsc, agree, lad, n_png),
                                           encoding="utf-8")

    # ---------------- de-identified console report ----------------
    print(f"\n{n_png} PNGs -> {U.PLOTS}")
    print(f"tables -> {U.TABLES}; SUMMARY.md -> {U.ANALYSIS / 'SUMMARY.md'}")
    print("\nFDR-significant (m4) by domain x contrast:")
    print(res[res.model == "m4"].pivot_table(index="contrast", columns="domain",
                                             values="sig_fdr", aggfunc="sum",
                                             fill_value=0).to_string())
    print(f"\nadjustment ladder: {wide['feature'].nunique()} m4-FDR-sig features "
          f"(echo {int((wide.domain == 'echo').sum())}, pft {int((wide.domain == 'pft').sum())}); "
          f"median BMI-step attenuation {100 * lad.attrs['med_bmi_atten']:.1f}%")
    print("\nlargest m3 -> m4 (BMI) attenuation:")
    for _, r in wide.nlargest(6, "pct_attenuation_bmi").iterrows():
        print(f"    {r['domain']:4s} {str(r['feature'])[:26]:26s} m3={r['effect_m3']:+.4f} -> "
              f"m4={r['effect_m4']:+.4f}  ({r['pct_attenuation_bmi']:+.1f}%)")

    U.log_milestone(f"Step 04 MeasWAS plots/tables: {n_png} PNGs (volcano/forest/gradient/ladder, "
                    "echo + pft kept separate), tophit + adjustment-ladder tables, SUMMARY.md.")


if __name__ == "__main__":
    main()
