"""Stage 4 -- assemble the fit-index tables into plots and one decision document.

This is the PAUSE point. It does not finalise anything: it produces, per domain, the
model-fit table, the enumeration plots, the class profiles at each candidate K and the
AHI validation, then states which K the pre-declared rule selects and why. Choosing K
is the user's call; ``05_finalize.py`` takes those choices and writes the per-patient
assignments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import viz  # noqa: E402
from common.config import RESULTS, SELECTION  # noqa: E402

viz.style()

DOMAINS = {
    "c1_physiologic": ("latent profile analysis", "c1"),
    "c2_symptomatic": ("mixed-mode latent class cluster analysis", "c2"),
    "c3_comorbidities": ("latent class analysis", "c3"),
}
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def enumeration_plot(fit: pd.DataFrame, domain: str, prefix: str) -> Path:
    """BIC/CAIC/sABIC curves, entropy, smallest class and split-half ARI vs K."""
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for col, style in [("bic", "-o"), ("caic", "--s"), ("sabic", ":^")]:
        if col in fit:
            ax[0].plot(fit.K, fit[col], style, label=col.upper(), ms=4)
    elbow = int(fit.bic_elbow_K.iloc[0])
    ax[0].axvline(elbow, color="crimson", lw=1, ls="--", label=f"elbow K={elbow}")
    ax[0].set_xlabel("K"); ax[0].set_ylabel("information criterion")
    ax[0].set_title("Information criteria"); ax[0].legend(fontsize=8)

    ax[1].plot(fit.K, fit.rel_entropy, "-o", ms=4)
    ax[1].axhline(SELECTION["min_relative_entropy"], color="crimson", ls="--", lw=1)
    ax[1].set_xlabel("K"); ax[1].set_ylabel("relative entropy")
    ax[1].set_title("Classification certainty"); ax[1].set_ylim(0, 1.02)

    ax[2].plot(fit.K, fit.min_class_frac, "-o", ms=4)
    ax[2].axhline(SELECTION["min_class_frac"], color="crimson", ls="--", lw=1)
    ax[2].set_xlabel("K"); ax[2].set_ylabel("smallest class (fraction)")
    ax[2].set_title("Smallest class"); ax[2].set_yscale("log")

    if "split_half_ari" in fit and fit.split_half_ari.notna().any():
        ax[3].plot(fit.K, fit.split_half_ari, "-o", ms=4)
        ax[3].axhline(SELECTION["min_split_half_ari"], color="crimson", ls="--", lw=1)
    ax[3].set_xlabel("K"); ax[3].set_ylabel("split-half ARI")
    ax[3].set_title("Replication across halves"); ax[3].set_ylim(0, 1.02)

    fig.suptitle(f"{domain} — class enumeration", fontweight="bold")
    fig.tight_layout()
    p = PLOTS / f"{prefix}_enumeration.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def profile_plot(prof: pd.DataFrame, K: int, domain: str, prefix: str,
                 value="std_diff", top=30) -> Path | None:
    sub = prof[prof.K == K]
    if sub.empty:
        return None

    # Row labels: prefer the clinical name over the bare code where one exists.
    idx = "indicator"
    if "phenotype" in sub.columns and sub.phenotype.notna().any():
        sub = sub.assign(_row=[
            f"{str(p)[:44]}  [{i}]" if isinstance(p, str) else str(i)
            for p, i in zip(sub.phenotype, sub.indicator)])
        idx = "_row"
    piv = sub.pivot_table(index=idx, columns="class", values=value)

    # A ratio's neutral point is 1, not 0. Rendering prevalence_ratio on a diverging
    # map centred at zero paints "no difference from the cohort" as a strong colour
    # and makes every class look elevated. Log2 fold-change is symmetric about 0, so
    # the midpoint means what a diverging map needs it to mean: no difference.
    if value == "prevalence_ratio":
        piv = np.log2(piv.replace(0, np.nan))
        cbar_label = "log2 prevalence ratio vs cohort\n(0 = same as cohort)"
        note = "log2 prevalence ratio"
    else:
        cbar_label = "standardised difference from the cohort mean (SD)"
        note = value

    order = piv.abs().max(axis=1).sort_values(ascending=False).head(top).index
    piv = piv.loc[order]

    # Final class names if the solution has been finalised; otherwise the fitting
    # index, which is all that exists at this stage of the pipeline.
    names, cols = [f"class {k}" for k in range(K)], list(piv.columns)
    fin = (RESULTS / "FINAL_lca.json")
    lab = (RESULTS / "CLASS_LABELS.json")
    if fin.exists() and lab.exists():
        F = json.loads(fin.read_text())["domains"].get(domain, {})
        L = json.loads(lab.read_text()).get(domain, {})
        if F.get("K") == K and L.get("K") == K:
            cols = [int(F["final_to_fit_index"][str(k)]) for k in range(K)]
            names = [L.get(str(k), f"class {k}") for k in range(K)]
    piv = piv[cols]

    fig, ax = plt.subplots(figsize=(3.4 + 1.25 * K, 0.30 * len(piv) + 1.9))
    lim = float(np.nanmax(np.abs(piv.to_numpy())))
    im = ax.imshow(piv.to_numpy(), cmap=viz.DIV, aspect="auto", vmin=-lim, vmax=lim)
    ax.set_xticks(range(K))
    ax.set_xticklabels([n.replace("-", "-\n") for n in names], fontsize=7.5)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_title(f"{domain} — K={K} class profile\n({note}, top {len(piv)} indicators)",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label(cbar_label, fontsize=8)
    cb.outline.set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    p = PLOTS / f"{prefix}_profile_K{K}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


lines = ["# Latent class / latent profile analysis — model selection",
         "",
         "Generated by `analysis/04_report.py`. **This is a decision document, not a "
         "final result.** Pick one K per domain, then run `05_finalize.py`.",
         "",
         "Selection rule (pre-declared in `common/config.py`, fixed before any K>1 "
         "was fitted): recommended K = the largest K at or below the BIC elbow that "
         f"also has relative entropy >= {SELECTION['min_relative_entropy']}, smallest "
         f"class >= {SELECTION['min_class_frac']:.0%} and >= "
         f"{SELECTION['min_class_n']} patients, a log-likelihood replicated by >= 2 "
         f"independent starts, split-half ARI >= {SELECTION['min_split_half_ari']}, "
         "and no degenerate (variance-collapsed) component. If nothing qualifies, no "
         "K is recommended — that is a real answer, not a failure.",
         ""]

summary = {}
for domain, (model_name, prefix) in DOMAINS.items():
    d = RESULTS / domain
    fit_path = d / f"{prefix}_fit_indices.csv"
    if not fit_path.exists():
        lines += [f"## {domain}", "", "_not yet fitted_", ""]
        continue
    fit = pd.read_csv(fit_path)
    ep = enumeration_plot(fit, domain, prefix)

    rec = fit.loc[fit.recommended, "K"]
    rec_k = int(rec.iloc[0]) if len(rec) else None
    elbow = int(fit.bic_elbow_K.iloc[0])
    summary[domain] = {"recommended_K": rec_k, "bic_elbow_K": elbow}

    lines += [f"## {domain} — {model_name}", ""]
    show = ["K", "loglik", "n_params", "bic", "bic_drop_rel_to_first", "sabic",
            "rel_entropy", "min_class_frac", "min_class_n", "min_component_sd",
            "degenerate", "split_half_ari", "ll_replicated", "recommended"]
    lines += [fit[[c for c in show if c in fit.columns]].to_markdown(index=False), ""]
    fired = bool(fit.elbow_ceiling_fired.iloc[0]) if "elbow_ceiling_fired" in fit else None
    maxk = int(fit.max_non_degenerate_K.iloc[0]) if "max_non_degenerate_K" in fit else None
    ndeg = int(fit.degenerate.sum()) if "degenerate" in fit else 0
    lines += [f"- BIC elbow: **K = {elbow}**"
              + ("" if fired is None else
                 ("" if fired else
                  f" — ⚠ the elbow never fired within the admissible range (K ≤ {maxk}); "
                  "the BIC curve was still paying for extra classes at the top of it, so "
                  "the ceiling is not binding and the remaining criteria are doing the work")),
              (f"- **{ndeg} of {len(fit)} fits were degenerate** (a component collapsed "
               f"to near-zero variance), so K > {maxk} is not estimable under this "
               "measurement model and is excluded from the elbow search — an unbounded "
               "likelihood has no comparable BIC." if ndeg else
               "- No fit was degenerate."),
              f"- Rule selects: **K = {rec_k}**" if rec_k
              else "- Rule selects: **no admissible K > 1**", ""]

    prof_path = d / f"{prefix}_class_profiles.csv"
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        val = "prevalence_ratio" if "prevalence_ratio" in prof.columns else "std_diff"
        cand = sorted({k for k in (rec_k, elbow, 2, 3) if k and k in set(prof.K)})
        for K in cand:
            pp = profile_plot(prof, K, domain, prefix, value=val)
            if pp:
                lines.append(f"![{domain} K={K}](plots/{pp.name})")
        lines.append("")

    ahi_path = d / f"{prefix}_ahi_validation.csv"
    if ahi_path.exists():
        ahi = pd.read_csv(ahi_path)
        lines += ["### AHI (external — never a model input; 3-step BCH)", ""]
        for K in sorted({k for k in (rec_k, elbow) if k}):
            sub = ahi[ahi.K == K]
            if len(sub):
                lines += [f"**K = {K}**", "",
                          sub.to_markdown(index=False), ""]

    for extra, title in [(f"{prefix}_sensitivity.csv", "Sensitivity arms (ARI vs primary)"),
                         (f"{prefix}_form_artifact_check.csv", "Intake-form artifact check"),
                         (f"{prefix}_device_diagnostics.csv", "Device / era composition"),
                         (f"{prefix}_utilisation_validation.csv",
                          "Healthcare-contact index by class")]:
        p = d / extra
        if p.exists():
            t = pd.read_csv(p)
            if "K" in t.columns and rec_k:
                t = t[t.K.isin([k for k in (rec_k, elbow) if k])]
            lines += [f"### {title}", "", t.head(24).to_markdown(index=False), ""]
    lines.append(f"![{domain} enumeration](plots/{ep.name})")
    lines.append("")

lines += ["## Next step", "",
          "Choose a K per domain and record it in `results/K_CHOICES.json`, e.g.",
          "", "```json",
          json.dumps({k: v["recommended_K"] for k, v in summary.items()}, indent=2),
          "```", "", "then run `python analysis/05_finalize.py`."]

(RESULTS / "MODEL_SELECTION.md").write_text("\n".join(lines), encoding="utf-8")
(RESULTS / "K_RECOMMENDED.json").write_text(json.dumps(summary, indent=2))
print(f"wrote {RESULTS / 'MODEL_SELECTION.md'}")
for k, v in summary.items():
    print(f"  {k:20s} elbow K={v['bic_elbow_K']}  rule selects K={v['recommended_K']}")
