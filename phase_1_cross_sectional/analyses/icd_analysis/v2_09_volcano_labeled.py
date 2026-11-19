# %% [markdown]
# v2 Step 9 - Labeled volcano plots for the severe_vs_none IPTW sensitivity analysis.
# Three volcanoes on SHARED axes for direct comparison, each labeling its top-25 most
# important (most significant; ties broken by larger |effect|) phecodes:
#   - IPTW marginal (primary sensitivity estimate)
#   - ATO / overlap-weighted marginal (exact-balance robustness arm)
#   - M4 (the published L1 conditional result being tested)
# Reuses the verified marginal_or_sandwich() from v2_08 (recomputes ATO p, not saved in the CSV).
#
# Usage: python v2_09_volcano_labeled.py

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U
import v2_08_ipw_severe_vs_none as ipw          # reuse load_design + marginal_or_sandwich + fdr
import statsmodels.formula.api as smf

OUT = U.V2_RESULTS / "sensitivity_ipw"
PLOTS = OUT / "plots"
COMBINED = U.V2_SUMMARY / "phewas_results_combined.csv"
TOPN = 25
MAXLEN = 30                                       # label truncation


def build_inputs() -> dict[str, pd.DataFrame]:
    """Return {method: df[phecode, name, log2or, neglog10p, sig_fdr, valid]} for the 3 methods."""
    grp, fm, codes, cov = ipw.load_design(U.V2_PYPHEWAS, None)
    g = grp["genotype"].to_numpy(np.float64)
    ps = np.clip(smf.logit("genotype ~ " + " + ".join(cov), data=grp).fit(disp=0, maxiter=200)
                 .predict(grp).to_numpy(), 1e-6, 1 - 1e-6)
    p_bar = g.mean()
    sw = g * (p_bar / ps) + (1 - g) * ((1 - p_bar) / (1 - ps))
    lo, hi = np.quantile(sw, ipw.TRUNC); sw = np.clip(sw, lo, hi)          # IPTW primary
    ato = g * (1 - ps) + (1 - g) * ps                                     # overlap weights

    # phecode -> name/category from the main combined table
    comb = pd.read_csv(COMBINED, dtype={"PheWAS Code": str})
    m4 = comb[(comb.contrast == "severe_vs_none") & (comb.model == "m4_bmi")].copy()
    names = m4.set_index("PheWAS Code")["PheWAS Name"].to_dict()

    def from_sandwich(w):
        r = ipw.marginal_or_sandwich(fm, g, w)
        r["phecode"] = codes
        r = r[r.valid & ~r.unstable].copy()
        r["log2or"] = np.log2(r["OR"])
        r["neglog10p"] = -np.log10(np.clip(r["p"], 1e-300, None))
        r["sig_fdr"] = r["p"] < ipw.fdr(r["p"])
        r["name"] = r["phecode"].map(names).fillna(r["phecode"])
        return r[["phecode", "name", "log2or", "neglog10p", "p", "sig_fdr"]]

    m4d = m4.dropna(subset=["p-val", "OR"]).copy()
    m4d = m4d[m4d["unstable"] != True]
    m4d["log2or"] = np.log2(m4d["OR"].clip(lower=1e-6))
    m4d["neglog10p"] = -np.log10(m4d["p-val"].clip(lower=1e-300))
    m4d = m4d.rename(columns={"PheWAS Code": "phecode", "PheWAS Name": "name",
                              "p-val": "p", "sig_fdr": "sig_fdr"})
    return {"IPTW (marginal)": from_sandwich(sw),
            "ATO / overlap (marginal)": from_sandwich(ato),
            "M4 (conditional, L1)": m4d[["phecode", "name", "log2or", "neglog10p", "p", "sig_fdr"]]}


def top_important(df: pd.DataFrame, n: int = TOPN) -> pd.DataFrame:
    return df.assign(_abs=df.log2or.abs()).sort_values(["neglog10p", "_abs"], ascending=False).head(n)


def plot_volcano(df, tops, title, out, xlim, ylim, fdr_line):
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    up = df[df.sig_fdr & (df.log2or > 0)]
    dn = df[df.sig_fdr & (df.log2or < 0)]
    ns = df[~df.sig_fdr]
    ax.scatter(ns.log2or, ns.neglog10p, s=7, c="#c9ccd1", alpha=0.5, linewidths=0, label="not FDR-sig")
    ax.scatter(up.log2or, up.neglog10p, s=12, c="#c0392b", alpha=0.65, linewidths=0, label="FDR-sig, OR>1")
    ax.scatter(dn.log2or, dn.neglog10p, s=12, c="#2c6fbb", alpha=0.65, linewidths=0, label="FDR-sig, OR<1")
    ax.scatter(tops.log2or, tops.neglog10p, s=26, facecolors="none", edgecolors="black", linewidths=0.7)

    if np.isfinite(fdr_line):
        ax.axhline(fdr_line, ls="--", lw=0.7, color="grey")
        ax.text(xlim[1], fdr_line, " FDR", va="bottom", ha="right", fontsize=7, color="grey")
    ax.axvline(0, lw=0.6, color="grey")

    texts = [ax.text(r.log2or, r.neglog10p, (r["name"][:MAXLEN] + ("…" if len(r["name"]) > MAXLEN else "")),
                     fontsize=6.5) for _, r in tops.iterrows()]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4),
                expand_points=(1.4, 1.6), force_text=(0.4, 0.6))

    ax.set(xlim=xlim, ylim=ylim, xlabel="log2 OR (Severe vs None)", ylabel="-log10 p",
           title=title)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out.name}  (labeled {len(tops)} phecodes)")


def main() -> None:
    data = build_inputs()
    tops = {k: top_important(v) for k, v in data.items()}

    # shared axes: contain every labeled point across the three, plus the bulk
    all_lab = pd.concat(tops.values())
    xabs = max(all_lab.log2or.abs().max(),
               max(np.percentile(v.log2or.abs(), 99) for v in data.values())) * 1.12
    ymax = all_lab.neglog10p.max() * 1.08
    xlim, ylim = (-xabs, xabs), (0, ymax)

    files = {"IPTW (marginal)": "volcano_ipw_labeled.png",
             "ATO / overlap (marginal)": "volcano_ato_labeled.png",
             "M4 (conditional, L1)": "volcano_m4_labeled.png"}
    print(f"Volcano plots (top {TOPN} labeled each, shared axes x={xlim[0]:.2f}..{xlim[1]:.2f}, y=0..{ylim[1]:.0f}):")
    for k, df in data.items():
        fline = -np.log10(ipw.fdr(df["p"])) if ipw.fdr(df["p"]) > 0 else np.nan
        plot_volcano(df, tops[k], f"Severe vs None - {k}\n(top {TOPN} by significance labeled)",
                     PLOTS / files[k], xlim, ylim, fline)


if __name__ == "__main__":
    main()
