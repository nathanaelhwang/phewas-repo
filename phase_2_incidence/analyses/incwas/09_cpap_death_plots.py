# %% [markdown]
# Step 09 - CPAP-usage-stratified mortality + death-incidence plots.
#
# (A) All-cause mortality (cumulative, KM 1-S) by OSA severity: None/Mild/Moderate/Severe.
# (B) Three panels (Mild, Moderate, Severe): No-OSA reference + the severity stratified by CPAP nightly
#     use (<2 / 2-4 / 4+ hr/night, from mean_pap_usage in MINUTES: <120 / 120-240 / >=240). Death is the
#     outcome (the canonical "effect of CPAP" endpoint); event = death, time = follow-up to death/censor
#     already in the follow-up frame.
#
# CAVEAT (printed on the figures): these are DESCRIPTIVE and confounded -- CPAP adherence is a
# healthy-adherer / immortal-time marker (adherent patients are systematically healthier and must
# survive the adherence-assessment window), so the adherence-mortality gradient is association, not a
# causal CPAP effect. CPAP is deliberately NOT used as a covariate elsewhere (it is a mediator).
#
# Usage: python 09_cpap_death_plots.py
# Emits plots/death_by_severity.png + plots/cpap_mortality_{mild,moderate,severe}.png.

# %%
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

import incwas_utils as U

CPAP = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
XMAX = 6.0
SEV_COLORS = {"None": "#4d4d4d", "Mild": "#f4a300", "Moderate": "#e8601c", "Severe": "#b2182b"}
BINS = [("<2 hr/night", "#d73027"), ("2–4 hr/night", "#fdae61"), ("4+ hr/night", "#1a9850")]
NONE_COLOR = "#4d4d4d"


def cpap_bin(minutes):
    if pd.isna(minutes):
        return None
    if minutes < 120:
        return "<2 hr/night"
    if minutes < 240:
        return "2–4 hr/night"
    return "4+ hr/night"


def km_line(ax, g, color, label):
    if len(g) < 20 or g["is_death"].sum() < 3:
        return
    kmf = KaplanMeierFitter()
    kmf.fit(g["followup_years"].to_numpy(), g["is_death"].astype(int).to_numpy())
    cd = kmf.cumulative_density_
    ax.plot(cd.index, cd.iloc[:, 0].to_numpy() * 100, drawstyle="steps-post", color=color, lw=2,
            label=f"{label}  (n={len(g):,}, deaths={int(g['is_death'].sum()):,})")


def at_risk_table(ax, groups):
    lines = ["at risk:  " + "   ".join(f"y{t}" for t in range(int(XMAX) + 1))]
    for lab, g in groups:
        counts = [int((g["followup_years"].to_numpy() >= t).sum()) for t in range(int(XMAX) + 1)]
        lines.append(f"{lab[:14]:14s}  " + "  ".join(f"{c:6,}" for c in counts))
    ax.text(0.0, -0.32, "\n".join(lines), transform=ax.transAxes, fontsize=6.5,
            family="monospace", va="top")
    ax.figure.subplots_adjust(bottom=0.34)


def finish(ax, title, legend_title):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("years since index sleep study")
    ax.set_ylabel("cumulative all-cause mortality (%)")
    ax.set_title(title)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92,
                    title=legend_title, title_fontsize=9)
    leg._legend_box.align = "left"
    ax.grid(alpha=0.25)


def main() -> None:
    U.ensure_dirs()
    foll = pd.read_parquet(U.FOLLOWUP_FRAME)
    cpap = pd.read_parquet(CPAP)[["id", "mean_pap_usage"]]
    foll = foll.merge(cpap, on="id", how="left")
    foll["cpap_bin"] = foll["mean_pap_usage"].map(cpap_bin)

    # ---- (A) death incidence by OSA severity --------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    groups = []
    for sev in U.SEVERITIES:
        g = foll[foll["severity"] == sev]
        km_line(ax, g, SEV_COLORS[sev], sev)
        groups.append((sev, g))
    finish(ax, "All-cause mortality by OSA severity", "OSA severity")
    at_risk_table(ax, groups)
    fig.savefig(U.PLOTS / "death_by_severity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- (B) CPAP-usage-stratified mortality, one panel per severity --------
    none = foll[foll["severity"] == "None"]
    for sev in ["Mild", "Moderate", "Severe"]:
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        groups = [("No OSA", none)]
        km_line(ax, none, NONE_COLOR, "No OSA")
        sub = foll[foll["severity"] == sev]
        n_pap = sub["cpap_bin"].notna().sum()
        for lab, color in BINS:
            g = sub[sub["cpap_bin"] == lab]
            km_line(ax, g, color, f"{sev} · {lab}")
            groups.append((lab, g))
        cov = 100 * n_pap / len(sub) if len(sub) else 0
        finish(ax, f"{sev} OSA — all-cause mortality by CPAP nightly use\n"
                   f"(descriptive; healthy-adherer/immortal-time confounded · PAP data on {cov:.0f}% of {sev})",
               f"{sev} OSA: CPAP use")
        at_risk_table(ax, groups)
        fig.savefig(U.PLOTS / f"cpap_mortality_{sev.lower()}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # de-identified summary
    print("CPAP bin sizes by severity (patients with PAP data):")
    tab = (foll[foll["cpap_bin"].notna()].groupby(["severity", "cpap_bin"]).size()
           .unstack(fill_value=0).reindex(U.SEVERITIES))
    print(tab.to_string())
    print("\nCrude mortality by severity:")
    for sev in U.SEVERITIES:
        g = foll[foll["severity"] == sev]
        print(f"   {sev:9s}: {100*g['is_death'].mean():.2f}% ({int(g['is_death'].sum()):,}/{len(g):,})")
    print(f"\nwrote death_by_severity.png + cpap_mortality_{{mild,moderate,severe}}.png -> {U.PLOTS}")
    U.log_milestone("Step 09 CPAP/death plots: mortality by severity + CPAP-nightly-use-stratified "
                    "mortality (Mild/Mod/Severe vs No-OSA); descriptive, healthy-adherer caveat noted.")


if __name__ == "__main__":
    main()
