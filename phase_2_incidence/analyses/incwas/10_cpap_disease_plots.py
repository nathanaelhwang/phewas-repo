# %% [markdown]
# Step 10 - CPAP-usage-stratified INCIDENT-DISEASE curves for the pilot phecodes.
#
# For each pilot phecode, a 3-panel figure (Mild | Moderate | Severe OSA). Each panel plots the
# Aalen-Johansen cumulative incidence of the disease (death competing) for four groups:
#   No OSA (reference)  +  that severity split by CPAP nightly use: <2 / 2-4 / 4+ hr/night.
# Same at-risk / harmonized-washout / event definition as the main incidence CIF (step 06); the only
# change is stratifying the OSA arm by mean_pap_usage (minutes: <120 / 120-240 / >=240).
#
# CAVEAT (on the figure): descriptive only -- CPAP adherence is a healthy-adherer / immortal-time
# marker, so between-strata gaps are association, not CPAP's causal effect on disease incidence.
#
# Usage: python 10_cpap_disease_plots.py
# Emits plots/pilot_cpap/<phecode>_cpap.png (one 3-panel figure per pilot phecode). DE-IDENTIFIED.

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
from lifelines import AalenJohansenFitter

import incwas_utils as U
import _cox_engine as E

CPAP = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
OUTDIR = U.PLOTS / "pilot_cpap"
XMAX = 5.0
NONE_COLOR = "#4d4d4d"
BINS = [("<2 hr/night", "#d73027"), ("2–4 hr/night", "#fdae61"), ("4+ hr/night", "#1a9850")]
MIN_EV = 5   # minimum incident events to draw a stratum line


def cpap_bin(m):
    if pd.isna(m):
        return None
    if m < 120:
        return "<2 hr/night"
    if m < 240:
        return "2–4 hr/night"
    return "4+ hr/night"


def aj_line(ax, d, color, label):
    n_ev = int((d["etype"] == 1).sum())
    lab = f"{label}  (n={len(d):,}, ev={n_ev:,})"
    if len(d) < 20 or n_ev < MIN_EV:
        ax.plot([], [], color=color, lw=2, ls=":", label=lab + " —sparse")
        return
    try:
        ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
        ajf.fit(d["time"].to_numpy(), d["etype"].to_numpy(), event_of_interest=1)
        cif = ajf.cumulative_density_
        ax.plot(cif.index, cif.iloc[:, 0].to_numpy() * 100, drawstyle="steps-post",
                color=color, lw=2, label=lab)
    except Exception:
        ax.plot([], [], color=color, lw=2, ls=":", label=lab + " —na")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    foll, blocks = E.prep_followup()
    prev, excl = E.prev_index(), E.exclude_map()
    cpap = pd.read_parquet(CPAP).set_index("id")["mean_pap_usage"]
    foll = foll.copy()
    foll["cpap_bin"] = foll["id"].map(cpap).map(cpap_bin)   # .map preserves the frame's index order

    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: r.phenotype for r in dim.itertuples()}
    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    ev = ev[ev["phecode"].isin(U.PILOT_PHECODES)]

    n_done = 0
    for phe, ev_slice in ev.groupby("phecode", sort=False):
        d = E.build_survival(phe, ev_slice, foll, prev, excl)
        if len(d) == 0 or int(d["event"].sum()) < U.MIN_EVENTS:
            continue
        d = d.assign(etype=np.where(d["event"] == 1, 1, np.where(d["is_death"], 2, 0)))
        none = d[d["severity"] == "None"]

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
        for ax, sev in zip(axes, ["Mild", "Moderate", "Severe"]):
            aj_line(ax, none, NONE_COLOR, "No OSA")
            sub = d[d["severity"] == sev]
            for binlab, color in BINS:
                aj_line(ax, sub[sub["cpap_bin"] == binlab], color, f"{sev} · {binlab}")
            ax.set_xlim(0, XMAX)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{sev} OSA")
            ax.set_xlabel("years since index")
            leg = ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.92)
            leg._legend_box.align = "left"
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("cumulative incidence (%)")
        fig.suptitle(f"{lab.get(phe, phe)}  [{phe}] — incident onset by CPAP nightly use "
                     f"(Aalen–Johansen, death competing)\n"
                     f"descriptive; healthy-adherer / immortal-time confounded — association, not CPAP's causal effect",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUTDIR / f"{phe.replace('.', '_')}_cpap.png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        n_done += 1

    print(f"wrote {n_done} CPAP-stratified disease figures -> {OUTDIR}")
    U.log_milestone(f"Step 10 CPAP disease plots: {n_done} pilot phecodes x 3 severity panels "
                    "(No-OSA + <2/2-4/4+ hr CPAP), incident-onset CIF; descriptive/confounded caveat.")


if __name__ == "__main__":
    main()
