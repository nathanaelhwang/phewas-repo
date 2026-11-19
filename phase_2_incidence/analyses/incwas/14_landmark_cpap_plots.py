# %% [markdown]
# Step 14 - Landmark CPAP-adherence CIF plots (immortal-time-corrected).
#
# One figure per pilot phecode, two panels (grace 90 d | 180 d). Each panel: Aalen-Johansen cumulative
# incidence of the disease by 5-level CPAP adherence, among OSA patients, follow-up measured FROM THE
# LANDMARK (index+grace). The immortal-time-corrected companion to steps 09/10 (which were biased):
# expect the adherence curves to sit much closer together than the naive versions.
#
# CAVEAT (on figure): landmark fixes immortal-time only; healthy-adherer confounding remains
# (negative-control median HR ~1.01) -> associational, not causal.
#
# Usage: python 14_landmark_cpap_plots.py
# Emits plots/landmark_cpap/<phecode>_landmark.png. DE-IDENTIFIED.

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

OUTDIR = U.PLOTS / "landmark_cpap"
CPAP = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
GRACE_WINDOWS = [90, 180]
XMAX = 5.0
ADH_COLORS = {"4+ hr/night": "#1a9850", "2-4 hr/night": "#a6d96a", "0-2 hr/night": "#fdae61",
              "never-started": "#d73027", "unknown": "#999999"}


def aj_line(ax, d, color, label):
    n_ev = int((d["etype"] == 1).sum())
    lab = f"{label}  (n={len(d):,}, ev={n_ev:,})"
    if len(d) < 20 or n_ev < 5:
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
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: r.phenotype for r in dim.itertuples()}

    foll, _ = E.prep_followup()
    cpap = pd.read_parquet(CPAP).set_index("id")
    foll = foll.copy()
    foll["mean_pap_usage"] = foll["id"].map(cpap["mean_pap_usage"])
    foll["setup_date"] = foll["id"].map(cpap["setup_date"])   # adherence computed per window below

    prev, excl = E.prev_index(), E.exclude_map()
    graces = {w: E.grace_index(w) for w in GRACE_WINDOWS}
    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    ev = ev[ev["phecode"].isin(U.PILOT_PHECODES)]
    ev_groups = dict(list(ev.groupby("phecode", sort=False)))

    n_done = 0
    for phe in U.PILOT_PHECODES:
        if phe not in ev_groups:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
        drew = False
        for ax, window in zip(axes, GRACE_WINDOWS):
            d = E.build_survival(phe, ev_groups[phe], foll, prev, excl,
                                 grace_days=window, grace_by_phe=graces[window])
            d = d[d["severity"] != "None"].copy()
            if len(d) == 0 or int(d["event"].sum()) < U.MIN_EVENTS:
                ax.set_visible(False)
                continue
            drew = True
            d["adherence"] = E.adherence_landmark(d["mean_pap_usage"], d["setup_date"], d["idx"], window)
            d = d.assign(etype=np.where(d["event"] == 1, 1, np.where(d["is_death"], 2, 0)))
            for lvl in E.ADH_LEVELS:
                aj_line(ax, d[d["adherence"] == lvl], ADH_COLORS[lvl], lvl)
            ax.set_xlim(0, XMAX)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(f"years since landmark (index + {window} d)")
            ax.set_title(f"grace {window} d")
            leg = ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.92,
                            title="CPAP adherence")
            leg._legend_box.align = "left"
            ax.grid(alpha=0.25)
        if not drew:
            plt.close(fig)
            continue
        axes[0].set_ylabel("cumulative incidence (%)")
        fig.suptitle(f"{lab.get(phe, phe)}  [{phe}] — incident onset by CPAP adherence, LANDMARK "
                     f"(immortal-time corrected, OSA patients)\n"
                     f"descriptive; healthy-adherer confounding remains — association, not CPAP's causal effect",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUTDIR / f"{phe.replace('.', '_')}_landmark.png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        n_done += 1

    print(f"wrote {n_done} landmark CPAP figures -> {OUTDIR}")
    U.log_milestone(f"Step 14 landmark CPAP plots: {n_done} pilot phecodes x (grace 90|180d) adherence "
                    "CIF, immortal-time corrected.")


if __name__ == "__main__":
    main()
