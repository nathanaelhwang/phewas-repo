# %% [markdown]
# Step 06 - Pilot plots: cumulative-incidence curves by OSA severity + an HR forest.
#
# For each pilot phecode, an Aalen-Johansen cumulative-incidence-function (CIF) curve per severity
# group (None / Mild / Moderate / Severe), treating death as a COMPETING event (KM 1-S overstates
# absolute risk when severe-OSA mortality competes). Cumulative incidence (not survival) is plotted
# because the outcomes are rare. A number-at-risk table sits below each panel. Also a forest plot of
# the severe-vs-none m4 hazard ratios across the pilot. Reuses the engine's harmonized at-risk frame.
#
# Usage: python 06_pilot_plots.py [--jobs 6]
# Emits plots/pilot/<phecode>_cif.png + plots/pilot_hr_forest.png. DE-IDENTIFIED (curves/counts only).

# %%
import os
os.environ["MPLBACKEND"] = "Agg"   # force headless backend before matplotlib import (avoids broken Qt in workers)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from lifelines import AalenJohansenFitter

import incwas_utils as U
import _cox_engine as E

COLORS = {"None": "#4d4d4d", "Mild": "#f4a300", "Moderate": "#e8601c", "Severe": "#b2182b"}
XMAX = 5.0


def cif_plot(phecode, label, ev_slice):
    foll, _ = E.prep_followup()
    prev, excl = E.prev_index(), E.exclude_map()
    d = E.build_survival(phecode, ev_slice, foll, prev, excl)
    if len(d) == 0 or int(d["event"].sum()) < U.MIN_EVENTS:
        return None
    # competing-risks event code: 1 = incident phecode, 2 = death, 0 = censored
    d = d.assign(etype=np.where(d["event"] == 1, 1, np.where(d["is_death"], 2, 0)))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    atrisk = {}
    for sev in U.SEVERITIES:
        s = d[d["severity"] == sev]
        n_ev = int((s["etype"] == 1).sum())
        try:
            ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
            ajf.fit(s["time"].to_numpy(), s["etype"].to_numpy(), event_of_interest=1)
            cif = ajf.cumulative_density_
            ax.plot(cif.index, cif.iloc[:, 0].to_numpy() * 100, drawstyle="steps-post",
                    color=COLORS[sev], lw=2, label=f"{sev}  (n={len(s):,}, events={n_ev:,})")
        except Exception:
            ax.plot([], [], color=COLORS[sev], lw=2, label=f"{sev}  (n={len(s):,}, events={n_ev:,})")
        atrisk[sev] = [int((s["time"].to_numpy() >= t).sum()) for t in range(int(XMAX) + 1)]

    ax.set_xlim(0, XMAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("years since index sleep study")
    ax.set_ylabel("cumulative incidence (%)")
    ax.set_title(f"{label}  [{phecode}]\nincident onset by OSA severity (Aalen–Johansen, death competing)")
    leg = ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92,
                    title="OSA severity", title_fontsize=9)
    leg._legend_box.align = "left"
    ax.grid(alpha=0.25)

    # number-at-risk table
    tbl = "at risk:  " + "   ".join(f"y{t}" for t in range(int(XMAX) + 1))
    lines = [tbl] + [f"{sev[:4]:4s}  " + "  ".join(f"{atrisk[sev][t]:6,}" for t in range(int(XMAX) + 1))
                     for sev in U.SEVERITIES]
    ax.text(0.0, -0.30, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            family="monospace", va="top")
    fig.subplots_adjust(bottom=0.32)
    out = U.PILOT_PLOTS / f"{phecode.replace('.', '_')}_cif.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return phecode


def forest(dim):
    d = pd.read_csv(U.ANALYSIS / "results_cox_pilot.csv", dtype={"phecode": str})
    h = d[(d.contrast == "severe_vs_none") & (d.model == "m4") & d.effect.notna()].copy()
    h = h.sort_values("effect")
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.42 * len(h))))
    y = np.arange(len(h))
    ax.errorbar(h["effect"], y, xerr=[h["effect"] - h["ci_low"], h["ci_high"] - h["effect"]],
                fmt="o", color="#2166ac", ecolor="#999999", capsize=2, ms=5)
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{str(r.phenotype)[:32]} [{r.phecode}]" for r in h.itertuples()], fontsize=8)
    ax.set_xlabel("severe-vs-none hazard ratio (m4-adjusted)")
    ax.set_title("Incident-disease risk in severe OSA — pilot phecodes (Cox m4)")
    for r, yy in zip(h.itertuples(), y):
        star = "*" if r.sig_fdr else ""
        ax.text(r.ci_high + 0.005, yy, f"{r.effect:.2f}{star}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(U.PLOTS / "pilot_hr_forest.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    U.ensure_dirs()

    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: r.phenotype for r in dim.itertuples()}
    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    ev = ev[ev["phecode"].isin(U.PILOT_PHECODES)]
    groups = [(p, s) for p, s in ev.groupby("phecode", sort=False)]

    done = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(cif_plot)(p, lab.get(p, p), s) for p, s in groups)
    done = [x for x in done if x]
    forest(dim)
    print(f"wrote {len(done)} CIF panels -> {U.PILOT_PLOTS}")
    print(f"wrote forest -> {U.PLOTS / 'pilot_hr_forest.png'}")
    U.log_milestone(f"Step 06 pilot plots: {len(done)} Aalen-Johansen CIF panels (4 severity groups, "
                    f"competing-risk death) + severe-vs-none m4 HR forest.")


if __name__ == "__main__":
    main()
