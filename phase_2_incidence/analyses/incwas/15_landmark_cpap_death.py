# %% [markdown]
# Step 15 - Death incidence stratified by CPAP adherence, LANDMARK design (immortal-time corrected).
#
# All-cause mortality (KM cumulative incidence) by 5-level within-window CPAP adherence, among OSA
# patients, follow-up measured FROM THE LANDMARK (index+grace). The landmark analog of step 09 (which
# was index-anchored / immortal-time biased and used the 3-level therapy-anchored bin). Two panels
# (grace 90 | 180). Adherence is derived STRICTLY from [index, index+grace] (adherence_landmark).
# Death is the event (no competing risk here) -> plain KM cumulative mortality.
#
# Also fits the adjusted Cox (adherence vs 4+ hr/night ref, + m4 covariates + severity) -> the crude
# curves are heavily HEALTHY-ADHERER confounded (adherent patients are systematically healthier; the
# classic "adherence-to-placebo" effect is strongest for mortality), so the adjusted HR is the number
# to read, and it is still associational, not causal.
#
# Usage: python 15_landmark_cpap_death.py
# Emits plots/landmark_cpap_death.png + analysis/results_cpap_landmark_death.csv. DE-IDENTIFIED.

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
import _cox_engine as E

CPAP = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
GRACE_WINDOWS = [90, 180]
XMAX = 5.0
ADH_ORDER = ["4+ hr/night", "2-4 hr/night", "0-2 hr/night", "never-started", "unknown"]
ADH_COLORS = {"4+ hr/night": "#1a9850", "2-4 hr/night": "#a6d96a", "0-2 hr/night": "#fdae61",
              "never-started": "#d73027", "unknown": "#999999"}
ADH_DUMMY = {"adh_0_2": "0-2 hr/night", "adh_2_4": "2-4 hr/night",
             "adh_never": "never-started", "adh_unknown": "unknown"}   # all vs 4+ hr/night (ref)


def build_death(foll, window):
    """Landmark death frame: OSA patients alive/enrolled past index+window; time from the landmark."""
    land = foll["idx"] + pd.to_timedelta(window, unit="D")
    d = foll[(foll["severity"] != "None") & (foll["censor_date"] > land)].copy()
    d["adherence"] = E.adherence_landmark(d["mean_pap_usage"], d["setup_date"], d["idx"], window)
    d["time"] = (d["censor_date"] - land).dt.days / 365.25
    d["event"] = d["is_death"].astype(int)
    return d[d["time"] > 0].copy()


def km_line(ax, d, color, label):
    n_ev = int(d["event"].sum())
    lab = f"{label}  (n={len(d):,}, deaths={n_ev:,})"
    if len(d) < 20 or n_ev < 5:
        ax.plot([], [], color=color, lw=2, ls=":", label=lab + " —sparse")
        return
    kmf = KaplanMeierFitter()
    kmf.fit(d["time"].to_numpy(), d["event"].to_numpy())
    cd = kmf.cumulative_density_
    ax.plot(cd.index, cd.iloc[:, 0].to_numpy() * 100, drawstyle="steps-post", color=color, lw=2, label=lab)


def cox_death(d, blocks, window):
    """Adjusted Cox: adherence (ref 4+h) + m4 covariates + severity_ord -> HRs for mortality."""
    for c, lvl in ADH_DUMMY.items():
        d[c] = (d["adherence"] == lvl).astype(float)
    n_ref = int((d["adherence"] == "4+ hr/night").sum())
    if n_ref < U.MIN_EVENTS:
        return [dict(window_days=window, level="(levels infeasible)", n_ref=n_ref)]
    adh_cols = [c for c in ADH_DUMMY if d[c].nunique() > 1]
    cov = blocks["models"]["m4"] + ["severity_ord"]
    rows = []
    try:
        cph, kept = E._fit(d, adh_cols + cov)
        for c in adh_cols:
            if c in kept:
                e, se, lo, hi, p = E._eff(cph, c)
                rows.append(dict(window_days=window, level=f"{ADH_DUMMY[c]}_vs_4plus", n_ref=n_ref,
                                 HR=round(e, 3), ci_low=round(lo, 3), ci_high=round(hi, 3), p=p))
    except Exception as ex:
        rows.append(dict(window_days=window, level="error", n_ref=n_ref, note=str(ex)[:50]))
    return rows


def main() -> None:
    U.ensure_dirs()
    foll, blocks = E.prep_followup()
    cpap = pd.read_parquet(CPAP).set_index("id")
    foll = foll.copy()
    foll["mean_pap_usage"] = foll["id"].map(cpap["mean_pap_usage"])
    foll["setup_date"] = foll["id"].map(cpap["setup_date"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    cox_rows = []
    for ax, window in zip(axes, GRACE_WINDOWS):
        d = build_death(foll, window)
        for lvl in ADH_ORDER:
            km_line(ax, d[d["adherence"] == lvl], ADH_COLORS[lvl], lvl)
        ax.set_xlim(0, XMAX)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"years since landmark (index + {window} d)")
        ax.set_title(f"grace {window} d")
        leg = ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.92,
                        title="CPAP adherence")
        leg._legend_box.align = "left"
        ax.grid(alpha=0.25)
        cox_rows += cox_death(d, blocks, window)
        m = d[["adherence", "event"]].groupby("adherence")["event"].agg(["size", "sum", "mean"])
        print(f"\n[grace {window}d] crude mortality by adherence (n={len(d):,}, deaths={int(d['event'].sum()):,}):")
        for lvl in ADH_ORDER:
            if lvl in m.index:
                print(f"   {lvl:14s} n={int(m.loc[lvl,'size']):>6,} deaths={int(m.loc[lvl,'sum']):>4,} "
                      f"({100*m.loc[lvl,'mean']:.2f}%)")

    axes[0].set_ylabel("cumulative all-cause mortality (%)")
    fig.suptitle("All-cause mortality by CPAP adherence, LANDMARK (immortal-time corrected, OSA patients)\n"
                 "crude curves are healthy-adherer confounded — read the adjusted HRs; association, not "
                 "CPAP's causal effect", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = U.PLOTS / "landmark_cpap_death.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    cox = pd.DataFrame(cox_rows)
    cox.to_csv(U.ANALYSIS / "results_cpap_landmark_death.csv", index=False)
    print(f"\nadjusted mortality HRs (vs 4+ hr/night, m4 + severity):")
    print(cox.to_string(index=False))
    print(f"\nwrote {out}\nwrote {U.ANALYSIS / 'results_cpap_landmark_death.csv'}")
    U.log_milestone("Step 15 landmark CPAP death: all-cause mortality by 5-level within-window adherence "
                    "(grace 90|180), KM curves + adjusted Cox HRs; healthy-adherer caveat.")


if __name__ == "__main__":
    main()
