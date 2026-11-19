# %% [markdown]
# Step 13 - Landmark CPAP-adherence survival-curve export (tabular, for user plotting).
#
# Landmark-anchored (grace 90 & 180 d) incidence data by 5-level CPAP adherence, among OSA patients,
# for the FDR-significant phecodes. Mirrors step 11 but: time origin = index+GRACE; exposure = adherence
# level (4+/2-4/0-2 hr/night / never-started / unknown) instead of severity; immortal-time-corrected.
#
# Outputs (results/incwas_results/survival_curves_landmark/):
#   survival_long_landmark.parquet   per-patient (no id): phecode, phenotype, window_days, time_years,
#                                    etype (0 cens/1 incident/2 death), severity, adherence
#   curve_adherence.csv              plot-ready Aalen-Johansen CIF at every fitted event (OSA-pooled)
#   README.md                        schema + caveats
# time_years is measured FROM THE LANDMARK. DE-IDENTIFIED.

# %%
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from _event_curves import event_curve

import incwas_utils as U
import _cox_engine as E

OUTDIR = U.RESULTS / "survival_curves_landmark"
CPAP = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
GRACE_WINDOWS = [90, 180]
HORIZON = 5
MIN_EV = 5
SEV_CONTRASTS = ["omnibus", "trend", "mild_vs_none", "moderate_vs_none",
                 "severe_vs_none", "ahi_ge5", "ahi_ge15"]


def curve_rows(d, phe, pheno, window, group):
    t, ev = d["time"].to_numpy(float), d["etype"].to_numpy()
    if len(d) < 20 or int((ev == 1).sum()) < MIN_EV:
        return None
    curve = event_curve(t, ev, HORIZON)
    return pd.DataFrame({"phecode": phe, "phenotype": pheno, "window_days": window, "group": group,
                         **curve.to_dict("list")})


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: r.phenotype for r in dim.itertuples()}
    res = pd.read_csv(U.ANALYSIS / "results_cox.csv", dtype={"phecode": str})
    m4 = res[(res.model == "m4") & res.contrast.isin(SEV_CONTRASTS)]
    fdr = sorted(m4[m4.sig_fdr]["phecode"].unique())

    foll, _ = E.prep_followup()
    cpap = pd.read_parquet(CPAP).set_index("id")
    foll = foll.copy()
    foll["mean_pap_usage"] = foll["id"].map(cpap["mean_pap_usage"])
    foll["setup_date"] = foll["id"].map(cpap["setup_date"])   # adherence computed per window below

    prev, excl = E.prev_index(), E.exclude_map()
    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev["phecode"].astype(str)
    ev = ev[ev["phecode"].isin(fdr)]
    ev_groups = dict(list(ev.groupby("phecode", sort=False)))

    long_parts, curve_parts = [], []
    for window in GRACE_WINDOWS:
        grace = E.grace_index(window)
        for phe in fdr:
            if phe not in ev_groups:
                continue
            d = E.build_survival(phe, ev_groups[phe], foll, prev, excl,
                                 grace_days=window, grace_by_phe=grace)
            d = d[d["severity"] != "None"].copy()
            if len(d) == 0 or int(d["event"].sum()) < U.MIN_EVENTS:
                continue
            d["adherence"] = E.adherence_landmark(d["mean_pap_usage"], d["setup_date"], d["idx"], window)
            d = d.assign(etype=np.where(d["event"] == 1, 1, np.where(d["is_death"], 2, 0)))
            long_parts.append(pd.DataFrame({
                "phecode": phe, "phenotype": lab.get(phe, phe), "window_days": window,
                "time_years": np.round(d["time"].to_numpy(), 5), "etype": d["etype"].to_numpy(),
                "severity": d["severity"].to_numpy(), "adherence": d["adherence"].to_numpy()}))
            for lvl in E.ADH_LEVELS:                       # 5 adherence curves, OSA-pooled
                r = curve_rows(d[d["adherence"] == lvl], phe, lab.get(phe, phe), window, lvl)
                if r is not None:
                    curve_parts.append(r)

    pd.concat(long_parts, ignore_index=True).to_parquet(OUTDIR / "survival_long_landmark.parquet", index=False)
    pd.concat(curve_parts, ignore_index=True).to_csv(OUTDIR / "curve_adherence.csv", index=False)
    print(f"long rows: {sum(len(p) for p in long_parts):,}; curve rows: {sum(len(p) for p in curve_parts):,}")
    print(f"-> {OUTDIR}")

    (OUTDIR / "README.md").write_text(
        "# incwas — landmark CPAP-adherence survival export\n\n"
        "Immortal-time-corrected (grace-period LANDMARK) incidence data by 5-level CPAP adherence, among "
        "OSA patients, for the FDR-significant phecodes. **`time_years` is measured from the landmark "
        "(index + `window_days`), not from diagnosis.** Two windows: `window_days` = 90 (approximate: the "
        "therapy-anchored 90-day compliance window closes post-landmark for ~81%) and 180 (genuine: closed "
        "for ~82%).\n\n"
        "## Files\n"
        "- `survival_long_landmark.parquet` — per-patient (no id): `phecode, phenotype, window_days, "
        "time_years, etype` (0=censored, 1=incident phecode, 2=death competing), `severity, adherence`.\n"
        "- `curve_adherence.csv` — plot-ready Aalen-Johansen cumulative incidence by adherence level "
        "(OSA-pooled), every fitted event plus annual checkpoints: `phecode, phenotype, window_days, group, time_years, cif_pct, "
        "n_at_risk, n_events_cum`.\n\n"
        "Adherence levels: `4+ hr/night` (reference in the Cox), `2-4 hr/night`, `0-2 hr/night`, "
        "`never-started` (no device), `unknown` (device set up, usage not captured).\n\n"
        "## Caveats\n"
        "- The landmark removes **immortal-time bias** (clock starts at the landmark; survival-to-landmark "
        "required). It does NOT remove **healthy-adherer confounding** — see the negative-control panel in "
        "`results_cpap_landmark_{90,180}.csv` (median HR ~1.01). Associational, not causal.\n"
        "- Competing-risk CIF (death competing); do not redraw as 1-KM.\n"
        "- The adjusted adherence HRs are in `analysis/results_cpap_landmark_{90,180}.csv`; these curves "
        "are the unadjusted descriptive companion.\n", encoding="utf-8")
    U.log_milestone("Step 13 landmark CPAP export: per-patient + AJ CIF by adherence (grace 90 & 180d) "
                    "for FDR-sig phecodes -> survival_curves_landmark/.")


if __name__ == "__main__":
    main()
