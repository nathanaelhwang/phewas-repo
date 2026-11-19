# %% [markdown]
# Step 11 - Export tabular survival data for the FDR-significant phecodes, so curves can be drawn in
# any tool (R survminer/ggsurvplot, cmprsk, python, ...). Two stratifications: OSA severity, and CPAP
# nightly use within each severity (vs No-OSA). Same harmonized-washout at-risk / competing-death
# definition as the CIF plots (steps 06/10).
#
# FDR-significant set = FDR-sig at m4 in ANY severity contrast (trend / mild-mod-severe-vs-none /
# omnibus / AHI>=5 / AHI>=15). Sleep apnea 327.3* (circular OSA-recoding control) is included but
# flagged (osa_control=True) so it can be dropped.
#
# Outputs (results/incwas_results/survival_curves/):
#   survival_long.parquet   per-patient rows: phecode, phenotype, time_years, etype, severity, cpap_bin
#                           (etype: 0=censored, 1=incident phecode, 2=death competing). NO patient id.
#   curve_severity.csv      plot-ready CIF by severity, at every fitted event: + risk/event counts
#   curve_cpap.csv          plot-ready CIF by CPAP use within each severity panel (+ No-OSA reference)
#   fdr_phecodes.csv        the phecode list: n_atrisk, n_events, which contrasts are FDR-sig, flags
# DE-IDENTIFIED: only aggregate counts to stdout; patient-level table has no id and lands on disk.

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

OUTDIR = U.RESULTS / "survival_curves"
CPAP = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
SEV_CONTRASTS = ["omnibus", "trend", "mild_vs_none", "moderate_vs_none",
                 "severe_vs_none", "ahi_ge5", "ahi_ge15"]
HORIZON = 6
MIN_EV = 5   # minimum incident events to emit a fitted curve for a group


def cpap_bin(m):
    if pd.isna(m):
        return None
    return "<2 hr/night" if m < 120 else ("2-4 hr/night" if m < 240 else "4+ hr/night")


def curve_rows(d, phe, pheno, label, extra):
    """Fitted AJ CIF at every jump, with annual checkpoints, for one group `d`."""
    t = d["time"].to_numpy(dtype=float)
    ev = d["etype"].to_numpy()
    n_ev = int((ev == 1).sum())
    if len(d) < 20 or n_ev < MIN_EV:
        return None
    curve = event_curve(t, ev, HORIZON)
    return pd.DataFrame({"phecode": phe, "phenotype": pheno, **extra, **curve.to_dict("list")})


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(U.ANALYSIS / "results_cox.csv", dtype={"phecode": str})
    m4 = res[(res.model == "m4") & res.contrast.isin(SEV_CONTRASTS)]
    sig = m4[m4.sig_fdr]
    fdr_phe = sorted(sig["phecode"].unique(), key=lambda p: [int(x) for x in p.split(".") + ["0"]][:2]
                     if p.replace(".", "").isdigit() else [9999])
    sig_by = sig.groupby("phecode")["contrast"].apply(lambda s: ",".join(sorted(set(s))))
    sev_hr = m4[m4.contrast == "severe_vs_none"].set_index("phecode")["effect"]
    print(f"FDR-significant phecodes (m4, any severity contrast): {len(fdr_phe)}", flush=True)

    foll, _ = E.prep_followup()
    prev, excl = E.prev_index(), E.exclude_map()
    cpap = pd.read_parquet(CPAP).set_index("id")["mean_pap_usage"]
    foll = foll.copy()
    foll["cpap_bin"] = foll["id"].map(cpap).map(cpap_bin)

    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode": str}, keep_default_na=False)
    lab = {r.phecode: (r.phenotype, r.category) for r in dim.itertuples()}
    ev_all = pd.read_parquet(U.EVENTS)
    ev_all["phecode"] = ev_all["phecode"].astype(str)
    ev_all = ev_all[ev_all["phecode"].isin(fdr_phe)]
    ev_groups = dict(list(ev_all.groupby("phecode", sort=False)))

    long_parts, sev_parts, cpap_parts, summ = [], [], [], []
    for phe in fdr_phe:
        pheno, cat = lab.get(phe, (phe, ""))
        d = E.build_survival(phe, ev_groups[phe], foll, prev, excl)
        if len(d) == 0:
            continue
        d = d.assign(etype=np.where(d["event"] == 1, 1, np.where(d["is_death"], 2, 0)))

        # per-patient long rows (no id). Relabel "None" -> "No OSA": the string "None" is in pandas'
        # default NA list and would read back as NaN.
        sev_disp = np.where(d["severity"].to_numpy() == "None", "No OSA", d["severity"].to_numpy())
        long_parts.append(pd.DataFrame({
            "phecode": phe, "phenotype": pheno,
            "time_years": np.round(d["time"].to_numpy(), 5), "etype": d["etype"].to_numpy(),
            "severity": sev_disp, "cpap_bin": d["cpap_bin"].to_numpy()}))

        # severity-stratified curves
        for sev in U.SEVERITIES:
            disp = "No OSA" if sev == "None" else sev
            r = curve_rows(d[d["severity"] == sev], phe, pheno, disp, {"group": disp})
            if r is not None:
                sev_parts.append(r)

        # CPAP-stratified curves (per severity panel; No-OSA = None group as reference in each panel)
        none = d[d["severity"] == "None"]
        for panel in ["Mild", "Moderate", "Severe"]:
            r0 = curve_rows(none, phe, pheno, "No OSA", {"panel": panel, "group": "No OSA"})
            if r0 is not None:
                cpap_parts.append(r0)
            sub = d[d["severity"] == panel]
            for b in ["<2 hr/night", "2-4 hr/night", "4+ hr/night"]:
                r = curve_rows(sub[sub["cpap_bin"] == b], phe, pheno, b, {"panel": panel, "group": b})
                if r is not None:
                    cpap_parts.append(r)

        summ.append(dict(phecode=phe, phenotype=pheno, category=cat,
                         n_atrisk=len(d), n_events=int((d["etype"] == 1).sum()),
                         severe_hr=round(float(sev_hr.get(phe, np.nan)), 3),
                         sig_contrasts=sig_by.get(phe, ""),
                         osa_control=phe.startswith("327.3")))

    pd.concat(long_parts, ignore_index=True).to_parquet(OUTDIR / "survival_long.parquet", index=False)
    pd.concat(sev_parts, ignore_index=True).to_csv(OUTDIR / "curve_severity.csv", index=False)
    pd.concat(cpap_parts, ignore_index=True).to_csv(OUTDIR / "curve_cpap.csv", index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUTDIR / "fdr_phecodes.csv", index=False)

    print(f"phecodes exported: {len(summ)} (osa_control flagged: {int(summ['osa_control'].sum())})")
    print(f"  survival_long.parquet rows: {sum(len(p) for p in long_parts):,}")
    print(f"  curve_severity.csv rows: {sum(len(p) for p in sev_parts):,}")
    print(f"  curve_cpap.csv rows: {sum(len(p) for p in cpap_parts):,}")
    print(f"  -> {OUTDIR}")
    U.log_milestone(f"Step 11 survival export: per-patient + plot-ready CIF tables (severity & CPAP "
                    f"strata) for {len(summ)} FDR-significant phecodes -> survival_curves/.")


if __name__ == "__main__":
    main()
