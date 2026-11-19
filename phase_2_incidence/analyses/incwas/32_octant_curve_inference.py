"""Add CI, 3-year log-rank, and No OSA aggregates without changing model results.

Run via the local output-withholding runner. Only reviewed derived inputs are
read; row-level frames stay in memory and output consists only of aggregate
curves, suppressed annual counts, and validation metadata.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incwas_utils as U
import _cox_engine as E
import _pheno_engine as P
from _aj_inference import aalen_johansen_ci, cause_specific_logrank

OUT = P.OUT / "website_exports"
HORIZON = 3.
RISK_TIMES = [0., 1., 2., 3.]
THRESHOLD = 11


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":"))+"\n", encoding="utf-8")


def load_cohorts():
    osa = P.assign_octants(P.load_exposure(with_archetype=False)).reset_index(drop=True)
    followup = pd.read_parquet(U.FOLLOWUP_FRAME)
    no_osa = followup[followup.AHI.notna() & (followup.AHI < 5)].copy().reset_index(drop=True)
    no_osa["id"] = no_osa["id"].astype(str)
    no_osa["idx"] = pd.to_datetime(no_osa["idx"])
    no_osa["censor_date"] = pd.to_datetime(no_osa["censor_date"])
    original_n = len(osa)
    original_overlap = len(set(osa.id) & set(no_osa.id))
    missing_ahi = int(osa.AHI.isna().sum())
    osa = osa[osa.AHI.notna() & (osa.AHI >= 5)].copy().reset_index(drop=True)
    overlap = len(set(osa.id) & set(no_osa.id))
    audit = {"osa_n": len(osa), "no_osa_n": len(no_osa), "overlap_n": overlap,
             "original_phenotype_n":original_n,"original_no_osa_overlap_n":original_overlap,
             "original_missing_ahi_n":missing_ahi,
             "osa_ahi_min": float(osa.AHI.min()), "no_osa_ahi_max": float(no_osa.AHI.max()),
             "no_osa_definition": "AHI < 5 in the existing IncWAS follow-up spine",
             "osa_definition": "Existing three-domain phenotype cohort restricted to AHI >= 5; original octant assignments and cut points retained",
             "model_note": "Adjusted Cox estimates retain the original 25,380-person phenotype cohort. Curves exclude 154 people with AHI < 5 and therefore use a different cohort.",
             "no_osa_requires_three_domain_scores": False}
    if overlap or osa.AHI.isna().any() or (osa.AHI < 5).any():
        save_json(OUT / "octant_inference_cohort_audit.json", audit)
        raise ValueError("Original phenotype cohort is not disjoint from No OSA")
    if no_osa.id.duplicated().any() or osa.id.duplicated().any():
        raise ValueError("Cohort identifiers are not unique")
    return osa, no_osa, audit


def set_cohort(frame):
    E._PREP, E._ID2IX = (frame, {}), {v:i for i,v in enumerate(frame.id)}
    E._PREVIX = None
    return E.prev_index()


def series(surv, withheld=False, allow_sparse=False):
    t, event, death = (surv[c].to_numpy() for c in ("time", "event", "is_death"))
    at, cumulative = P.risk_counts(t, event, RISK_TIMES)
    risk = [{"time_years": year, "n_at_risk": int(n) if n >= THRESHOLD else None,
             "n_events_cum": int(e) if e >= THRESHOLD else None,
             "suppressed": bool(n < THRESHOLD or e < THRESHOLD)}
            for year, n, e in zip(RISK_TIMES, at, cumulative)]
    withheld = withheld or (not allow_sparse and (int(at[0]) < THRESHOLD or int(cumulative[-1]) < THRESHOLD))
    if withheld:
        return None, risk
    grid = P.event_time_grid(t, event, HORIZON)
    f, variance, low, high = aalen_johansen_ci(t, event, death, grid)
    np.testing.assert_allclose(f, P.aalen_johansen(t,event,death,grid), rtol=0, atol=1e-13)
    return {"time_years": grid.tolist(), "cif_pct": (100*f).tolist(),
            "ci_low_pct": (100*low).tolist(), "ci_high_pct": (100*high).tolist()}, risk


def main():
    osa, no_osa, audit = load_cohorts()
    if "--cohort-audit" in sys.argv:
        save_json(OUT / "octant_inference_cohort_audit.json", audit)
        return
    ev = pd.read_parquet(U.EVENTS)
    ev["phecode"] = ev.phecode.astype(str)
    ev = ev[ev.id.isin(set(osa.id) | set(no_osa.id))]
    ev = ev[~ev.phecode.map(P.is_circular_outcome)]
    dim = pd.read_csv(U.PHECODE_DIM, dtype={"phecode":str}, keep_default_na=False)
    old = pd.read_csv(OUT / "octant_cif_metadata.csv", dtype={"outcome":str})
    prev_osa = set_cohort(osa)
    prev_no_osa = set_cohort(no_osa)
    exclusion = E.exclude_map()
    assets = []
    for (level, outcome), rows in old.groupby(["level", "outcome"], sort=False):
        if level == "phecode":
            osa_surv = E.build_survival(outcome, ev[ev.phecode == outcome], osa, prev_osa, exclusion)
            no_surv = E.build_survival(outcome, ev[ev.phecode == outcome], no_osa, prev_no_osa, exclusion)
        else:
            osa_surv = P.system_events(osa, ev, dim, outcome, rule="anynew")
            no_surv = P.system_events(no_osa, ev, dim, outcome, rule="anynew")
        pooled, _ = series(osa_surv)
        panels = []
        for row in rows.itertuples():
            mask = osa_surv.octant == row.octant
            # Match the established public withholding policy for rare focal events.
            withheld = bool(row.suppressed)
            focal, focal_risk = series(osa_surv[mask], withheld)
            rest, rest_risk = series(osa_surv[~mask], withheld)
            # Publishing a disclosed focal count plus its rest would reveal the
            # pooled count and allow subtraction to recover a sparse focal count.
            for point in focal_risk + rest_risk:
                point["n_events_cum"], point["suppressed"] = None, True
            available = focal is not None and rest is not None
            panels.append({"octant": row.octant,
                           "curves": {"focal":focal,"other_seven":rest} if available else None,
                           "risk_table":{"focal":focal_risk,"other_seven":rest_risk},
                           "logrank_p": cause_specific_logrank(osa_surv.time,osa_surv.event,mask) if available else None})
        curve, risk = series(no_surv)
        combined = pd.concat([no_surv,osa_surv], ignore_index=True)
        no_osa_mask = np.arange(len(combined)) < len(no_surv)
        no_osa_payload = {"curve_status":"available" if curve is not None else "withheld_event_count_suppression",
                          "curve":curve,"risk_table":risk,
                          "logrank_p": cause_specific_logrank(combined.time,combined.event,no_osa_mask) if curve is not None and pooled is not None else None}
        if level == "phecode" and outcome == "428.2":
            # User-requested local manuscript figure; not a public website asset.
            manuscript_panels = []
            for octant in P.OCTANTS:
                mask = osa_surv.octant == octant
                focal, _ = series(osa_surv[mask], allow_sparse=True)
                rest, _ = series(osa_surv[~mask], allow_sparse=True)
                manuscript_panels.append({"octant":octant,"curve_status":"available",
                    "curves":{"focal":focal,"other_seven":rest},
                    "low_event_count":bool(((osa_surv.event == 1)&(osa_surv.time <= HORIZON)&mask).sum() < THRESHOLD),
                    "logrank_p":cause_specific_logrank(osa_surv.time,osa_surv.event,mask)})
            manuscript_no_osa = {k:v for k,v in no_osa_payload.items() if k != "risk_table"}
            save_json(U.PROJECT / "tmp/phewas_260919d/figure12_curves.json",
                      {"curve_granularity":"event_time","level":level,"outcome_id":outcome,"outcome_name":str(rows.outcome_name.iloc[0]),
                       "overall":{"curve_status":"available","curve":pooled},
                       "panels":manuscript_panels,"no_osa":manuscript_no_osa,
                       "cohort":audit,"audience":"local manuscript; not a public website asset"})
        _, pooled_risk = series(osa_surv)
        # Retain the website's complementary suppression of pooled event counts.
        for point in pooled_risk:
            point["n_events_cum"], point["suppressed"] = None, True
        assets.append({"level":level,"outcome_id":outcome,"overall":pooled,"overall_risk_table":pooled_risk,
                       "overall_mean_followup_years":float(osa_surv.time.mean()),
                       "panels":panels,"no_osa":no_osa_payload})
    save_json(OUT / "octant_curve_inference.json", {"schema_version":1,"horizon_years":HORIZON,
              "confidence_level":.95,"ci_method":"Tie-aware Greenwood / infinitesimal-jackknife variance; pointwise log-log transform",
              "logrank_method":"Two-sided cause-specific log-rank; competing death censored; administratively censored at 3 years; nominal p-values",
              "comparison":"Each OSA phenotype versus pooled other seven OSA; No OSA versus all eight OSA pooled",
              "cohort":audit,"assets":assets})


if __name__ == "__main__":
    main()
