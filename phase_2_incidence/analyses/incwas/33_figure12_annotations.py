"""Aggregate median index AHI and existing M4 HRs for the nine Figure 12 panels.

Run through the reviewed local output-withholding runner. Medians use the exact
outcome-specific baseline risk sets used by script32; no patient rows are saved.
No new Cox model is fitted by this script.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incwas_utils as U
import _cox_engine as E
import _pheno_engine as P

spec = importlib.util.spec_from_file_location("curve_inference", Path(__file__).with_name("32_octant_curve_inference.py"))
curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curves)


def main():
    osa, no_osa, cohort_audit = curves.load_cohorts()
    prev_osa = curves.set_cohort(osa)
    prev_no_osa = curves.set_cohort(no_osa)
    events = pd.read_parquet(U.EVENTS, columns=["id", "phecode", "second_post_date"])
    events = events[events.phecode.astype(str) == "428.2"]
    exclusion = E.exclude_map()
    osa_surv = E.build_survival("428.2", events, osa, prev_osa, exclusion)
    no_osa_surv = E.build_survival("428.2", events, no_osa, prev_no_osa, exclusion)
    model_path = P.OUT / "website_exports/octant_cif_metadata.csv"
    metadata = pd.read_csv(model_path, dtype={"outcome":str})
    metadata = metadata[(metadata.level == "phecode") & (metadata.outcome == "428.2")]
    annotations = []
    for octant in P.OCTANTS:
        group = osa_surv[osa_surv.octant == octant]
        model = metadata[metadata.octant == octant].iloc[0]
        annotations.append({"octant":octant,"median_ahi":float(group.AHI.median()),
                            "median_ahi_unit":"events/hour","ahi_n":int(group.AHI.notna().sum()),
                            "m4_hr":float(model.hr_m4),"m4_ci_low":float(model.ci_low),
                            "m4_ci_high":float(model.ci_high),
                            "m4_source":"existing original-cohort one-versus-other-seven M4 model",
                            "m4_hr_available":True})
    annotations.append({"octant":"no-osa","median_ahi":float(no_osa_surv.AHI.median()),
                        "median_ahi_unit":"events/hour","ahi_n":int(no_osa_surv.AHI.notna().sum()),
                        "m4_hr":None,"m4_ci_low":None,"m4_ci_high":None,
                        "m4_source":"No matching No OSA versus eight-phenotype pooled M4 model in original octant results",
                        "m4_hr_available":False})
    # Reconcile median denominators against the exact previously exported curve cohort.
    public = U.PROJECT / "website/public/data/phenotype-survival/phecode/428-2.json"
    expected = json.loads(public.read_text())
    panels = {panel["octant"]:panel for panel in expected["panels"]}
    for annotation in annotations:
        risk = (expected["no_osa"]["risk_table"] if annotation["octant"] == "no-osa"
                else panels[annotation["octant"]]["risk_table"]["focal"])
        assert annotation["ahi_n"] == risk[0]["n_at_risk"]
    curves.save_json(U.PROJECT / "tmp/phewas_260919d/figure12_annotations.json",
                     {"outcome_id":"428.2","outcome_name":"Heart failure NOS",
                      "median_ahi_definition":"Median index AHI among each plotted focal group's heart-failure-free baseline risk set after the control-exclusion-family washout",
                      "m4_note":cohort_audit["model_note"],
                      "median_denominators_match_all_nine_curves":True,
                      "m4_source":"results/incwas_results/phenotype_exposure/website_exports/octant_cif_metadata.csv",
                      "annotations":annotations})


if __name__ == "__main__":
    main()
