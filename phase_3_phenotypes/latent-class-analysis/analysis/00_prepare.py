"""Stage 0 -- build the model matrices for all three domains.

Reads only from ``latent-class-analysis/data`` (treated as read-only) and writes
prepared matrices to ``work/`` plus a JSON prep report per domain to ``results/``.
Nothing here is a model fit; this script is cheap and safe to re-run.

Patient identifiers are carried through as a key column so assignments can be
written back, but are never printed or summarised.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import prep  # noqa: E402
from common.config import DATA, RESULTS, WORK  # noqa: E402

REPORT: dict = {}


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(WORK / f"{name}.parquet", index=False)
    print(f"    wrote work/{name}.parquet  {df.shape}")


# ============================================================ c1 physiologic
print("[c1_physiologic] latent profile analysis inputs")
c1_all = pd.read_csv(DATA / "c1_physiologic" / "c1_features_maximal.csv", dtype={"MRN": str})
assert c1_all.MRN.is_unique, "c1 must be one row per patient (index study)"

excl = c1_all.STUDY_DEVICE.isin(prep.C1_EXCLUDE_DEVICES)
c1 = c1_all.loc[~excl].reset_index(drop=True)
print(f"    device QC: excluded {int(excl.sum()):,} studies on "
      f"{prep.C1_EXCLUDE_DEVICES} -> {len(c1):,} retained")

Z, r = prep.transform_c1(c1, prep.C1_FEATURES)
c1_model = pd.DataFrame(Z, columns=prep.C1_FEATURES)
c1_model.insert(0, "MRN", c1.MRN.to_numpy())
_save(c1_model, "c1_model")

# Carried alongside the model matrix, never modelled: AHI is the distal outcome,
# and device/era/supine-availability are the artifact checks. STUDY_DEVICE is
# near-collinear with calendar era, so the year is carried with it and the two are
# treated as one confound rather than two.
c1_aux = c1[["MRN", "STUDY_DATE", "STUDY_DEVICE", "index_ahi", "max_ahi"]].copy()
c1_aux["study_year"] = pd.to_datetime(c1_aux.STUDY_DATE, errors="coerce").dt.year
c1_aux["supine_ratio_observed"] = c1["supine_nonsupine_AHI_ratio"].notna()
c1_aux["index_ahi_lt5"] = c1_aux.index_ahi < 5
_save(c1_aux, "c1_aux")

r["n_patients"] = int(len(c1))
r["n_excluded_by_device_qc"] = int(excl.sum())
r["excluded_devices"] = prep.C1_EXCLUDE_DEVICES
r["dropped_features"] = prep.C1_DROPPED_REASONS
r["device_counts"] = c1.STUDY_DEVICE.value_counts().to_dict()
r["device_x_supine_observed"] = (
    pd.crosstab(c1.STUDY_DEVICE, c1["supine_nonsupine_AHI_ratio"].notna(),
                normalize="index").round(4).to_dict())
r["n_missingness_patterns"] = int(
    c1[prep.C1_FEATURES].isna().astype(int).astype(str).agg("".join, axis=1).nunique())
r["complete_case_n"] = int(c1[prep.C1_FEATURES].notna().all(axis=1).sum())
r["n_index_ahi_lt5"] = int(c1_aux.index_ahi_lt5.sum())
r["odi_ahi_correlation"] = {
    "pearson": round(float(c1[["ODI4PCT"]].join(c1.index_ahi).corr().iloc[0, 1]), 4)}
r["note"] = (
    "supine_nonsupine_AHI_ratio is 73% missing and missing for 100% of WatchPAT "
    "studies, so its missingness is device-determined rather than MAR given the "
    "modelled variables. Retained per user directive; piloting showed it is inert "
    "(dropping it moved the K=2 solution's minimum component SD from 0.0957 to "
    "0.0955 and the smallest class from 36.68% to 36.66%). Script 01 refits without "
    "it and reports the ARI. ODI4PCT correlates 0.961 with index AHI, so script 01 "
    "also fits a co-primary model without it.")
REPORT["c1_physiologic"] = r
print(f"    {len(c1):,} patients x {len(prep.C1_FEATURES)} features "
      f"({', '.join(prep.C1_FEATURES)})")
print(f"    dropped: {', '.join(prep.C1_DROPPED_REASONS)}")
print(f"    {r['n_missingness_patterns']} missingness patterns; "
      f"max |skew| {max(abs(v) for v in r['skew_before'].values()):.1f} -> "
      f"{max(abs(v) for v in r['skew_after'].values()):.1f} after Yeo-Johnson")

# ============================================================ c2 symptomatic
print("\n[c2_symptomatic] mixed-mode latent class cluster analysis inputs")
c2_raw = pd.read_csv(DATA / "c2_symptomatic" / "c2_features_maximal.csv", dtype={"MRN": str})
assert c2_raw.MRN.is_unique, "c2 must be one row per patient"

c2, sentinel_report = prep.clean_c2_sentinels(c2_raw)
for c, d in sentinel_report["sentinel_9_to_nan"].items():
    print(f"    sentinel QC: {c} value 9 -> NaN ({d['n_9_nulled']:,} rows; "
          f"only {d['n_8']} rows have the value 8 below it)")
for c, d in sentinel_report["implausible_to_nan"].items():
    print(f"    range QC: {c} outside {d['range']} -> NaN ({d['n_nulled']} rows)")

scales, groups, r2 = prep.build_c2_scales(c2)
scales, levels = prep.encode_for_stepmix(scales, groups)
scales.insert(0, "MRN", c2.MRN.to_numpy())
_save(scales, "c2_model_scales")

# sensitivity arm: every original questionnaire column, with the one-hot blocks
# still collapsed and the same type rule applied (both fixes are corrections, not
# preferences, so they hold in every arm).
item_cols = [c for c in c2.columns if c not in ("MRN", "index_ahi", "max_ahi")
             and not c.startswith(("BEDTIME_", "WAKETIME_"))]
items = c2[item_cols].copy()
items["BEDTIME_band"] = scales["BEDTIME_band"].to_numpy()
items["WAKETIME_band"] = scales["WAKETIME_band"].to_numpy()
item_groups, item_meta = prep.assign_measurement_models(
    items, list(items.columns), force_binary=prep.C2_SINGLE_BINARY + prep.C2_CC)
for c in ("BEDTIME_band", "WAKETIME_band"):
    for g in item_groups.values():
        if c in g:
            g.remove(c)
item_groups["multinoulli"] += ["BEDTIME_band", "WAKETIME_band"]
items, item_levels = prep.encode_for_stepmix(items, item_groups)
items.insert(0, "MRN", c2.MRN.to_numpy())
_save(items, "c2_model_items")
(RESULTS / "c2_measurement_models.csv").write_text(
    pd.DataFrame(r2["measurement_models"]).to_csv(index=False))

c2_aux = c2[["MRN", "index_ahi", "max_ahi"]].copy()
c2_aux["form_pattern"] = prep.c2_form_pattern(c2).to_numpy()
c2_aux["n_blocks_present"] = c2_aux.form_pattern.str.count("1")
# STUDY_DATE lives only in c1; join it so the form-version check can be read as an
# era effect rather than an unexplained pattern
c2_aux = c2_aux.merge(c1_aux[["MRN", "study_year"]], on="MRN", how="left")
_save(c2_aux, "c2_aux")

fp = c2_aux.form_pattern.value_counts()
r2["n_patients"] = int(len(c2))
r2["indicator_groups"] = {k: v for k, v in groups.items()}
r2["multinoulli_levels"] = levels
r2["sentinel_cleaning"] = sentinel_report
r2["n_form_patterns"] = int(fp.size)
r2["top_form_pattern_share"] = round(float(fp.iloc[0] / len(c2)), 4)
r2["top5_form_pattern_share"] = round(float(fp.head(5).sum() / len(c2)), 4)
r2["n_item_level_columns"] = int(items.shape[1] - 1)
r2["item_groups"] = {k: len(v) for k, v in item_groups.items()}
r2["note"] += (
    " Missingness is block-structured by intake-form version (5 patterns cover "
    f"{r2['top5_form_pattern_share']:.1%} of the cohort). Script 02 reports Cramer's V "
    "of class x form_pattern and class x study_year as an explicit check that the "
    "classes are symptom phenotypes rather than questionnaire-version cohorts.")
REPORT["c2_symptomatic"] = r2
print(f"    {len(c2):,} patients; scale-score arm {scales.shape[1]-1} indicators "
      f"({len(groups['gaussian'])} gaussian / {len(groups['bernoulli'])} bernoulli / "
      f"{len(groups['multinoulli'])} multinoulli), item arm {r2['n_item_level_columns']}")
print(f"    gaussian block: {', '.join(groups['gaussian'])}")

# ============================================================ c3 comorbidities
print("\n[c3_comorbidities] latent class analysis inputs")
c3 = pd.read_csv(DATA / "c3_comorbidities" / "c3_features_maximal.csv", dtype={"MRN": str})
assert c3.MRN.is_unique, "c3 must be one row per patient"
labels = pd.read_csv(DATA / "c3_comorbidities" / "c3_phecode_labels.csv",
                     dtype={"phecode": str}).dropna(subset=["phecode"])
feat3 = [c for c in c3.columns if c not in ("MRN", "index_ahi", "max_ahi")]

rolled, roll_report = prep.rollup_phecodes(c3, feat3)
keep, meta = prep.select_c3_indicators(rolled, labels)
meta.to_csv(RESULTS / "c3_indicator_selection.csv", index=False)

c3_model = rolled[keep].copy()
c3_model.insert(0, "MRN", c3.MRN.to_numpy())
_save(c3_model, "c3_model")

c3_aux = c3[["MRN", "index_ahi", "max_ahi"]].copy()
# utilisation proxy computed over ALL 656 original codes, not just the kept ones
c3_aux["n_phecodes_all"] = prep.c3_utilisation_index(c3, feat3).to_numpy()
c3_aux["n_phecodes_kept"] = c3_model[keep].sum(axis=1).to_numpy()
_save(c3_aux, "c3_aux")

r3 = {
    "n_patients": int(len(c3)),
    "n_codes_raw": len(feat3),
    "n_roots_after_rollup": roll_report["n_after"],
    "n_indicators_kept": len(keep),
    "min_prevalence": prep.C3_MIN_PREVALENCE,
    "dropped_categories": sorted(prep.C3_DROP_CATEGORIES),
    "prevalence_of_kept": {
        "min": round(float(c3_model[keep].mean().min()), 4),
        "median": round(float(c3_model[keep].mean().median()), 4),
        "max": round(float(c3_model[keep].mean().max()), 4),
    },
    "kept_by_category": meta[meta.kept].category.value_counts().to_dict(),
    "utilisation_index": {
        "median_all_codes": int(c3_aux.n_phecodes_all.median()),
        "p90_all_codes": int(c3_aux.n_phecodes_all.quantile(0.90)),
    },
    "note": (
        "Roll-up is a logical OR onto the integer phecode. The pyPheWAS build maps "
        "each ICD to exactly one phecode with no hierarchy roll-up, so parent and "
        "child codes are largely disjoint (empirical P(parent=1|child=1) = 0.27): "
        "roll-up de-fragments a condition rather than double-counting it. "
        "n_phecodes_all is a healthcare-contact proxy and is never a model input."
    ),
}
REPORT["c3_comorbidities"] = r3
print(f"    {len(c3):,} patients; {len(feat3)} codes -> {roll_report['n_after']} roots "
      f"-> {len(keep)} chronic-disease indicators (prev >= {prep.C3_MIN_PREVALENCE:.0%})")

# ==================================================================== cohorts
shared = set(c1.MRN) & set(c2.MRN) & set(c3.MRN)
declared = set(pd.read_csv(DATA / "shared" / "shared_mrns.csv", dtype={"MRN": str}).MRN)
pd.DataFrame({"MRN": sorted(shared)}).to_parquet(WORK / "shared_mrns.parquet", index=False)
REPORT["cohorts"] = {
    "c1_maximal": int(len(c1)), "c2_maximal": int(len(c2)), "c3_maximal": int(len(c3)),
    "shared_intersection": int(len(shared)),
    "prior_shared_intersection": int(len(declared)),
    "lost_to_c1_device_qc": int(len(declared - shared)),
    "note": ("The shared cohort is smaller than the prior run's 71,433 purely because "
             "the c1 device QC removed patients whose only study was on a device with "
             "a corrupted or absent oximetry block. Every prior shared patient not "
             "retained here is accounted for by that exclusion."),
}
assert declared - shared <= set(c1_all.loc[excl, "MRN"]), \
    "shared-cohort shrinkage must be fully explained by the c1 device exclusion"
print(f"\n[cohorts] maximal {len(c1):,} / {len(c2):,} / {len(c3):,}; "
      f"shared {len(shared):,} (prior {len(declared):,}; "
      f"{len(declared - shared):,} dropped by c1 device QC)")

(RESULTS / "00_prep_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
print(f"\nwrote {RESULTS / '00_prep_report.json'}")
