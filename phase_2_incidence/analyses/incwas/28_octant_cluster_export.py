"""Cluster-distribution export for the website's octant-phenotype section (D, E, F).

The site already shows the 17 point estimates in `octant_clinical_signature.csv`. That file
is a bare octant x variable matrix of means, so the page cannot say how many patients each
mean rests on, how spread the variable is, or how much of the cohort was missing it. This
writes the same 17 measures in long form with n, coverage, dispersion and quartiles, plus
the cohort reference and the standardised difference from it, and adds the demographic /
follow-up / severity measures that were already available from safe aggregate sources.

Every statistic is reproduced from the same sources and with the same filters that stage 10
used (`c1_features_maximal` minus the three corrupted study devices, `c2_model_scales`,
`c3_model` / `c3_aux`), so the means here must equal the published signature matrix -- that
equality is asserted, not assumed.

Two measures the site asked for are NOT here, deliberately:

  * **CPAP initiation and adherence.** `followup/cpap_usage.parquet` holds 92,373 of the
    157,025 spine patients, and nothing in the data dictionary establishes whether an
    absent row means "never started" or "not captured by the PopulationCohort scan". Its
    `mean_pap_usage` units are flagged UNCONFIRMED in the extract script that built it. A
    proportion needs a denominator that means something, so it is left out rather than
    published with an unverifiable one.
  * **Race/ethnicity.** Excluded per the disclosure rules; one cell in the octant Table 1
    falls below the threshold and the breakdown has not been approved for release.

PHI: patient-level frames are read into memory and never written, printed or sampled. All
output is octant-level. Counts below DISCLOSURE_MIN are blanked with a `suppressed` flag.

Usage:  python 28_octant_cluster_export.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent-class-analysis"
                       / "analysis"))
import incwas_utils as U  # noqa: E402
import _pheno_engine as P  # noqa: E402
from common import prep  # noqa: E402
from common.config import DATA, RESULTS as LCA_RESULTS, WORK  # noqa: E402

OUT = P.OUT / "website_exports"
PH = LCA_RESULTS / "phenotypes"
SPINE = U.PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet"
DISCLOSURE_MIN = 11
OVERALL = "overall"                     # the cohort reference row, not an octant

BITS = {"mild-all": (0, 0, 0), "hypoxemia-predominant": (1, 0, 0),
        "symptom-predominant": (0, 1, 0), "comorbidity-predominant": (0, 0, 1),
        "hypoxemic-symptomatic": (1, 1, 0), "hypoxemic-comorbid": (1, 0, 1),
        "symptomatic-comorbid": (0, 1, 1), "high-all": (1, 1, 1)}
ORDER = list(BITS)
AXIS_NAMES = ("physiologic", "symptom", "comorbidity")

# (column, metric_id, label, domain, type, unit) -- the first 17 are stage 10's, in order
METRICS = [
    ("ODI4PCT", "odi4pct", "ODI 4% (events/h)", "physiologic", "continuous", "events/h"),
    ("MINIMUM_SPO2", "min_spo2", "minimum SpO2 (%)", "physiologic", "continuous", "%"),
    ("MEAN_SPO2", "mean_spo2", "mean SpO2 (%)", "physiologic", "continuous", "%"),
    ("T90%", "t90_fraction", "T90 (fraction)", "physiologic", "continuous", "fraction"),
    ("ESS_total", "ess_total", "Epworth total", "symptom", "continuous", "points"),
    ("ISI_total", "isi_total", "Insomnia Severity Index", "symptom", "continuous", "points"),
    ("FOSQ_impair", "fosq_impair", "FOSQ impairment", "symptom", "continuous", "points"),
    ("PHQ2_total", "phq2_total", "PHQ-2", "symptom", "continuous", "points"),
    ("STOP_total", "stop_total", "STOP score", "symptom", "continuous", "points"),
    ("n_phecodes_all", "n_phecodes_distinct", "distinct phecodes", "comorbidity",
     "continuous", "count"),
    ("278", "cond_obesity", "obesity", "comorbidity", "binary", "proportion"),
    ("272", "cond_hyperlipidemia", "hyperlipidemia", "comorbidity", "binary", "proportion"),
    ("401", "cond_hypertension", "hypertension", "comorbidity", "binary", "proportion"),
    ("250", "cond_impaired_fasting_glucose", "impaired fasting glucose", "comorbidity",
     "binary", "proportion"),
    ("300", "cond_anxiety", "anxiety disorders", "comorbidity", "binary", "proportion"),
    ("530", "cond_gerd", "GERD", "comorbidity", "binary", "proportion"),
    ("index_ahi", "index_ahi", "index AHI (events/h)", "external", "continuous",
     "events/h"),
    # --- additions from safe aggregate sources ---------------------------------
    ("age", "age_years", "age", "demographic", "continuous", "years"),
    ("bmi", "bmi", "BMI", "demographic", "continuous", "kg/m2"),
    ("female", "female", "female", "demographic", "binary", "proportion"),
    ("sev_none", "ahi_severity_none", "AHI < 5 (no OSA)", "external", "binary",
     "proportion"),
    ("sev_mild", "ahi_severity_mild", "AHI 5-15 (mild)", "external", "binary",
     "proportion"),
    ("sev_moderate", "ahi_severity_moderate", "AHI 15-30 (moderate)", "external",
     "binary", "proportion"),
    ("sev_severe", "ahi_severity_severe", "AHI 30+ (severe)", "external", "binary",
     "proportion"),
    ("followup_years", "followup_years", "incwas follow-up", "follow-up", "continuous",
     "years"),
    ("enc_per_year", "enc_per_year", "baseline encounter rate", "utilisation",
     "continuous", "encounters/year"),
]
REQUIRED_17 = [m[1] for m in METRICS[:17]]


def build() -> pd.DataFrame:
    """One row per patient in the shared cohort, one column per metric. In memory only."""
    ph = pd.read_csv(PH / "cross_domain_phenotypes.csv", dtype={"MRN": str})
    c1 = pd.read_csv(DATA / "c1_physiologic" / "c1_features_maximal.csv",
                     dtype={"MRN": str})
    c1 = c1[~c1.STUDY_DEVICE.isin(prep.C1_EXCLUDE_DEVICES)]     # stage 10's filter
    c2 = pd.read_parquet(WORK / "c2_model_scales.parquet")
    c3 = pd.read_parquet(WORK / "c3_model.parquet")
    c3a = pd.read_parquet(WORK / "c3_aux.parquet")[["MRN", "n_phecodes_all"]]

    d = ph[["MRN", "octant"]].merge(
        c1[["MRN", "index_ahi", "ODI4PCT", "MINIMUM_SPO2", "MEAN_SPO2", "T90%"]],
        on="MRN", how="left")
    d = d.merge(c2[["MRN", "ESS_total", "ISI_total", "FOSQ_impair", "PHQ2_total",
                    "STOP_total"]], on="MRN", how="left")
    d = d.merge(c3a, on="MRN", how="left").merge(
        c3[["MRN", "278", "272", "401", "250", "300", "530"]], on="MRN", how="left")

    spine = pd.read_parquet(SPINE)[["id", "age", "sex_M", "bmi"]]
    spine["id"] = spine.id.astype(str)
    d = d.merge(spine, left_on="MRN", right_on="id", how="left").drop(columns=["id"])
    d["female"] = 1.0 - d.sex_M

    f = pd.read_parquet(U.FOLLOWUP_FRAME)[["id", "followup_years", "enc_per_year"]]
    f["id"] = f.id.astype(str)
    d = d.merge(f, left_on="MRN", right_on="id", how="left").drop(columns=["id"])

    # AHI severity from the external index AHI, on the project's standard cut points
    for nm, lo, hi in (("sev_none", 0, 5), ("sev_mild", 5, 15),
                       ("sev_moderate", 15, 30), ("sev_severe", 30, np.inf)):
        d[nm] = np.where(d.index_ahi.isna(), np.nan,
                         ((d.index_ahi >= lo) & (d.index_ahi < hi)).astype(float))
    d["octant"] = pd.Categorical(d.octant, categories=ORDER, ordered=True)
    assert d.octant.notna().all(), "every patient must fall in exactly one octant"
    return d


def stats_for(v: pd.Series, kind: str, n_group: int) -> dict:
    """Aggregate summary of one metric in one group. `v` never leaves this function."""
    x = v.dropna()
    n = len(x)
    out = {"n_nonmissing": n, "coverage_pct": round(100 * n / n_group, 2)}
    if n == 0:
        return {**out, "estimate": np.nan, "mean": np.nan, "sd": np.nan,
                "median": np.nan, "q1": np.nan, "q3": np.nan,
                "numerator": pd.NA, "denominator": pd.NA, "proportion": np.nan}
    if kind == "binary":
        num = int(x.sum())
        return {**out, "estimate": num / n, "mean": np.nan, "sd": np.nan,
                "median": np.nan, "q1": np.nan, "q3": np.nan,
                "numerator": num, "denominator": n, "proportion": num / n}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {**out, "estimate": float(x.mean()), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)), "median": float(med), "q1": float(q1),
            "q3": float(q3), "numerator": pd.NA, "denominator": pd.NA,
            "proportion": np.nan}


def distributions(d: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = []
    for col, mid, label, dom, kind, unit in METRICS:
        ref = stats_for(d[col], kind, len(d))
        # cohort SD is the yardstick for both kinds: sqrt(p(1-p)) for a proportion is the
        # SD of the same 0/1 column, so one rule covers the table
        scale = (np.sqrt(ref["estimate"] * (1 - ref["estimate"])) if kind == "binary"
                 else ref["sd"])
        for grp in ORDER + [OVERALL]:
            sub = d if grp == OVERALL else d[d.octant == grp]
            s = ref if grp == OVERALL else stats_for(sub[col], kind, len(sub))
            sd_diff = (np.nan if not scale or not np.isfinite(scale) or scale == 0
                       or not np.isfinite(s["estimate"])
                       else (s["estimate"] - ref["estimate"]) / scale)
            rows.append({"octant": grp, "metric_id": mid, "metric_label": label,
                         "domain": dom, "metric_type": kind, "unit": unit, **s,
                         "cohort_estimate": ref["estimate"],
                         "standardized_difference": sd_diff})
    t = pd.DataFrame(rows)
    for c in ("estimate", "mean", "sd", "median", "q1", "q3", "proportion",
              "cohort_estimate", "standardized_difference"):
        t[c] = t[c].astype(float).round(6)
    mask = pd.Series(False, index=t.index)
    for c in ("n_nonmissing", "numerator", "denominator"):
        m = t[c].notna() & (t[c].astype("Float64") < DISCLOSURE_MIN)
        t.loc[m, c] = pd.NA
        mask |= m
        t[c] = t[c].astype("Int64")
    t["suppressed"] = mask
    return t, int(mask.sum())


def summary(d: pd.DataFrame) -> pd.DataFrame:
    o = pd.read_csv(PH / "octants.csv")
    cuts = json.loads((PH / "octant_cuts.json").read_text())
    o = o.set_index("octant").reindex(ORDER)
    rows = []
    for oc in ORDER:
        b = BITS[oc]
        hi = [AXIS_NAMES[i] for i, x in enumerate(b) if x]
        rows.append({
            "octant": oc, "label": oc,
            "glyph": "".join("■" if x else "□" for x in b),
            "axes_high": " + ".join(hi) if hi else "none",
            "physiology_high": bool(b[0]), "symptom_high": bool(b[1]),
            "comorbidity_high": bool(b[2]),
            # recomputed from n rather than taken from octants.csv's 4-dp fraction, so the
            # column sums to 100 instead of 99.99
            "n": int(o.loc[oc, "n"]),
            "pct": round(100 * float(o.loc[oc, "n"]) / float(o["n"].sum()), 6),
            "median_ahi": float(o.loc[oc, "median_ahi"]),
            "median_phecodes": float(o.loc[oc, "median_codes"]),
            "score_c1_cut": cuts["score_c1"], "score_c2_cut": cuts["score_c2"],
            "score_c3_cut": cuts["score_c3"],
            "classification_coverage_pct": 100.0,
            "cut_point_basis": "cohort-derived median of the gradient score; NOT a "
                               "clinical threshold and does not transfer"})
    return pd.DataFrame(rows)


def correlations() -> tuple[pd.DataFrame, float]:
    ph = pd.read_csv(PH / "cross_domain_phenotypes.csv",
                     usecols=["score_c1", "score_c2", "score_c3"])
    c = ph.corr(method="spearman")
    published = json.loads((PH / "phenotype_summary.json").read_text())["score_spearman"]
    dev = max(abs(c.loc[a, b] - published[a][b]) for a in c.index for b in c.columns)
    rows = [{"score_x": a, "score_y": b, "spearman_rho": round(float(c.loc[a, b]), 6)}
            for a in c.index for b in c.columns]
    return pd.DataFrame(rows), float(dev)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    print(f"shared phenotype cohort: {len(d):,} patients", flush=True)

    dist, n_sup = distributions(d)
    summ = summary(d)
    corr, corr_dev = correlations()

    # the 17 published means must reproduce exactly, or the sources have drifted
    sig = pd.read_csv(PH / "octant_clinical_signature.csv", index_col=0)
    lab = {m[2]: m[1] for m in METRICS[:17]}
    chk = dist[dist.octant != OVERALL].set_index(["metric_id", "octant"])["estimate"]
    dev = max(abs(chk.loc[(lab[r], c)] - sig.loc[r, c])
              for r in sig.index for c in sig.columns)
    print(f"max |export - published signature matrix| = {dev:.2e} "
          f"(published file is rounded to 3 dp)")
    assert dev < 5e-4, "octant means no longer reproduce the published signature"
    print(f"max |recomputed - published Spearman| = {corr_dev:.2e}")

    dist.to_csv(OUT / "octant_cluster_distributions.csv", index=False, encoding="utf-8")
    summ.to_csv(OUT / "octant_cluster_summary.csv", index=False, encoding="utf-8")
    corr.to_csv(OUT / "octant_score_correlations.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"key": k, "value": str(v)} for k, v in {
        "n_shared_cohort": len(d), "metrics": len(METRICS),
        "required_17_present": sorted(set(REQUIRED_17)) == sorted(
            set(dist.metric_id) & set(REQUIRED_17)),
        "groups_per_metric": f"8 octants + '{OVERALL}' cohort reference",
        "suppressed_cells_distributions": n_sup,
        "signature_max_dev": dev, "spearman_max_dev": corr_dev,
        "standardized_difference_def":
            "(group estimate - cohort estimate) / cohort SD; for a binary metric the "
            "cohort SD is sqrt(p(1-p)). This is a difference from the cohort the group "
            "belongs to, NOT a two-independent-group SMD.",
        "omitted_fields": "CPAP initiation, CPAP adherence distribution, race/ethnicity",
    }.items()]).to_csv(OUT / "_cluster_export_audit.csv", index=False)

    print(f"\noctant_cluster_distributions.csv {len(dist):,} rows "
          f"({n_sup} suppressed count cells)")
    print(f"octant_cluster_summary.csv       {len(summ):,} rows, n sums to "
          f"{int(summ.n.sum()):,}, pct sums to {summ.pct.sum():.2f}%")
    print(f"octant_score_correlations.csv    {len(corr):,} rows")
    print("\ncoverage by metric (% of the 70,880 shared cohort):")
    print(dist[dist.octant == OVERALL][["metric_id", "coverage_pct"]]
          .to_string(index=False))
    U.log_milestone(
        f"Octant website export (D/E/F): {len(dist)} cluster-distribution rows over "
        f"{len(METRICS)} metrics x 8 octants + cohort reference, plus summary and score "
        f"correlations -> phenotype_exposure/website_exports/.")


if __name__ == "__main__":
    main()
