"""Build the c1_physiologic patient-level feature table (maximal OSA cohort).

Cohort  : patients with any sleep study of AHI >= 5 (OSA present).
Grain   : one row per patient = the patient's earliest (index) study -- the same index that
          anchors the c2 (intake) and c3 (pre-index phecode) tables.
Features: 11 physiologic desaturation/oximetry/pulse metrics + 2 engineered ratios
          (T90_over_AHI, supine_nonsupine_AHI_ratio).  Raw AHI / supine-AHI / non-supine-AHI /
          analysis-duration are NOT inputs; AHI is retained only as an external reference.

Outputs (clustering/data/c1_physiologic/):
  c1_features_maximal.csv   MRN, STUDY_DATE, STUDY_DEVICE, index_ahi, max_ahi, <13 features>
  c1_outlier_report.csv     step-1 z-score / extreme-outlier summary
  c1_build_report.json      cleaning rules fired + cohort accounting
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import prep                                   # noqa: E402
from qc import zscore_outlier_report          # noqa: E402

SRC = Path(r"D:\phewas_data\clustering\c1_physiologic.csv")
OUT = Path(__file__).resolve().parents[2] / "data" / "c1_physiologic"
MIN_AHI = 5.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC, low_memory=False, dtype={"MRN": str})
    df["MRN"] = df["MRN"].str.zfill(12)
    n_studies = len(df)

    cleaned, clean_report = prep.clean_c1_raw(df)
    cleaned = prep.engineer_c1(cleaned)

    # study-date: treat pre-1985 epoch placeholders as missing so 'earliest' is meaningful
    sd = pd.to_datetime(cleaned["STUDY_DATE"], errors="coerce")
    placeholder = sd.dt.year < 1985
    sd = sd.mask(placeholder)
    cleaned["_sd"] = sd
    n_placeholder = int(placeholder.sum())

    ahi = cleaned["AHI"]
    max_ahi = cleaned.groupby("MRN")["AHI"].transform("max")
    osa = cleaned[max_ahi >= MIN_AHI].copy()
    n_osa_patients = osa["MRN"].nunique()

    # one row per patient = earliest valid study (NaT dates sorted last, stable for ties).
    # drop_duplicates keeps the first *row* -- unlike GroupBy.first(), which would take the
    # first non-null of each column and fabricate a cross-study row for multi-study patients.
    osa = osa.sort_values(["MRN", "_sd"], na_position="last", kind="stable")
    idx = osa.drop_duplicates(subset="MRN", keep="first").reset_index(drop=True)
    idx = idx.rename(columns={"AHI": "index_ahi"})
    idx["max_ahi"] = idx["MRN"].map(cleaned.groupby("MRN")["AHI"].max())

    keep = ["MRN", "STUDY_DATE", "STUDY_DEVICE", "index_ahi", "max_ahi"] + prep.C1_FEATS
    out = idx[keep].copy()
    out.to_csv(OUT / "c1_features_maximal.csv", index=False)

    rep = zscore_outlier_report(out, prep.C1_FEATS)
    rep.to_csv(OUT / "c1_outlier_report.csv", index=False)

    report = {
        "source": str(SRC), "min_ahi": MIN_AHI,
        "n_studies_raw": n_studies, "n_study_date_placeholder": n_placeholder,
        "n_osa_patients": int(n_osa_patients), "n_rows_out": len(out),
        "model_features": prep.C1_FEATS,
        "dropped_from_inputs": prep.RATIO_INPUTS + ["ANALYSIS_DURATION"],
        "engineered": {
            "T90_over_AHI": "T90% / AHI",
            "supine_nonsupine_AHI_ratio": "AHI_SUPINE / AHI_NONSUPINE for supine% in [25,75] & non-supine AHI>0",
        },
        "engineered_coverage": {
            "T90_over_AHI_pct_present": round(float(out["T90_over_AHI"].notna().mean()), 4),
            "supine_ratio_pct_present": round(float(out["supine_nonsupine_AHI_ratio"].notna().mean()), 4),
        },
        "index_ahi_lt5_count": int((out["index_ahi"] < 5).sum()),   # earliest study non-OSA
        "cleaning": clean_report,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
    }
    (OUT / "c1_build_report.json").write_text(json.dumps(report, indent=2))
    print(f"c1 built: {len(out)} OSA patients x {len(prep.C1_FEATS)} features "
          f"(supine-ratio present {report['engineered_coverage']['supine_ratio_pct_present']:.1%}, "
          f"index-study non-OSA {report['index_ahi_lt5_count']})")


if __name__ == "__main__":
    main()
