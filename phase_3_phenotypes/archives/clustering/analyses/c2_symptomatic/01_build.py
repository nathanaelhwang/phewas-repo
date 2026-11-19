"""Build the c2_symptomatic patient-level feature table (maximal OSA cohort).

Reuses the validated D:\\phewas_data\\clustering\\c2_symptomatic.csv (74,433 OSA patients x 90
ordinal/binary/one-hot questionnaire features, MRN key) built in prior work, and attaches the
AHI reference columns (index/max) from the c1 build so AHI can be used as an external comparison.

Outputs (clustering/data/c2_symptomatic/):
  c2_features_maximal.csv   MRN, index_ahi, max_ahi, <90 features>
  c2_outlier_report.csv     step-1 z-score summary (bounded items -> no clinical range rules)
  c2_build_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from qc import zscore_outlier_report          # noqa: E402

SRC = Path(r"D:\phewas_data\clustering\c2_symptomatic.csv")
C1 = Path(__file__).resolve().parents[2] / "data" / "c1_physiologic" / "c1_features_maximal.csv"
OUT = Path(__file__).resolve().parents[2] / "data" / "c2_symptomatic"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC, dtype={"MRN": str})
    df["MRN"] = df["MRN"].str.zfill(12)
    feats = [c for c in df.columns if c != "MRN"]

    ref = pd.read_csv(C1, usecols=["MRN", "index_ahi", "max_ahi"], dtype={"MRN": str})
    ref["MRN"] = ref["MRN"].str.zfill(12)
    out = df.merge(ref, on="MRN", how="left")
    out = out[["MRN", "index_ahi", "max_ahi"] + feats]
    out.to_csv(OUT / "c2_features_maximal.csv", index=False)

    zscore_outlier_report(out, feats).to_csv(OUT / "c2_outlier_report.csv", index=False)

    report = {
        "source": str(SRC), "n_rows_out": len(out), "n_features": len(feats),
        "features": feats,
        "ahi_ref_missing": int(out["max_ahi"].isna().sum()),
        "note": "questionnaire items are ordinal/binary/one-hot and bounded; no clinical range "
                "cleaning applied. NO log1p at reduce time. Cohort already OSA (max AHI>=5), "
                "patient-level, from the prior c2 build.",
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
    }
    (OUT / "c2_build_report.json").write_text(json.dumps(report, indent=2))
    print(f"c2 built: {len(out)} OSA patients x {len(feats)} features "
          f"(AHI ref missing {report['ahi_ref_missing']})")


if __name__ == "__main__":
    main()
