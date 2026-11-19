"""Build the c3_comorbidities patient-level feature table (maximal OSA cohort).

Feature matrix = patient x PheCode BINARY presence of pre-index diagnoses.
  * Phenome  : pre-study ICD-9/10 from icds_all.csv (ADATE <= earliest STUDY_DATE), the
               established v1 extract (97.5% coverage of the OSA roster).
  * Mapping  : pyPheWAS v1.2 ICD9 + ICD10-beta maps, exact code match with a longest-prefix
               fallback (recovers data codes more specific than the map stem).
  * Cohort   : OSA roster (max AHI >= 5, from the c1 build) intersected with patients that have
               any pre-index ICD.
  * Presence : phecode = 1 if the patient has >= 1 mapped pre-index code, else 0.
  * Filter   : keep phecodes present in >= 1% of the c3 cohort.

Outputs (clustering/data/c3_comorbidities/):
  c3_features_maximal.csv   MRN, index_ahi, max_ahi, <phecode 0/1 columns>
  c3_phecode_labels.csv     phecode, phenotype, category, prevalence
  c3_outlier_report.csv     step-1 summary (binary -> prevalence)
  c3_build_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from qc import zscore_outlier_report          # noqa: E402

ICDS = Path(r"C:\Users\natha\projects\work\phewas\results\archived results"
            r"\icd_analysis_results\phenotype\icds_all.csv")
RES = Path(r"C:\Users\natha\anaconda3\envs\myenv2\Lib\site-packages\pyPheWAS\resources")
DIM = Path(r"C:\Users\natha\projects\work\phewas\results\incwas_results\event_store\phecode_dim.csv")
C1 = Path(__file__).resolve().parents[2] / "data" / "c1_physiologic" / "c1_features_maximal.csv"
OUT = Path(__file__).resolve().parents[2] / "data" / "c3_comorbidities"
PREV_FLOOR = 0.01


def _norm(code: str, icd_type: int) -> str:
    c = str(code).strip().upper()
    return c


def build_code_map() -> pd.DataFrame:
    """Return DataFrame [ICD_TYPE, ICD_CODE, phecode] mapping every distinct data code."""
    m9 = pd.read_csv(RES / "phecode_map_v1_2_icd9.csv", dtype=str, usecols=["ICD9", "PheCode"])
    m10 = pd.read_csv(RES / "phecode_map_v1_2_icd10_beta.csv", dtype=str, usecols=["ICD10", "PheCode"])
    maps = {9: dict(zip(m9["ICD9"].str.strip().str.upper(), m9["PheCode"].str.strip())),
            10: dict(zip(m10["ICD10"].str.strip().str.upper(), m10["PheCode"].str.strip()))}

    con = duckdb.connect()
    codes = con.execute(
        f"SELECT DISTINCT ICD_TYPE, ICD_CODE FROM read_csv_auto('{ICDS.as_posix()}', "
        f"header=true, types={{'ICD_CODE':'VARCHAR'}})"
    ).fetchdf()
    con.close()

    out_rows = []
    for icd_type, d in maps.items():
        sub = codes[codes["ICD_TYPE"] == icd_type]
        for raw in sub["ICD_CODE"].dropna().unique():
            key = str(raw).strip().upper()
            pc = d.get(key)
            if pc is None:                                  # longest-prefix fallback
                k = key
                while pc is None and len(k) > 2:
                    k = k[:-1]
                    if k.endswith("."):
                        k = k[:-1]
                    pc = d.get(k)
            if pc is not None:
                out_rows.append((icd_type, raw, pc))
    return pd.DataFrame(out_rows, columns=["ICD_TYPE", "ICD_CODE", "phecode"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roster = pd.read_csv(C1, usecols=["MRN", "index_ahi", "max_ahi"], dtype={"MRN": str})
    roster["MRN"] = roster["MRN"].str.zfill(12)

    code_map = build_code_map()

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.register("code_map", code_map)
    con.register("roster", roster[["MRN"]].assign(MRN=lambda d: d["MRN"].str.zfill(12)))
    con.execute(f"""
        CREATE TEMP TABLE pairs AS
        SELECT DISTINCT lpad(CAST(i.id AS VARCHAR), 12, '0') AS MRN, m.phecode AS phecode
        FROM read_csv_auto('{ICDS.as_posix()}', header=true, types={{'ICD_CODE':'VARCHAR'}}) i
        JOIN code_map m ON i.ICD_TYPE = m.ICD_TYPE AND i.ICD_CODE = m.ICD_CODE
        WHERE lpad(CAST(i.id AS VARCHAR), 12, '0') IN (SELECT MRN FROM roster)
    """)
    cohort_n = con.execute("SELECT COUNT(DISTINCT MRN) FROM pairs").fetchone()[0]
    prev = con.execute(f"""
        SELECT phecode, COUNT(DISTINCT MRN) AS n
        FROM pairs GROUP BY phecode
        HAVING COUNT(DISTINCT MRN) >= {PREV_FLOOR} * {cohort_n}
        ORDER BY n DESC
    """).fetchdf()
    kept = set(prev["phecode"])
    long = con.execute("SELECT MRN, phecode FROM pairs").fetchdf()
    con.close()

    long = long[long["phecode"].isin(kept)]
    mat = (long.assign(v=np.int8(1))
           .pivot_table(index="MRN", columns="phecode", values="v", fill_value=0)
           .astype(np.int8))
    # every OSA roster patient with >=1 pre-index code is a row; reindex to that cohort
    cohort_mrns = sorted(set(long["MRN"]))
    mat = mat.reindex(cohort_mrns).fillna(0).astype(np.int8)
    mat.index.name = "MRN"
    mat = mat.reset_index()

    out = roster.merge(mat, on="MRN", how="inner")
    phe_cols = [c for c in mat.columns if c != "MRN"]
    out = out[["MRN", "index_ahi", "max_ahi"] + phe_cols]
    out.to_csv(OUT / "c3_features_maximal.csv", index=False)

    dim = pd.read_csv(DIM, dtype={"phecode": str})
    prev["prevalence"] = (prev["n"] / cohort_n).round(4)
    labels = prev.merge(dim, on="phecode", how="left")[
        ["phecode", "phenotype", "category", "n", "prevalence"]]
    labels.to_csv(OUT / "c3_phecode_labels.csv", index=False)
    zscore_outlier_report(out, phe_cols).to_csv(OUT / "c3_outlier_report.csv", index=False)

    report = {
        "icds_source": str(ICDS), "prevalence_floor": PREV_FLOOR,
        "n_cohort": int(cohort_n), "n_phecodes_kept": len(phe_cols),
        "n_rows_out": len(out),
        "note": "binary pre-index phecode presence (>=1 mapped code). NO log1p at reduce time; "
                "z-score standardized. Cohort = OSA roster (max AHI>=5) with >=1 pre-index ICD.",
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "duckdb": duckdb.__version__, "python": sys.version.split()[0]},
    }
    (OUT / "c3_build_report.json").write_text(json.dumps(report, indent=2))
    print(f"c3 built: {len(out)} OSA patients x {len(phe_cols)} phecodes "
          f"(>= {PREV_FLOOR:.0%} prevalence; cohort_n={cohort_n})")


if __name__ == "__main__":
    main()
