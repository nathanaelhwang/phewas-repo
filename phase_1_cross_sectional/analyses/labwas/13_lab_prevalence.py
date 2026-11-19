# %% [markdown]
# Step 13 - Per-lab prevalence / missingness table (data completeness).
# For each mapped LOINC analyte, over the ICD v2 cohort (denominator = 222,109):
#   prevalence = fraction of patients with >=1 measurement in the window (tested),
#   missingness = 1 - prevalence,
# reported for the 1yr and 5yr windows, plus prevalence by OSA severity in the 5yr
# window (to surface DIFFERENTIAL / informative missingness across the exposure).
# in_analysis flags whether the lab cleared the >=500pt & >=1000obs analytic floor.
# Output: analysis/lab_prevalence_missingness.csv.  DE-IDENTIFIED aggregates only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import labwas_utils as U

V2_COHORT = (U.PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()
ANALYSIS = U.RESULTS / "analysis"
WINDOWS = [("1yr", 1), ("5yr", 5)]
SEV = ["None", "Mild", "Moderate", "Severe"]


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{U.duckdb_tmp(__file__)}'")
    num = (U.LABS_DIR / "labs_long_numeric.parquet").as_posix()
    cw = (U.CROSSWALK_DIR / "loinc_crosswalk.csv").as_posix()

    denom = con.sql(f"SELECT count(*) FROM read_parquet('{V2_COHORT}')").fetchone()[0]
    dsev = dict(con.sql(f"SELECT severity, count(*) FROM read_parquet('{V2_COHORT}') GROUP BY 1").df().values)
    print(f"v2 cohort denominator: {denom:,}  ({dsev})")

    dd = pd.read_csv(U.RESULTS / "column_data_dictionary.csv")          # the 204 mapped analytes
    loincs = "','".join(dd["loinc_code"].astype(str))

    out = dd[["loinc_code", "loinc_long_common_name"]].copy()
    for label, years in WINDOWS:
        # distinct tested patients per (loinc, severity) within the window
        pc = con.sql(f"""
            WITH x AS (SELECT lab, loinc_code FROM read_csv('{cw}', header=true, all_varchar=true)
                       WHERE loinc_code IN ('{loincs}')),
                 c AS (SELECT id, severity, CAST(STUDY_DATE AS DATE) idx FROM read_parquet('{V2_COHORT}')),
                 n AS (SELECT id, lab, lab_date FROM read_parquet('{num}'))
            SELECT x.loinc_code, c.severity, count(DISTINCT c.id) AS n_pt
            FROM n JOIN x USING (lab) JOIN c USING (id)
            WHERE n.lab_date >= (c.idx - to_years({years}))::DATE AND n.lab_date < c.idx
            GROUP BY x.loinc_code, c.severity
        """).df()
        tot = pc.groupby("loinc_code")["n_pt"].sum().rename(f"n_{label}")
        out = out.merge(tot, on="loinc_code", how="left")
        out[f"n_{label}"] = out[f"n_{label}"].fillna(0).astype(int)
        out[f"prevalence_{label}"] = (out[f"n_{label}"] / denom).round(4)
        out[f"missingness_{label}"] = (1 - out[f"prevalence_{label}"]).round(4)
        if label == "5yr":  # differential missingness across the exposure
            wide = pc.pivot_table(index="loinc_code", columns="severity", values="n_pt", fill_value=0)
            for s in SEV:
                col = wide.get(s, pd.Series(0, index=wide.index))
                out[f"prev_{s}_5yr"] = out["loinc_code"].map((col / dsev[s]).round(4)).fillna(0)
        # in-analysis flag
        kept = set(pd.read_parquet(ANALYSIS / f"targets_{label}.parquet", columns=["loinc_code"])["loinc_code"])
        out[f"in_analysis_{label}"] = out["loinc_code"].isin(kept)

    cat = pd.read_csv(U.RESULTS / "loinc_category.csv")[["loinc_code", "category"]]
    out = out.merge(cat, on="loinc_code", how="left")
    cols = (["loinc_code", "loinc_long_common_name", "category",
             "n_5yr", "prevalence_5yr", "missingness_5yr", "in_analysis_5yr",
             "n_1yr", "prevalence_1yr", "missingness_1yr", "in_analysis_1yr"]
            + [f"prev_{s}_5yr" for s in SEV])
    out = out[cols].sort_values("prevalence_5yr", ascending=False)
    dst = U.RESULTS / "lab_prevalence_missingness.csv"
    out.to_csv(dst, index=False)

    print(f"\nWrote {dst}  ({len(out)} analytes)")
    print(f"analytes >=90% prevalence (5yr): {(out['prevalence_5yr']>=0.90).sum()}")
    print(f"analytes <10% prevalence (5yr):  {(out['prevalence_5yr']<0.10).sum()}")
    print("\n=== most complete (top 8) ===")
    print(out.head(8)[["loinc_code", "loinc_long_common_name", "prevalence_5yr", "missingness_5yr"]].to_string(index=False))
    print("\n=== most missing (bottom 8) ===")
    print(out.tail(8)[["loinc_code", "loinc_long_common_name", "prevalence_5yr", "missingness_5yr"]].to_string(index=False))
    U.log_milestone("Step 13: per-lab prevalence/missingness table written (lab_prevalence_missingness.csv).")


if __name__ == "__main__":
    main()
