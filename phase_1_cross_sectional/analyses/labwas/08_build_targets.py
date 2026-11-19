# %% [markdown]
# Step 08 - Build the LabWAS analytic target tables (one per window).
#
# For each (patient, LOINC analyte, window) compute the three targets, restricted
# to the ICD v2 OSA cohort and windowed to that cohort's index STUDY_DATE:
#   - mean, median  (value; unit_rules harmonization applied)
#   - count         (n observations in window)
#   - order_rate    = count / obs_years,  obs_years = LEAST(cohort.obs_window, N)
#     (annualized ordering frequency; denominator = ICD observation years capped at the window)
# Keep analytes meeting the dual floor on the v2 analytic sample (>=500 patients,
# >=1000 obs, per window). Output: analysis/targets_{1yr,5yr}.parquet (long).
# DE-IDENTIFIED summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import labwas_utils as U

V2_COHORT = (U.PROJECT / "results" / "icd_analysis_v2_results" / "cohort" / "cohort.parquet").as_posix()
ANALYSIS = U.RESULTS / "analysis"
WINDOWS = [("1yr", 1), ("5yr", 5)]

# unit-conversion rules (same value expression as 05_build_matrices.py)
VAL = ("CASE WHEN s.factor IS NOT NULL AND n.value_num >= s.threshold "
       "THEN n.value_num * s.factor ELSE n.value_num END")


def build(con: duckdb.DuckDBPyConnection, label: str, years: int) -> None:
    num = (U.LABS_DIR / "labs_long_numeric.parquet").as_posix()
    cw = (U.CROSSWALK_DIR / "loinc_crosswalk.csv").as_posix()
    rules = (U.CROSSWALK_DIR / "unit_rules.csv").as_posix()

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pv AS
        WITH s AS (SELECT descriptor, try_cast(threshold AS DOUBLE) threshold, try_cast(factor AS DOUBLE) factor
                   FROM read_csv('{rules}', header=true, all_varchar=true)),
             x AS (SELECT lab, loinc_code FROM read_csv('{cw}', header=true, all_varchar=true)
                   WHERE loinc_code IS NOT NULL AND loinc_code <> ''),
             c AS (SELECT id, CAST(STUDY_DATE AS DATE) AS idx, obs_window FROM read_parquet('{V2_COHORT}')),
             n AS (SELECT id, lab, lab_date, value_num FROM read_parquet('{num}'))
        SELECT c.id, x.loinc_code,
               avg({VAL})    AS mean,
               median({VAL}) AS median,
               count(*)      AS count,
               count(*) / greatest(least(c.obs_window, {years}), 1.0/365) AS order_rate
        FROM n JOIN x USING (lab) JOIN c USING (id)
               LEFT JOIN s ON s.descriptor = n.lab
        WHERE n.lab_date >= (c.idx - to_years({years}))::DATE AND n.lab_date < c.idx
        GROUP BY c.id, x.loinc_code, c.obs_window
    """)

    # dual floor on the v2 analytic sample
    keep = con.sql(f"""
        SELECT loinc_code FROM pv GROUP BY loinc_code
        HAVING count(DISTINCT id) >= {U.MIN_PATIENTS} AND sum(count) >= {U.MIN_OBS}
    """).df()["loinc_code"].tolist()
    keset = "','".join(keep)

    out = ANALYSIS / f"targets_{label}.parquet"
    con.execute(f"""
        COPY (SELECT id, loinc_code, mean, median, count, order_rate
              FROM pv WHERE loinc_code IN ('{keset}'))
        TO '{out.as_posix()}' (FORMAT PARQUET)
    """)
    n_rows = con.sql(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    n_pt = con.sql(f"SELECT count(DISTINCT id) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"[{label}] {len(keep):,} analytes kept | {n_rows:,} (patient,lab) rows | {n_pt:,} patients")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{U.duckdb_tmp(__file__)}'")
    ncoh = con.sql(f"SELECT count(*) FROM read_parquet('{V2_COHORT}')").fetchone()[0]
    print(f"v2 cohort: {ncoh:,} patients")
    for label, years in WINDOWS:
        build(con, label, years)
    U.log_milestone("Step 08 LabWAS targets: analysis/targets_{1yr,5yr}.parquet built (mean/median/count/order_rate).")


if __name__ == "__main__":
    main()
