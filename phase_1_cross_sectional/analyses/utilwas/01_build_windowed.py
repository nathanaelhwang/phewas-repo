# %% [markdown]
# Step 01 - Build the windowed utilization store (the ONLY pass over the 104 GB CSV).
#
# Keeps every ORIGINAL utilization.csv column for cohort patients whose encounter date
# (ADATE) falls in [index - 5yr, index) -- lower inclusive, upper exclusive -- plus derived
# columns: id (normalized MRN), enc_date (parsed ADATE), los_days (numeric LOS), severity
# (from the v2 ICD cohort), and an in_1yr flag. Diagnosis-line grain is preserved (needed for
# the CODE_SUBTYPE physician/allied logic and the overlap export); encounters are counted
# downstream as COUNT(DISTINCT UTIL_ID).
#
# Emits:
#   results/utilwas_results/windowed/utilization_5yr.parquet   (the analytic store)
#   D:/phewas_data/utilization_5yr.csv, utilization_1yr.csv    (requested literal files)
# DE-IDENTIFIED summary to stdout only; patient-level files land on local disk.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import utilwas_utils as U


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    # v2 cohort spine: id, index date, severity
    con.execute(f"""
        CREATE TEMP TABLE c AS
        SELECT id, CAST(STUDY_DATE AS DATE) AS idx, severity
        FROM read_parquet('{U.COHORT}')
    """)
    n_cohort = con.sql("SELECT count(*) FROM c").fetchone()[0]
    print(f"cohort patients (v2 spine): {n_cohort:,}")

    # ---- the single streaming pass over utilization.csv --------------------
    con.execute(f"""
        COPY (
          WITH u AS (
            SELECT *,
                   {U.MRN_SQL}                       AS id,
                   try_strptime(ADATE, '{U.SAS}')::DATE AS enc_date,
                   TRY_CAST(LOS AS DOUBLE)           AS los_days
            FROM read_csv_auto('{U.UTIL_CSV}', {U.CSV_READ})
          )
          SELECT u.*, c.severity,
                 (u.enc_date >= (c.idx - to_years(1))::DATE) AS in_1yr
          FROM u JOIN c ON u.id = c.id
          WHERE u.enc_date >= (c.idx - to_years(5))::DATE
            AND u.enc_date <  c.idx
        ) TO '{U.STORE_5YR}' (FORMAT PARQUET)
    """)

    # ---- QC off the (small) store ------------------------------------------
    n_rows, n_util, n_pat, dmin, dmax = con.sql(f"""
        SELECT count(*), count(DISTINCT UTIL_ID), count(DISTINCT id),
               min(enc_date), max(enc_date)
        FROM read_parquet('{U.STORE_5YR}')
    """).fetchone()
    n_rows1, n_util1, n_pat1 = con.sql(f"""
        SELECT count(*), count(DISTINCT UTIL_ID), count(DISTINCT id)
        FROM read_parquet('{U.STORE_5YR}') WHERE in_1yr
    """).fetchone()
    print(f"5yr store : rows={n_rows:,}  encounters={n_util:,}  patients={n_pat:,}  enc_date {dmin}..{dmax}")
    print(f"1yr subset: rows={n_rows1:,}  encounters={n_util1:,}  patients={n_pat1:,}")
    print(f"cohort patients with >=1 windowed encounter: {n_pat:,}/{n_cohort:,} ({100*n_pat/n_cohort:.1f}%)")

    # ---- requested literal windowed files (all original columns) -----------
    con.execute(f"COPY (SELECT * FROM read_parquet('{U.STORE_5YR}')) "
                f"TO '{U.UTIL_5YR_CSV}' (FORMAT CSV, HEADER)")
    con.execute(f"COPY (SELECT * FROM read_parquet('{U.STORE_5YR}') WHERE in_1yr) "
                f"TO '{U.UTIL_1YR_CSV}' (FORMAT CSV, HEADER)")
    print(f"wrote {U.UTIL_5YR_CSV}")
    print(f"wrote {U.UTIL_1YR_CSV}")

    U.log_milestone(
        f"Step 01 windowed store: {n_rows:,} diagnosis-line rows / {n_util:,} encounters for "
        f"{n_pat:,}/{n_cohort:,} cohort patients within 5yr pre-index (1yr subset {n_util1:,} encounters).")


if __name__ == "__main__":
    main()
