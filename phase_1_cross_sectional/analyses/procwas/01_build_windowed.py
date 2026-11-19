# %% [markdown]
# ProcWAS Step 01 - Windowed procedure store (the ONLY pass over proc.csv, 11.7 GB).
#
# Collapses the diagnosis-line grain to distinct procedure ORDERS (COUNT is meaningful only at
# order grain, not dx-line grain), for v2-cohort patients whose CONTACT_DATE falls in
# [index - 5yr, index) -- lower inclusive, upper exclusive -- with an in_1yr flag. Each order is
# classified via the crosswalk (PROC_CODE == 'CPT primary') into Clear descriptor / Procedure type /
# Body system. Procedure key = ORDER_PROC_ID (fallback id|date|code when that id is missing).
#
# -> results/procwas_results/windowed/procedures_5yr.parquet   (one row per distinct order)
# DE-IDENTIFIED summary to stdout only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import procwas_utils as U


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=48, threads=8)
    con.execute(U.STRIP_B)
    con.register("xw", U.crosswalk())

    # v2 cohort spine: id (12-char MRN), index date
    con.execute(f"""
        CREATE TEMP TABLE c AS
        SELECT id, CAST(STUDY_DATE AS DATE) AS idx FROM read_parquet('{U.COHORT}')
    """)
    n_cohort = con.sql("SELECT count(*) FROM c").fetchone()[0]

    # ---- single streaming pass: strip -> COMPLETED -> distinct orders -> classify -> window ----
    con.execute(f"""
        COPY (
          WITH raw AS (
            SELECT {U.PROC_ID_SQL} AS id,
                   TRY_CAST(TRY_CAST(ORDER_PROC_ID AS DOUBLE) AS BIGINT) AS opid,
                   TRY_CAST(CONTACT_DATE AS DATE) AS contact_date,
                   strip_b(PROC_CODE) AS proc_code
            FROM read_csv('{U.PROC}', {U.CSV_READ}, header=true)
            WHERE strip_b(ORDER_STATUS) = 'COMPLETED'
          ),
          keyed AS (
            SELECT id,
                   COALESCE(CAST(opid AS VARCHAR),
                            id||'|'||CAST(contact_date AS VARCHAR)||'|'||COALESCE(proc_code,'?')) AS proc_key,
                   (opid IS NULL) AS synth_key,
                   contact_date, proc_code
            FROM raw
          ),
          orders AS (
            SELECT id, proc_key, bool_or(synth_key) AS synth_key,
                   any_value(contact_date) AS contact_date, any_value(proc_code) AS proc_code
            FROM keyed GROUP BY id, proc_key
          )
          SELECT o.id, o.proc_key, o.synth_key, o.contact_date, o.proc_code,
                 x.clear_descriptor, x.procedure_type, x.body_system,
                 (o.contact_date >= (c.idx - to_years(1))::DATE) AS in_1yr
          FROM orders o
          JOIN c ON o.id = c.id
          LEFT JOIN xw x ON upper(o.proc_code) = x.key_u
          WHERE o.contact_date >= (c.idx - to_years(5))::DATE
            AND o.contact_date <  c.idx
        ) TO '{U.STORE_5YR}' (FORMAT PARQUET)
    """)

    # ---- de-identified QC off the (small) store ----
    n_rows, n_proc, n_pat, dmin, dmax = con.sql(f"""
        SELECT count(*), count(DISTINCT (id||proc_key)), count(DISTINCT id), min(contact_date), max(contact_date)
        FROM read_parquet('{U.STORE_5YR}')""").fetchone()
    n_rows1, n_pat1 = con.sql(f"""SELECT count(*), count(DISTINCT id)
        FROM read_parquet('{U.STORE_5YR}') WHERE in_1yr""").fetchone()
    n_synth = con.sql(f"SELECT count(*) FROM read_parquet('{U.STORE_5YR}') WHERE synth_key").fetchone()[0]
    n_xw = con.sql(f"SELECT count(*) FROM read_parquet('{U.STORE_5YR}') WHERE clear_descriptor IS NOT NULL").fetchone()[0]

    print(f"cohort patients (v2 spine): {n_cohort:,}")
    print(f"5yr store : orders={n_rows:,}  patients={n_pat:,}  contact_date {dmin}..{dmax}")
    print(f"1yr subset: orders={n_rows1:,}  patients={n_pat1:,}")
    print(f"cohort patients with >=1 windowed procedure: {n_pat:,}/{n_cohort:,} ({100*n_pat/n_cohort:.1f}%)")
    print(f"orders keyed by ORDER_PROC_ID: {n_rows-n_synth:,}; synthetic-key fallback: {n_synth:,} "
          f"({100*n_synth/max(n_rows,1):.2f}%)")
    print(f"orders with a crosswalk match: {n_xw:,} ({100*n_xw/max(n_rows,1):.1f}%)")

    U.log_milestone(f"Step 01 windowed procedure store: {n_rows:,} distinct orders for "
                    f"{n_pat:,}/{n_cohort:,} cohort patients in 5yr pre-index (1yr {n_rows1:,}).")
    con.close()


if __name__ == "__main__":
    main()
