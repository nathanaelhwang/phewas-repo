# %% [markdown]
# Step 1 - Extract labs from PopulationCohort251103.csv (~101 GB) in ONE streaming pass.
#
# The source is fully denormalized (~1000x cartesian explosion per MRN), so the
# load-bearing move is `SELECT DISTINCT (id, lab, lab_date, lab_result)` to
# recover true (patient, analyte, date, value) events before any aggregation.
#
# Emits (all parquet):
#   cohort/lab_anchor.parquet                : id, index_date (earliest plausible study)
#   labs/labs_long_numeric.parquet           : id, lab, lab_date, value_num  (DISTINCT, de-exploded)
#   labs/labs_nonnumeric_quarantine.parquet  : id, lab, lab_date, lab_result (DISTINCT, audit)
# DE-IDENTIFIED summary only (counts/rates; never row-level values).

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import labwas_utils as U

# Numeric coercion: drop thousands-commas and a single leading comparator
# (<, >, <=, >=, ~), then require the WHOLE remainder to cast cleanly. This is
# deliberately strict -- ranges ("12-15"), unit-bearing ("5.4 mg/dL") and
# qualitative ("POSITIVE") results fail the cast and land in the quarantine,
# rather than being silently mis-parsed.
NUMEXPR = (
    "try_cast(trim(regexp_replace(replace(trim(lab_result), ',', ''),"
    " '^(<=|>=|<|>|~)\\s*', '')) AS DOUBLE)"
)


def main() -> None:
    U.ensure_dirs()
    anchor_out = (U.COHORT_DIR / "lab_anchor.parquet").as_posix()
    num_out = (U.LABS_DIR / "labs_long_numeric.parquet").as_posix()
    quar_out = (U.LABS_DIR / "labs_nonnumeric_quarantine.parquet").as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{U.duckdb_tmp(__file__)}'")

    # ---- single streaming scan -> slim projection of the 5 needed columns ----
    print("Scanning PopulationCohort251103.csv (~101 GB, one pass)...", flush=True)
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE src AS
        SELECT {U.MRN_SQL} AS id,
               try_strptime(STUDY_DATE, '{U.SAS}')::DATE AS sdate,
               nullif(trim(lab), '')                     AS lab,
               try_strptime(lab_date, '{U.SAS}')::DATE   AS lab_date,
               nullif(trim(lab_result), '')              AS lab_result
        FROM read_csv('{U.POP_COHORT}', header=true, quote='"', escape='"',
                      all_varchar=true, ignore_errors=true)
    """)
    print(f"  src materialized [{time.time() - t0:.0f}s]", flush=True)

    # ---- A. anchor: earliest plausible study per MRN (one row per MRN) -------
    con.execute(f"""
        COPY (
            SELECT id, min(sdate) AS index_date
            FROM src
            WHERE id IS NOT NULL AND sdate IS NOT NULL
              AND year(sdate) >= {U.MIN_STUDY_YEAR} AND sdate <= current_date
            GROUP BY id
        ) TO '{anchor_out}' (FORMAT PARQUET)
    """)
    print(f"  wrote anchor [{time.time() - t0:.0f}s]", flush=True)

    # ---- de-explode: DISTINCT lab events, numeric parsed once ---------------
    con.execute(f"""
        CREATE TEMP TABLE lab_events AS
        SELECT id, lab, lab_date, lab_result, {NUMEXPR} AS value_num
        FROM (
            SELECT DISTINCT id, lab, lab_date, lab_result
            FROM src
            WHERE id IS NOT NULL AND lab IS NOT NULL
              AND lab_result IS NOT NULL AND lab_date IS NOT NULL
        )
    """)
    con.execute("DROP TABLE src")
    print(f"  de-exploded lab_events [{time.time() - t0:.0f}s]", flush=True)

    # ---- B. numeric long  /  C. non-numeric quarantine ----------------------
    con.execute(f"""
        COPY (SELECT id, lab, lab_date, value_num FROM lab_events
              WHERE value_num IS NOT NULL)
        TO '{num_out}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (SELECT id, lab, lab_date, lab_result FROM lab_events
              WHERE value_num IS NULL)
        TO '{quar_out}' (FORMAT PARQUET)
    """)

    # ---- DE-IDENTIFIED summary ---------------------------------------------
    n_anchor = con.sql(f"SELECT count(*) FROM read_parquet('{anchor_out}')").fetchone()[0]
    n_events = con.sql("SELECT count(*) FROM lab_events").fetchone()[0]
    n_num = con.sql("SELECT count(*) FROM lab_events WHERE value_num IS NOT NULL").fetchone()[0]
    n_quar = n_events - n_num
    n_labs = con.sql("SELECT count(DISTINCT lab) FROM lab_events WHERE value_num IS NOT NULL").fetchone()[0]

    print("\n==== Step 1 extract summary ====", flush=True)
    print(f"MRNs with a plausible index study (>= {U.MIN_STUDY_YEAR}): {n_anchor:,}")
    print(f"distinct lab events (de-exploded):  {n_events:,}")
    print(f"  numeric (kept):                   {n_num:,} ({100 * n_num / n_events:.1f}%)")
    print(f"  non-numeric (quarantined):        {n_quar:,} ({100 * n_quar / n_events:.1f}%)")
    print(f"distinct numeric analytes:          {n_labs:,}")
    print(f"total scan time: {time.time() - t0:.0f}s")
    U.log_milestone(
        f"Step 1 extract: {n_anchor:,} anchored MRNs; {n_events:,} distinct lab events "
        f"({n_num:,} numeric / {n_quar:,} quarantined); {n_labs:,} numeric analytes."
    )


if __name__ == "__main__":
    main()
