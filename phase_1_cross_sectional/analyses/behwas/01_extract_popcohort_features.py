# %% [markdown]
# Step 01 - Extract the PopulationCohort-embedded behavioral features in ONE pass over
# the ~101 GB cartesian-exploded PopulationCohort251103.csv (the expensive step).
#
# PopulationCohort carries three of the five BehWAS features as columns embedded in the
# giant exploded file. Two are LIFETIME per-person aggregates (constant within MRN, no usable
# date -> cannot be windowed), one is a dated immunization event (windowable):
#   NO-SHOW  : no_show, total_noshow, no_show_pct, total_noshow_pct, office_visit  (per-person)
#   OFF-WORK : off_work (1/2), off_work_days                                        (per-person)
#   FLU SHOT : IMMUNE_DATE (influenza only; distinct (id, date) events)             (dated)
#
# We restrict to the v2 cohort during the scan, then collapse: max() over each per-person
# column (constant within MRN) and DISTINCT (id, immune_date) for the flu events.
#
# Emits:
#   features/popcohort_person.parquet : id + no-show/off-work per-person aggregates
#   features/flu_events.parquet       : id, immune_date (one row per distinct flu shot)
# DE-IDENTIFIED summary to stdout only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import behwas_utils as U


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=150, threads=16)

    con.execute(f"""
        CREATE TEMP TABLE c AS
        SELECT id, CAST(STUDY_DATE AS DATE) AS idx FROM read_parquet('{U.COHORT}')
    """)
    n_cohort = con.sql("SELECT count(*) FROM c").fetchone()[0]
    print(f"cohort patients (v2 spine): {n_cohort:,}")

    # ---- the single streaming pass over PopulationCohort (cohort rows only) ----
    print("Scanning PopulationCohort251103.csv (~101 GB, one pass)...", flush=True)
    con.execute(f"""
        CREATE TEMP TABLE pc AS
        SELECT id, no_show, total_noshow, no_show_pct, total_noshow_pct, office_visit,
               off_work, off_work_days, immune_date
        FROM (
          SELECT {U.MRN_SQL}                              AS id,
                 TRY_CAST(no_show AS DOUBLE)              AS no_show,
                 TRY_CAST(total_noshow AS DOUBLE)         AS total_noshow,
                 TRY_CAST(no_show_pct AS DOUBLE)          AS no_show_pct,
                 TRY_CAST(total_noshow_pct AS DOUBLE)     AS total_noshow_pct,
                 TRY_CAST(office_visit AS DOUBLE)         AS office_visit,
                 TRY_CAST(off_work AS INTEGER)            AS off_work,
                 TRY_CAST(off_work_days AS DOUBLE)        AS off_work_days,
                 COALESCE(try_strptime(IMMUNE_DATE, '{U.SAS}'),
                          try_strptime(IMMUNE_DATE, '{U.SAS_DATE}'))::DATE AS immune_date
          FROM read_csv('{U.POPCOHORT_CSV}', {U.CSV_READ})
        )
        WHERE id IN (SELECT id FROM c)
    """)
    n_pc = con.sql("SELECT count(*) FROM pc").fetchone()[0]
    print(f"  cohort rows materialized from PopulationCohort: {n_pc:,}", flush=True)

    # ---- per-person aggregates (columns constant within MRN -> max()) ----------
    person_out = (U.FEATURES_DIR / "popcohort_person.parquet").as_posix()
    con.execute(f"""
        COPY (
          SELECT id,
                 max(no_show)          AS no_show,
                 max(total_noshow)     AS total_noshow,
                 max(no_show_pct)      AS no_show_pct,
                 max(total_noshow_pct) AS total_noshow_pct,
                 max(office_visit)     AS office_visit,
                 max(off_work)         AS off_work,
                 max(off_work_days)    AS off_work_days
          FROM pc GROUP BY id
        ) TO '{person_out}' (FORMAT PARQUET)
    """)

    # ---- flu events (dated, windowable) ----------------------------------------
    flu_out = (U.FEATURES_DIR / "flu_events.parquet").as_posix()
    con.execute(f"""
        COPY (SELECT DISTINCT id, immune_date FROM pc WHERE immune_date IS NOT NULL)
        TO '{flu_out}' (FORMAT PARQUET)
    """)

    # ---- DE-IDENTIFIED coverage summary ----------------------------------------
    n_person = con.sql(f"SELECT count(*) FROM read_parquet('{person_out}')").fetchone()[0]
    ns = con.sql(f"""SELECT count(*) FILTER (WHERE COALESCE(total_noshow, no_show) > 0),
                            count(*) FILTER (WHERE off_work IN (1, 2))
                     FROM read_parquet('{person_out}')""").fetchone()
    n_flu_ev, n_flu_pat, dmin, dmax = con.sql(f"""
        SELECT count(*), count(DISTINCT id), min(immune_date), max(immune_date)
        FROM read_parquet('{flu_out}')""").fetchone()
    print("\n==== Step 01 extract summary ====")
    print(f"per-person rows (unique cohort ids seen): {n_person:,}/{n_cohort:,} "
          f"({100*n_person/n_cohort:.1f}%)")
    print(f"  patients with >=1 no-show:   {ns[0]:,}")
    print(f"  patients with an off_work code (1/2): {ns[1]:,}")
    print(f"flu events (distinct id,date): {n_flu_ev:,} for {n_flu_pat:,} patients; dates {dmin}..{dmax}")

    U.log_milestone(
        f"Step 01 PopulationCohort features: {n_person:,}/{n_cohort:,} cohort ids collapsed "
        f"(no-show + off-work per-person); {n_flu_ev:,} distinct flu events for {n_flu_pat:,} patients.")


if __name__ == "__main__":
    main()
