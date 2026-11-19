# %% [markdown]
# Step 01 - The dated phecode event store (the ONLY pass over the 104 GB utilization.csv).
#
# Maps every ICD9/ICD10 diagnosis (ADATE <= 2023-05-31) for cohort patients to a phecode
# (pyPheWAS v1.2), then collapses to ONE row per (id, phecode) carrying the dates needed for a
# time-to-event incidence analysis with a rule-of-1 washout + rule-of-2 incident rule:
#
#   has_pre          any occurrence on/before index (STUDY_DATE)  -> EXCLUDED from at-risk (washout)
#   min_pre_date     earliest baseline occurrence
#   first_post_date  1st distinct post-index date                 (rule-of-1 sensitivity)
#   second_post_date 2nd distinct post-index date  = INCIDENT EVENT DATE (rule-of-2, primary)
#   n_post_dates     # distinct post-index dates
#
# Index-day codes (adate == idx) count as BASELINE (never incident) -> no immortal-time leak.
# The output is sparse (a row only where a patient ever had that phecode), a few hundred MB.
# Scale levers: 4-column projection pushdown, cohort semijoin applied early, single COPY.
#
# Usage:
#   python 01_build_event_store.py --limit 5000000   # fast sanity slice
#   python 01_build_event_store.py                   # full pass (~1-2.5h)
# Emits event_store/phecode_events.parquet + phecode_dim.csv. DE-IDENTIFIED summary to stdout.

# %%
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import incwas_utils as U


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="row cap on the CSV scan (0 = full)")
    ap.add_argument("--out", type=str, default=U.EVENTS, help="output parquet path")
    a = ap.parse_args()

    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    # v2 cohort spine: id + index date (all 222,109; step 03 restricts to the follow-up spine).
    con.execute(f"""
        CREATE TEMP TABLE c AS
        SELECT id, CAST(STUDY_DATE AS DATE) AS idx FROM read_parquet('{U.COHORT}')
    """)
    n_cohort = con.sql("SELECT count(*) FROM c").fetchone()[0]
    print(f"cohort patients (v2 spine): {n_cohort:,}", flush=True)

    # ICD -> phecode v1.2 map (icd9 + icd10_beta), deduped on (code-system, dotted code)
    con.execute(U.phecode_map_sql())

    limit_sql = f"LIMIT {a.limit}" if a.limit else ""

    # ---- resolve DISTINCT diagnosis codes -> phecode via exact-then-longest-prefix match -------
    # The v1.2 icd10-beta map often carries only 3-char stems (e.g. I48) while billing uses I48.91;
    # exact matching drops those. We resolve on the small set of distinct codes (a light 2-col scan),
    # then equijoin back to the full row set. Longest matching prefix wins; deliberate NULL-phecode
    # map entries make a code unmapped (we do not fall back past them).
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE codes AS
        SELECT DISTINCT cst, code FROM (
          SELECT upper(trim(CODE_SUBTYPE)) AS cst, upper(trim(CODE)) AS code
          FROM read_csv_auto('{U.UTIL_CSV}', {U.CSV_READ}) {limit_sql}
        ) WHERE cst IN ('ICD9','ICD10') AND code IS NOT NULL
    """)
    con.execute("""
        CREATE TEMP TABLE code2phe AS
        WITH exact AS (
          SELECT c.cst, c.code, m.phecode
          FROM codes c JOIN map m ON c.cst = m.cst AND c.code = m.code
        ),
        need AS (
          SELECT c.cst, c.code FROM codes c
          LEFT JOIN exact e ON c.cst = e.cst AND c.code = e.code WHERE e.code IS NULL
        ),
        prefix AS (
          SELECT cst, code, phecode FROM (
            SELECT n.cst, n.code, m.phecode,
                   row_number() OVER (PARTITION BY n.cst, n.code ORDER BY length(m.code) DESC) rn
            FROM need n JOIN map m ON n.cst = m.cst AND n.code LIKE m.code || '%'
          ) WHERE rn = 1
        )
        SELECT cst, code, phecode FROM exact
        UNION ALL SELECT cst, code, phecode FROM prefix
    """)
    n_codes, n_mapped, n_prefix = con.sql("""
        SELECT (SELECT count(*) FROM codes),
               (SELECT count(*) FROM code2phe WHERE phecode IS NOT NULL),
               (SELECT count(*) FROM code2phe c WHERE phecode IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM map m WHERE m.cst=c.cst AND m.code=c.code))
    """).fetchone()
    print(f"distinct codes: {n_codes:,}; mapped: {n_mapped:,}; recovered via prefix: {n_prefix:,}", flush=True)

    con.execute(f"""
        COPY (
          WITH u AS (
            SELECT {U.MRN_SQL}                          AS id,
                   upper(trim(CODE))                    AS code,
                   upper(trim(CODE_SUBTYPE))            AS cst,
                   try_strptime(ADATE, '{U.SAS}')::DATE AS adate
            FROM read_csv_auto('{U.UTIL_CSV}', {U.CSV_READ}) {limit_sql}
          ),
          j AS (
            SELECT u.id, u.adate, c.idx, r.phecode
            FROM u
            JOIN c   ON u.id = c.id
            JOIN code2phe r ON u.cst = r.cst AND u.code = r.code
            WHERE u.cst IN ('ICD9','ICD10')
              AND r.phecode IS NOT NULL AND trim(r.phecode) <> ''
              AND u.adate IS NOT NULL
              AND u.adate <= DATE '{U.ADATE_CUTOFF}'
          ),
          pre_agg AS (
            SELECT id, phecode, TRUE AS has_pre, min(adate) AS min_pre_date
            FROM j WHERE adate <= idx GROUP BY id, phecode
          ),
          post_dates AS (
            SELECT DISTINCT id, phecode, adate FROM j WHERE adate > idx
          ),
          post_ranked AS (
            SELECT id, phecode, adate,
                   row_number() OVER (PARTITION BY id, phecode ORDER BY adate) AS rn
            FROM post_dates
          ),
          post_agg AS (
            SELECT id, phecode,
                   min(adate)                              AS first_post_date,
                   max(CASE WHEN rn = 2 THEN adate END)    AS second_post_date,
                   count(*)                                AS n_post_dates
            FROM post_ranked GROUP BY id, phecode
          )
          SELECT id, phecode,
                 COALESCE(pre.has_pre, FALSE)   AS has_pre,
                 pre.min_pre_date,
                 po.first_post_date, po.second_post_date,
                 COALESCE(po.n_post_dates, 0)   AS n_post_dates
          FROM pre_agg pre FULL OUTER JOIN post_agg po USING (id, phecode)
        ) TO '{a.out}' (FORMAT PARQUET)
    """)
    dt = time.time() - t0
    print(f"scan+aggregate done in {dt/60:.1f} min", flush=True)

    # phecode dimension (labels) for downstream joins
    con.execute(f"""COPY (SELECT phecode, any_value(phenotype) phenotype, any_value(category) category
                          FROM map WHERE phecode IS NOT NULL AND trim(phecode) <> ''
                          GROUP BY phecode ORDER BY phecode)
                    TO '{U.PHECODE_DIM}' (HEADER, DELIMITER ',')""")

    # ---- QC off the (small) event store ------------------------------------
    n_rows, n_pairs, n_phe, n_pat = con.sql(f"""
        SELECT count(*), count(DISTINCT (id||'|'||phecode)), count(DISTINCT phecode), count(DISTINCT id)
        FROM read_parquet('{a.out}')
    """).fetchone()
    n_prev, n_incid = con.sql(f"""
        SELECT sum(CASE WHEN has_pre THEN 1 ELSE 0 END),
               sum(CASE WHEN NOT has_pre AND second_post_date IS NOT NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{a.out}')
    """).fetchone()
    print(f"event store: rows={n_rows:,}  (id,phecode) pairs={n_pairs:,}  phecodes={n_phe:,}  patients={n_pat:,}")
    print(f"  baseline-prevalent (has_pre) rows: {n_prev:,}")
    print(f"  incident-eligible (no pre, >=2 post dates) rows: {n_incid:,}")

    # sanity: hypertension 401.1 -- baseline prevalence vs incident-eligible among at-risk
    for phe in ("401.1", "250.2"):
        r = con.sql(f"""
            SELECT count(*) FILTER (WHERE has_pre)                                        AS prevalent,
                   count(*) FILTER (WHERE NOT has_pre AND second_post_date IS NOT NULL)   AS incident2,
                   count(*) FILTER (WHERE NOT has_pre AND first_post_date IS NOT NULL)    AS incident1
            FROM read_parquet('{a.out}') WHERE phecode = '{phe}'
        """).fetchone()
        print(f"  phecode {phe}: prevalent={r[0]:,}  incident(>=2 codes)={r[1]:,}  incident(>=1 code)={r[2]:,}")

    U.log_milestone(
        f"Step 01 event store{' [LIMIT '+str(a.limit)+']' if a.limit else ''}: "
        f"{n_rows:,} (id,phecode) rows over {n_phe:,} phecodes / {n_pat:,} patients in {dt/60:.1f} min; "
        f"{n_prev:,} baseline-prevalent, {n_incid:,} incident-eligible (rule-of-2). -> phecode_events.parquet")


if __name__ == "__main__":
    main()
