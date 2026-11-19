# %% [markdown]
# Step 3 - Extract the ICD phenome + utilization burden from utilization.csv (104 GB).
# ONE streaming DuckDB pass, filtered to: cohort MRNs, CODE_SUBTYPE in {ICD9,ICD10},
# valid CODE, and ADATE <= STUDY_DATE (pre-study -> comorbidity precedes exposure).
# Emits:
#   phenotype/icds_all.csv      : id, ICD_CODE (dotted), ICD_TYPE (9/10), AgeAtICD  (1 row per id x code)
#   cohort/encounters.parquet   : id, n_encounters (distinct diagnostic encounters, pre-study)
# DE-IDENTIFIED summary only.

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import utils as U

UTIL = "D:/phewas_data/utilization.csv"
MEMBER = "D:/phewas_data/member.csv"
COHORT = (U.COHORT_DIR / "cohort_base.parquet").as_posix()
ICDS_OUT = (U.PHENO_DIR / "icds_all.csv").as_posix()
ENC_OUT = U.COHORT_DIR / "encounters.parquet"

SAS = "%d%b%Y:%H:%M:%S"


def main() -> None:
    U.PHENO_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{(Path(__file__).parent / '_duckdb_tmp').as_posix()}'")

    # cohort keys (id, study date, DOB) -- DOB from member.csv (non-temporal)
    con.execute(f"""
        CREATE TEMP TABLE keys AS
        SELECT c.id,
               CAST(c.STUDY_DATE AS DATE) AS sdate,
               COALESCE(try_strptime(m.DOB, '{SAS}'),
                        try_strptime(m.DOB, '%m/%d/%Y'))::DATE AS dob
        FROM read_parquet('{COHORT}') c
        JOIN (SELECT lpad(regexp_replace(CAST(MRN AS VARCHAR), '\\.0$', ''), 12, '0') AS id,
                     DOB
              FROM read_csv('{MEMBER}', header=true, quote='"', escape='"', all_varchar=true, ignore_errors=true)) m
          USING (id)
    """)
    nkeys = con.sql("SELECT count(*) FROM keys").fetchone()[0]
    print(f"cohort keys: {nkeys:,}")

    # ---- single filtered scan of utilization.csv -> temp table of events ----
    print("Scanning utilization.csv (~104 GB, one pass)...")
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE ev AS
        SELECT u.id,
               u.CODE AS ICD_CODE,
               CASE u.cst WHEN 'ICD9' THEN 9 WHEN 'ICD10' THEN 10 END AS ICD_TYPE,
               date_diff('day', k.dob, u.adate) / 365.2425 AS AgeAtICD,
               u.ENC_SID
        FROM (
            SELECT lpad(regexp_replace(CAST(MRN AS VARCHAR), '\\.0$', ''), 12, '0') AS id,
                   trim(CODE) AS CODE,
                   upper(trim(CODE_SUBTYPE)) AS cst,
                   COALESCE(try_strptime(ADATE, '{SAS}'),
                            try_strptime(ADATE, '%m/%d/%Y %H:%M:%S'),
                            try_strptime(ADATE, '%m/%d/%Y'))::DATE AS adate,
                   ENC_SID
            FROM read_csv('{UTIL}', header=true, quote='"', escape='"',
                          all_varchar=true, ignore_errors=true)
            WHERE CODE IS NOT NULL AND trim(CODE) <> ''
              AND upper(trim(CODE_SUBTYPE)) IN ('ICD9','ICD10')
        ) u
        JOIN keys k USING (id)
        WHERE u.adate IS NOT NULL AND u.adate <= k.sdate
    """)
    n_ev = con.sql("SELECT count(*) FROM ev").fetchone()[0]
    print(f"filtered pre-study ICD9/10 events: {n_ev:,}  [scan {time.time()-t0:.0f}s]")

    # ---- phenotype file: one row per (id, ICD_CODE, ICD_TYPE) ----
    con.execute(f"""
        COPY (
            SELECT id, ICD_CODE, ICD_TYPE, max(AgeAtICD) AS AgeAtICD
            FROM ev
            WHERE ICD_TYPE IS NOT NULL
            GROUP BY id, ICD_CODE, ICD_TYPE
        ) TO '{ICDS_OUT}' (HEADER, DELIMITER ',')
    """)

    # ---- n_encounters per patient (distinct diagnostic encounters, pre-study) ----
    enc = con.sql("""
        SELECT id, count(DISTINCT ENC_SID) AS n_encounters
        FROM ev WHERE ENC_SID IS NOT NULL
        GROUP BY id
    """).df()
    enc.to_parquet(ENC_OUT, index=False)

    # ---- DE-IDENTIFIED summary ----
    pheno_rows = con.sql(f"SELECT count(*) FROM read_csv('{ICDS_OUT}', header=true)").fetchone()[0]
    pat_with_dx = con.sql("SELECT count(DISTINCT id) FROM ev WHERE ICD_TYPE IS NOT NULL").fetchone()[0]
    typ = con.sql("SELECT ICD_TYPE, count(*) FROM ev WHERE ICD_TYPE IS NOT NULL GROUP BY ICD_TYPE ORDER BY ICD_TYPE").df()
    print("\n==== phenome summary ====")
    print(f"cohort with >=1 pre-study ICD9/10 dx: {pat_with_dx:,} / {nkeys:,} ({100*pat_with_dx/nkeys:.1f}%)")
    print(f"phenotype rows (id x code): {pheno_rows:,}")
    print(f"event ICD_TYPE split:\n{typ.to_string(index=False)}")
    print(f"n_encounters median: {enc['n_encounters'].median():.0f}; "
          f"p90: {enc['n_encounters'].quantile(0.9):.0f}; max: {enc['n_encounters'].max():,}")
    print(f"Wrote {ICDS_OUT}")
    print(f"Wrote {ENC_OUT}")
    U.log_milestone(f"Step 3 phenome: {pheno_rows:,} id-code rows; "
                    f"{pat_with_dx:,}/{nkeys:,} patients with pre-study dx.")


if __name__ == "__main__":
    main()
