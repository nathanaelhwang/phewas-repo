# %% [markdown]
# Step 2 - Extract BMI from PopulationCohort251103.csv (the only NON-temporal vitals
# source; 103 GB, cartesian-exploded). DISTINCT (id, measure_date, BMI) collapses the
# explosion to each patient's real vitals events; we then pick the BMI closest to the
# patient's index STUDY_DATE. Writes cohort/bmi.parquet. DE-IDENTIFIED summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import utils as U

PC = "D:/phewas_data/PopulationCohort251103.csv"
COHORT = (U.COHORT_DIR / "cohort_base.parquet").as_posix()
OUT = U.COHORT_DIR / "bmi.parquet"


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    total = con.sql(f"SELECT count(*) FROM read_parquet('{COHORT}')").fetchone()[0]

    print("Scanning PopulationCohort for BMI (this reads ~103 GB once)...")
    sql = f"""
    WITH cohort AS (
        SELECT id, CAST(STUDY_DATE AS DATE) AS sdate FROM read_parquet('{COHORT}')
    ),
    raw AS (
        SELECT lpad(regexp_replace(CAST(MRN AS VARCHAR), '\\.0$', ''), 12, '0') AS id,
               try_strptime(measure_date, '%m/%d/%Y')::DATE AS mdate,
               try_cast(BMI AS DOUBLE) AS bmi
        FROM read_csv('{PC}', header=true, quote='"', escape='"',
                      all_varchar=true, ignore_errors=true)
        WHERE BMI IS NOT NULL
    ),
    pc AS (
        SELECT DISTINCT id, mdate, bmi FROM raw
        WHERE id IN (SELECT id FROM cohort)
          AND mdate IS NOT NULL AND bmi BETWEEN 10 AND 80
    ),
    j AS (
        SELECT pc.id, pc.bmi, pc.mdate,
               abs(date_diff('day', pc.mdate, c.sdate)) AS dist
        FROM pc JOIN cohort c USING (id)
    ),
    ranked AS (
        SELECT *, row_number() OVER (PARTITION BY id ORDER BY dist ASC, mdate DESC) AS rn
        FROM j
    )
    SELECT id, bmi, mdate AS bmi_date, dist AS bmi_days_from_study
    FROM ranked WHERE rn = 1
    """
    bmi = con.sql(sql).df()
    U.COHORT_DIR.mkdir(parents=True, exist_ok=True)
    bmi.to_parquet(OUT, index=False)

    # ---- DE-IDENTIFIED coverage summary -----------------------------------
    n = len(bmi)
    print("\n==== BMI extraction summary ====")
    print(f"adult cohort: {total:,}")
    print(f"patients with a usable BMI: {n:,} ({100*n/total:.1f}%)")
    print(f"BMI min/median/max: {bmi['bmi'].min():.1f} / {bmi['bmi'].median():.1f} / {bmi['bmi'].max():.1f}")
    d = bmi["bmi_days_from_study"].abs()
    print(f"|days from study| median/p90/max: {d.median():.0f} / {d.quantile(0.9):.0f} / {d.max():.0f}")
    within1y = (d <= 365).mean()
    print(f"BMI within 1 year of study: {100*within1y:.1f}% of those with BMI")
    print(f"Wrote {OUT}")
    U.log_milestone(f"Step 2 BMI: {n:,}/{total:,} adults ({100*n/total:.1f}%) have a usable BMI.")


if __name__ == "__main__":
    main()
