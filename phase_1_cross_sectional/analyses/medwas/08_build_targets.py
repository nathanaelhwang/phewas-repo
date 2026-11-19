# %% [markdown]
# Step 08 - Build the binary "filled in window" targets per GPI-4 class.
#
# For each window (1yr, 5yr): a (patient, gpi4) pair is a 1 if the patient filled any drug
# in that class in [index - N yr, index) on FILL DATE; every other cohort patient is a 0.
# We store only the 1s (windowed membership over the 222,109 cohort patients) plus the kept
# classes -- step 10 reconstructs the whole-cohort 0/1 vector. A class is kept only if
# >= MIN_PATIENTS (500) distinct cohort patients filled it in that window.
#
# Emits per window: analysis/targets_{1yr,5yr}.parquet (id, gpi4 -- the 1s, kept classes)
#                   analysis/kept_classes_{1yr,5yr}.csv (gpi4, n_pt)
# DE-IDENTIFIED summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import medwas_utils as U


def main() -> None:
    U.ensure_dirs()
    fills = (U.FILLS_DIR / "distinct_fills.parquet").as_posix()

    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    con.execute(f"""
        CREATE TEMP TABLE c AS
        SELECT id, CAST(STUDY_DATE AS DATE) AS idx FROM read_parquet('{U.COHORT}')
    """)
    con.execute(f"""
        CREATE TEMP TABLE f AS
        SELECT id, gpi4, fill_date FROM read_parquet('{fills}') WHERE gpi4 <> 'UNCL'
    """)
    n_cohort = con.sql("SELECT count(*) FROM c").fetchone()[0]

    print(f"cohort patients: {n_cohort:,}")
    for label, years in U.WINDOWS:
        tgt = (U.ANALYSIS / f"targets_{label}.parquet").as_posix()
        kept_csv = (U.ANALYSIS / f"kept_classes_{label}.csv").as_posix()

        # windowed membership: distinct (cohort id, gpi4) with a fill in [idx-Nyr, idx)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE m AS
            SELECT DISTINCT c.id, f.gpi4
            FROM f JOIN c USING (id)
            WHERE f.fill_date >= (c.idx - to_years({years}))::DATE
              AND f.fill_date <  c.idx
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE kept AS
            SELECT gpi4, count(DISTINCT id) AS n_pt
            FROM m GROUP BY gpi4 HAVING count(DISTINCT id) >= {U.MIN_PATIENTS}
        """)
        con.execute(f"""COPY (SELECT m.id, m.gpi4 FROM m JOIN kept USING (gpi4))
                        TO '{tgt}' (FORMAT PARQUET)""")
        con.execute(f"""COPY (SELECT gpi4, n_pt FROM kept ORDER BY n_pt DESC)
                        TO '{kept_csv}' (HEADER, DELIMITER ',')""")

        n_all = con.sql("SELECT count(DISTINCT gpi4) FROM m").fetchone()[0]
        n_kept = con.sql("SELECT count(*) FROM kept").fetchone()[0]
        n_mem = con.sql("SELECT count(*) FROM (SELECT m.id FROM m JOIN kept USING (gpi4))").fetchone()[0]
        med = con.sql("""SELECT median(pp) FROM (
                            SELECT id, count(*) pp FROM (SELECT m.id FROM m JOIN kept USING(gpi4))
                            GROUP BY id)""").fetchone()[0]
        print(f"\n[{label}] window [idx-{years}yr, idx)")
        print(f"  GPI-4 classes with >=1 filler: {n_all}")
        print(f"  kept (>= {U.MIN_PATIENTS} patients):  {n_kept}")
        print(f"  membership rows (the 1s):      {n_mem:,}")
        print(f"  median kept-classes per patient (among users): {med:.0f}")

    U.log_milestone("Step 08 targets: binary 'filled in window' membership for 1yr & 5yr "
                    f"(>= {U.MIN_PATIENTS}-patient floor).")


if __name__ == "__main__":
    main()
