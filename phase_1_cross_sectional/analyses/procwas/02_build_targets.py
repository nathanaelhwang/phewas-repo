# %% [markdown]
# ProcWAS Step 02 - Per-patient association targets off the windowed procedure store.
#
# For windows 1yr & 5yr, keyed on cohort id:
#   per (id, Clear descriptor) distinct-order count -> rate outcome (procedures/yr); kept
#     descriptors = >=500 patients with >=1 order in-window (the WAS phenome).
#   per id total distinct-order count -> headline "all procedures/yr" (matched + unmatched).
# Present rows only (count>=1) over the 222,109 cohort; the model step reconstructs whole-cohort 0s.
# Also writes the descriptor -> (Procedure type, Body system) map used by both plot Parts.
# DE-IDENTIFIED summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import procwas_utils as U


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=32, threads=8)
    con.execute(f"CREATE TEMP TABLE s AS SELECT * FROM read_parquet('{U.STORE_5YR}')")

    # audit: which sleep/OSA-pathway descriptors are being removed (circular with the exposure)
    excl = con.sql(f"""SELECT clear_descriptor, count(DISTINCT id) n_pat, count(DISTINCT proc_key) n_orders
                       FROM s WHERE clear_descriptor IS NOT NULL AND {U.SLEEP_SQL}
                       GROUP BY 1 ORDER BY 2 DESC""").df()
    print(f"EXCLUDING {len(excl)} sleep/OSA-pathway descriptors:")
    for _, r in excl.iterrows():
        print(f"  - {r['clear_descriptor']}  (n_pat={r['n_pat']:,}, n_orders={r['n_orders']:,})")

    # descriptor -> broader categories (window-independent; one row per Clear descriptor)
    con.execute("""
        COPY (SELECT clear_descriptor, any_value(procedure_type) AS procedure_type,
                     any_value(body_system) AS body_system
              FROM s WHERE clear_descriptor IS NOT NULL GROUP BY clear_descriptor)
        TO '%s' (HEADER, DELIMITER ',')""" % (U.ANALYSIS / "descriptor_categories.csv").as_posix())

    for label, _ in U.WINDOWS:
        wf = "in_1yr" if label == "1yr" else "TRUE"

        # ---- per-descriptor counts + >=500-patient floor ----
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE per_desc AS
            SELECT id, clear_descriptor, count(DISTINCT proc_key) AS n_proc
            FROM s WHERE clear_descriptor IS NOT NULL AND NOT {U.SLEEP_SQL} AND {wf}
            GROUP BY id, clear_descriptor
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE kept AS
            SELECT clear_descriptor FROM per_desc
            GROUP BY clear_descriptor HAVING count(DISTINCT id) >= {U.MIN_PATIENTS}
        """)
        tgt = (U.ANALYSIS / f"targets_{label}.parquet").as_posix()
        con.execute(f"""COPY (SELECT p.id, p.clear_descriptor, p.n_proc
                             FROM per_desc p JOIN kept k USING (clear_descriptor))
                        TO '{tgt}' (FORMAT PARQUET)""")
        con.execute(f"""
            COPY (SELECT p.clear_descriptor,
                         any_value(c.procedure_type) AS procedure_type,
                         any_value(c.body_system) AS body_system,
                         count(DISTINCT p.id) AS n_pat, sum(p.n_proc) AS n_orders
                  FROM per_desc p JOIN kept k USING (clear_descriptor)
                  LEFT JOIN (SELECT clear_descriptor, any_value(procedure_type) procedure_type,
                                    any_value(body_system) body_system
                             FROM s WHERE clear_descriptor IS NOT NULL GROUP BY clear_descriptor) c
                    USING (clear_descriptor)
                  GROUP BY p.clear_descriptor ORDER BY n_pat DESC)
            TO '{(U.ANALYSIS / f"kept_descriptors_{label}.csv").as_posix()}' (HEADER, DELIMITER ',')""")

        # ---- whole-cohort total procedures/yr (matched + unmatched; sleep/OSA-pathway excluded) ----
        total = (U.ANALYSIS / f"total_{label}.parquet").as_posix()
        con.execute(f"""COPY (SELECT id, count(DISTINCT proc_key) AS n_proc
                             FROM s WHERE {wf} AND NOT {U.SLEEP_SQL} GROUP BY id)
                        TO '{total}' (FORMAT PARQUET)""")

        n_kept = con.sql("SELECT count(*) FROM kept").fetchone()[0]
        n_rows = con.sql("SELECT count(*) FROM per_desc p JOIN kept k USING (clear_descriptor)").fetchone()[0]
        n_pat_any = con.sql(f"SELECT count(DISTINCT id) FROM s WHERE {wf}").fetchone()[0]
        print(f"[{label}] kept descriptors (>={U.MIN_PATIENTS} pts): {n_kept} | present rows: {n_rows:,} "
              f"| patients with >=1 procedure: {n_pat_any:,}")

    U.log_milestone(f"Step 02 targets: per-descriptor procedure counts (>={U.MIN_PATIENTS}-pt floor) "
                    f"+ total-procedures, 1yr & 5yr.")
    con.close()


if __name__ == "__main__":
    main()
