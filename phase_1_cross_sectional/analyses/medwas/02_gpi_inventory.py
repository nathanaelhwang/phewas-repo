# %% [markdown]
# Step 02 - De-identified GPI inventory (sanity-check the binning before locking).
#
# Reads fills/distinct_fills.parquet (never the 34.8 GB CSV) and reports, among the
# 222,109 v2-cohort patients, the distinct-code cardinality at 2/4/6 digits and per-GPI-4
# patient/fill counts, so we can see how many Drug Classes will clear the >=500-patient
# floor before building matrices. Writes crosswalk/gpi_inventory.csv. DE-IDENTIFIED only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import medwas_utils as U


def main() -> None:
    fills = (U.FILLS_DIR / "distinct_fills.parquet").as_posix()
    out = (U.CROSSWALK_DIR / "gpi_inventory.csv").as_posix()

    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=64, threads=8)

    # cohort-relevant fills only (inner join to the 222k analytic cohort)
    con.execute(f"""
        CREATE TEMP TABLE cf AS
        SELECT f.id, f.gpi14, f.gpi4, f.gpi2, substr(f.gpi14, 1, 6) AS gpi6, f.gpi_source
        FROM read_parquet('{fills}') f
        JOIN (SELECT DISTINCT id FROM read_parquet('{U.COHORT}')) c USING (id)
    """)

    # per-GPI-4 inventory
    con.execute(f"""
        COPY (
            SELECT gpi4, any_value(gpi2) AS gpi2,
                   count(DISTINCT gpi14) AS n_gpi14, count(DISTINCT gpi6) AS n_gpi6,
                   count(DISTINCT id)    AS n_pt,    count(*)              AS n_fills,
                   avg(CASE WHEN gpi_source <> 'native' THEN 1.0 ELSE 0.0 END) AS frac_backfilled
            FROM cf GROUP BY gpi4 ORDER BY n_pt DESC
        ) TO '{out}' (HEADER, DELIMITER ',')
    """)

    tot = con.sql("""SELECT count(DISTINCT gpi2) g2, count(DISTINCT gpi4) g4,
                            count(DISTINCT gpi6) g6, count(DISTINCT id) pid, count(*) nf
                     FROM cf""").df().iloc[0]
    clear = con.sql(f"""SELECT count(*) FROM (
                          SELECT gpi4 FROM cf GROUP BY gpi4
                          HAVING count(DISTINCT id) >= {U.MIN_PATIENTS})""").fetchone()[0]
    top = con.sql("""SELECT gpi4, gpi2, count(DISTINCT id) n_pt FROM cf
                     GROUP BY gpi4, gpi2 ORDER BY n_pt DESC LIMIT 20""").df()

    print("==== Step 02 GPI inventory (cohort patients only) ====")
    print(f"distinct GPI-2 / GPI-4 / GPI-6:  {int(tot['g2'])} / {int(tot['g4'])} / {int(tot['g6'])}")
    print(f"cohort patients with >=1 fill:   {int(tot['pid']):,}")
    print(f"cohort fills:                    {int(tot['nf']):,}")
    print(f"GPI-4 classes clearing >=500-patient floor (all-time): {clear}")
    print("\nTop 20 GPI-4 classes by patient count (all-time, pre-window):")
    for _, r in top.iterrows():
        print(f"  {r['gpi4']}  (grp {r['gpi2']})  {int(r['n_pt']):,} patients")
    print(f"\nWrote {out}")
    U.log_milestone(f"Step 02 inventory: {int(tot['g4'])} GPI-4 classes; "
                    f"{clear} clear >=500-pt floor (all-time).")


if __name__ == "__main__":
    main()
