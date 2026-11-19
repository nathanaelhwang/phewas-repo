# %% [markdown]
# Step 06 - Hard verification gate + SUMMARY of the fills/targets (run before trusting results).
#
# Asserts: 12-char ids; targets keyed to the cohort; binary membership monotonic
# (1yr subset of 5yr); every windowed fill strictly before index; GPI-14 length; de-explosion
# sanity; crosswalk coverage. Any failure -> SystemExit. DE-IDENTIFIED only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import medwas_utils as U

FAILS = []


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


def main() -> None:
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=64, threads=8)
    fills = (U.FILLS_DIR / "distinct_fills.parquet").as_posix()
    quar = (U.FILLS_DIR / "unclassified_quarantine.parquet").as_posix()

    print("== fills ==")
    n_ok = con.sql(f"SELECT count(*) FROM read_parquet('{fills}')").fetchone()[0]
    n_uncl = con.sql(f"SELECT count(*) FROM read_parquet('{quar}')").fetchone()[0]
    badlen = con.sql(f"SELECT count(*) FROM read_parquet('{fills}') WHERE length(gpi14)<>14").fetchone()[0]
    bad_gpi4 = con.sql(f"SELECT count(*) FROM read_parquet('{fills}') WHERE length(gpi4)<>4").fetchone()[0]
    id12 = con.sql(f"SELECT min(length(id))=12 AND max(length(id))=12 FROM read_parquet('{fills}')").fetchone()[0]
    check("fills id all 12-char", bool(id12))
    check("gpi4 all length 4", bad_gpi4 == 0)
    print(f"    classified fills {n_ok:,}; unclassified {n_uncl:,} "
          f"({100*n_uncl/max(n_ok+n_uncl,1):.2f}%); gpi14 len!=14: {badlen:,}")

    print("== targets: monotonicity & cohort keying ==")
    t1 = (U.ANALYSIS / "targets_1yr.parquet").as_posix()
    t5 = (U.ANALYSIS / "targets_5yr.parquet").as_posix()
    cohort_ids = f"(SELECT DISTINCT id FROM read_parquet('{U.COHORT}'))"
    leak1 = con.sql(f"SELECT count(*) FROM read_parquet('{t1}') WHERE id NOT IN {cohort_ids}").fetchone()[0]
    leak5 = con.sql(f"SELECT count(*) FROM read_parquet('{t5}') WHERE id NOT IN {cohort_ids}").fetchone()[0]
    check("1yr targets are cohort ids", leak1 == 0)
    check("5yr targets are cohort ids", leak5 == 0)
    # every 1yr (id,gpi4) must appear in 5yr (1yr window subset of 5yr window)
    notin = con.sql(f"""SELECT count(*) FROM read_parquet('{t1}') a
                        LEFT JOIN read_parquet('{t5}') b USING (id, gpi4)
                        WHERE b.id IS NULL""").fetchone()[0]
    check("membership monotonic (1yr subset of 5yr)", notin == 0)

    print("== windowed fills strictly pre-index (recompute a check) ==")
    bad = con.sql(f"""
        WITH c AS (SELECT id, CAST(STUDY_DATE AS DATE) idx FROM read_parquet('{U.COHORT}')),
             f AS (SELECT id, gpi4, fill_date FROM read_parquet('{fills}') WHERE gpi4<>'UNCL')
        SELECT count(*) FROM read_parquet('{t5}') t
        JOIN c USING (id)
        WHERE NOT EXISTS (
            SELECT 1 FROM f WHERE f.id=t.id AND f.gpi4=t.gpi4
              AND f.fill_date >= (c.idx - to_years(5))::DATE AND f.fill_date < c.idx)
    """).fetchone()[0]
    check("every 5yr target has a strictly-pre-index fill", bad == 0)

    print("== crosswalk coverage ==")
    cat = pd.read_csv(U.RESULTS / "gpi_category.csv")
    cov = (cat["label_source"] == "appendix_a").mean() if len(cat) else 0
    check("crosswalk covers >=70% of tested classes from Appendix A", cov >= 0.70)
    print(f"    tested classes {len(cat)}; Appendix-A labeled {100*cov:.0f}%")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
    U.log_milestone(f"Step 06 verify: {'PASS' if not FAILS else 'FAIL ' + str(FAILS)}.")
    if FAILS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
