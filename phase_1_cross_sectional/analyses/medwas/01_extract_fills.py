# %% [markdown]
# Step 01 - Extract distinct pharmacy fills from medication.csv (~34.8 GB) in ONE pass.
#
# The source is denormalized (sleep-study x pharmacy-fill cartesian product, ~152M rows),
# so the load-bearing move is `SELECT DISTINCT` over the fill block (dropping the study /
# PAP columns) to recover true dispensings. GPI/NDC are normalized to zero-padded strings
# (robust to scientific-notation float damage); fill_date = COALESCE(RFDISPDT, RFSOLDDT,
# DPYR-year). ~2% null-GPI fills are backfilled NDC->GPI then GENNAME->GPI (modal), residual
# routed to an 'UNCL' quarantine (reported, never tested).
#
# Emits (parquet):
#   fills/distinct_fills.parquet          : id, gpi14, gpi4, gpi2, ndc, genname, brand,
#                                           rxnum, refill, fill_date, date_imprecise,
#                                           rxsup, units, gpi_source   (gpi4 <> 'UNCL')
#   fills/unclassified_quarantine.parquet : same schema, the UNCL residual
# DE-IDENTIFIED summary only (counts/rates; never row-level values).

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import medwas_utils as U


def sas(col: str) -> str:
    """DuckDB expr: parse a SAS 'DDMONYYYY:HH:MM:SS' or bare 'DDMONYYYY' string."""
    return (f"COALESCE(try_strptime(trim({col}), '{U.SAS}'), "
            f"try_strptime(trim({col}), '{U.SAS_DATE}'))")


# GPI -> canonical 14-char zero-padded string. 14-digit GPI (<= 9.9999e13) is below
# 2^53, so it is represented EXACTLY in a double; round->HUGEINT->VARCHAR recovers all
# digits, and lpad restores stripped leading zeros. gpi4 = first 4 (most-significant)
# digits, so it is safe from any float tail rounding.
def code_norm(col: str, width: int) -> str:
    hi = 10 ** width - 1  # largest value that fits in `width` digits
    return (f"CASE WHEN {col} IS NULL OR trim({col}) = '' THEN NULL "
            f"WHEN regexp_matches(trim({col}), '^[0-9]+$') THEN lpad(trim({col}), {width}, '0') "
            f"WHEN isfinite(try_cast({col} AS DOUBLE)) "
            f"AND try_cast({col} AS DOUBLE) BETWEEN 0 AND {hi} "
            f"THEN lpad(CAST(try_cast(round(try_cast({col} AS DOUBLE)) AS HUGEINT) AS VARCHAR), {width}, '0') "
            f"ELSE NULL END")


def main() -> None:
    U.ensure_dirs()
    out = (U.FILLS_DIR / "distinct_fills.parquet").as_posix()
    quar = (U.FILLS_DIR / "unclassified_quarantine.parquet").as_posix()

    con = duckdb.connect()
    U.configure_duckdb(con, __file__)

    t0 = time.time()
    print("Scanning medication.csv (~34.8 GB, one pass)...", flush=True)
    con.execute(f"""
        CREATE TEMP TABLE raw AS
        SELECT {U.MRN_SQL}                              AS id,
               nullif(trim(RXNUM), '')                  AS rxnum,
               try_cast(REFILL AS INTEGER)              AS refill,
               {code_norm('GPI', 14)}                   AS gpi14,
               {code_norm('NDC', 11)}                   AS ndc,
               upper(nullif(trim(GENNAME), ''))         AS genname,
               nullif(trim(BRAND), '')                  AS brand,
               ({sas('RFDISPDT')})::DATE                AS disp_date,
               ({sas('RFSOLDDT')})::DATE                AS sold_date,
               try_cast(DPYR AS INTEGER)                AS dpyr,
               try_cast(RXSUP AS DOUBLE)                AS rxsup,
               try_cast(UNITS AS DOUBLE)                AS units
        FROM read_csv('{U.MED_CSV}', header=true, quote='"', escape='"',
                      all_varchar=true, ignore_errors=true)
    """)
    n_raw = con.sql("SELECT count(*) FROM raw").fetchone()[0]
    print(f"  raw rows materialized: {n_raw:,} [{time.time()-t0:.0f}s]", flush=True)

    # ---- fill_date + de-explosion (DISTINCT over the fill block) -----------
    con.execute("""
        CREATE TEMP TABLE fills0 AS
        SELECT DISTINCT id, rxnum, refill, gpi14, ndc, genname, brand,
               COALESCE(disp_date, sold_date,
                        CASE WHEN dpyr BETWEEN 1900 AND 2100
                             THEN make_date(dpyr, 7, 1) ELSE NULL END) AS fill_date,
               (disp_date IS NULL AND sold_date IS NULL)               AS date_imprecise,
               rxsup, units
        FROM raw
        WHERE id IS NOT NULL
    """)
    con.execute("DROP TABLE raw")
    con.execute("""
        CREATE TEMP TABLE fills1 AS
        SELECT * FROM fills0
        WHERE fill_date IS NOT NULL AND fill_date <= current_date
    """)
    n_fills0 = con.sql("SELECT count(*) FROM fills0").fetchone()[0]
    n_fills = con.sql("SELECT count(*) FROM fills1").fetchone()[0]
    print(f"  distinct fills: {n_fills0:,}; with usable date: {n_fills:,} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- null-GPI backfill: NDC->modal gpi14, then GENNAME->modal gpi14 ----
    con.execute("""
        CREATE TEMP TABLE ndc_map AS
        SELECT ndc, arg_max(gpi14, cnt) AS gpi14 FROM (
            SELECT ndc, gpi14, count(*) cnt FROM fills1
            WHERE gpi14 IS NOT NULL AND ndc IS NOT NULL GROUP BY ndc, gpi14
        ) GROUP BY ndc
    """)
    con.execute("""
        CREATE TEMP TABLE gen_map AS
        SELECT genname, arg_max(gpi14, cnt) AS gpi14 FROM (
            SELECT genname, gpi14, count(*) cnt FROM fills1
            WHERE gpi14 IS NOT NULL AND genname IS NOT NULL GROUP BY genname, gpi14
        ) GROUP BY genname
    """)
    con.execute("""
        CREATE TEMP TABLE fills2 AS
        SELECT f.id, f.rxnum, f.refill,
               COALESCE(f.gpi14, nm.gpi14, gm.gpi14)                        AS gpi14,
               CASE WHEN f.gpi14 IS NOT NULL THEN 'native'
                    WHEN nm.gpi14 IS NOT NULL THEN 'ndc_backfill'
                    WHEN gm.gpi14 IS NOT NULL THEN 'genname_backfill'
                    ELSE 'unclassified' END                                AS gpi_source,
               f.ndc, f.genname, f.brand, f.fill_date, f.date_imprecise,
               f.rxsup, f.units
        FROM fills1 f
        LEFT JOIN ndc_map nm ON nm.ndc = f.ndc AND f.gpi14 IS NULL
        LEFT JOIN gen_map gm ON gm.genname = f.genname AND f.gpi14 IS NULL AND nm.gpi14 IS NULL
    """)
    con.execute("""
        CREATE TEMP TABLE fills3 AS
        SELECT id, rxnum, refill, gpi14,
               CASE WHEN gpi14 IS NULL THEN 'UNCL' ELSE substr(gpi14, 1, 4) END AS gpi4,
               CASE WHEN gpi14 IS NULL THEN 'UNCL' ELSE substr(gpi14, 1, 2) END AS gpi2,
               gpi_source, ndc, genname, brand, fill_date, date_imprecise, rxsup, units
        FROM fills2
    """)

    con.execute(f"""COPY (SELECT * FROM fills3 WHERE gpi4 <> 'UNCL')
                    TO '{out}' (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT * FROM fills3 WHERE gpi4 = 'UNCL')
                    TO '{quar}' (FORMAT PARQUET)""")

    # ---- DE-IDENTIFIED summary ---------------------------------------------
    src = con.sql("""SELECT gpi_source, count(*) n FROM fills3 GROUP BY gpi_source""").df()
    n_uncl = int(src.loc[src.gpi_source == "unclassified", "n"].sum())
    n_ok = n_fills - n_uncl
    d = con.sql("""SELECT count(DISTINCT gpi14) g14, count(DISTINCT gpi4) g4,
                          count(DISTINCT gpi2) g2, count(DISTINCT id) pid,
                          sum(CASE WHEN date_imprecise THEN 1 ELSE 0 END) imp,
                          sum(CASE WHEN length(gpi14) <> 14 THEN 1 ELSE 0 END) badlen
                   FROM fills3 WHERE gpi4 <> 'UNCL'""").df().iloc[0]

    print("\n==== Step 01 extract summary ====", flush=True)
    print(f"raw exploded rows:              {n_raw:,}")
    print(f"distinct fills (usable date):   {n_fills:,}  "
          f"(de-explosion ratio {n_raw/max(n_fills,1):.1f}x)")
    print(f"  classified:                   {n_ok:,} ({100*n_ok/n_fills:.1f}%)")
    print(f"  unclassified (quarantined):   {n_uncl:,} ({100*n_uncl/n_fills:.1f}%)")
    print("  gpi source breakdown:")
    for _, r in src.sort_values("n", ascending=False).iterrows():
        print(f"    {r['gpi_source']:18s} {int(r['n']):,}")
    print(f"date-imprecise (DPYR-only):     {int(d['imp']):,} "
          f"({100*int(d['imp'])/max(n_ok,1):.1f}% of classified)")
    print(f"distinct patients (any fill):   {int(d['pid']):,}")
    print(f"distinct GPI-14 / GPI-4 / GPI-2: {int(d['g14']):,} / {int(d['g4']):,} / {int(d['g2']):,}")
    print(f"gpi14 with length != 14 (check): {int(d['badlen']):,}")
    print(f"total time: {time.time()-t0:.0f}s")

    U.log_milestone(
        f"Step 01 extract: {n_raw:,} raw -> {n_fills:,} distinct fills "
        f"({n_ok:,} classified / {n_uncl:,} UNCL); "
        f"{int(d['g4'])} GPI-4 classes, {int(d['pid']):,} patients."
    )


if __name__ == "__main__":
    main()
