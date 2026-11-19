# %% [markdown]
# Compute per-analyte value distribution stats for the KEPT analytes, as input to
# the per-analyte cleaning-spec derivation. De-identified aggregates only.
# Writes results/labwas_results/preclean_stats.csv.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import labwas_utils as U


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{U.duckdb_tmp(__file__)}'")
    num = (U.LABS_DIR / "labs_long_numeric.parquet").as_posix()
    cw = (U.CROSSWALK_DIR / "loinc_crosswalk.csv").as_posix()
    dd = pd.read_csv(U.RESULTS / "column_data_dictionary.csv")   # the kept analytes
    kept = set(dd["loinc_code"].astype(str))

    con.execute(f"""
        CREATE TEMP TABLE j AS
        SELECT x.loinc_code, n.value_num
        FROM read_parquet('{num}') n
        JOIN (SELECT lab, loinc_code FROM read_csv('{cw}', header=true, all_varchar=true)
              WHERE loinc_code <> '') x USING (lab)
    """)
    stats = con.sql("""
        SELECT loinc_code, count(*) AS n_obs,
               round(min(value_num), 4)                       AS mn,
               round(quantile_cont(value_num, 0.01), 4)       AS p1,
               round(quantile_cont(value_num, 0.05), 4)       AS p5,
               round(quantile_cont(value_num, 0.50), 4)       AS p50,
               round(quantile_cont(value_num, 0.95), 4)       AS p95,
               round(quantile_cont(value_num, 0.99), 4)       AS p99,
               round(max(value_num), 4)                       AS mx,
               round(avg((value_num < 0)::INT), 4)            AS frac_neg,
               round(avg((value_num = 0)::INT), 4)            AS frac_zero
        FROM j GROUP BY loinc_code
    """).df()
    stats = stats[stats["loinc_code"].isin(kept)]

    meta = dd[["loinc_code", "loinc_long_common_name", "loinc_component", "loinc_system",
               "loinc_scale", "source_descriptors", "n_mrn_1yr", "n_mrn_5yrs"]]
    out = meta.merge(stats, on="loinc_code", how="inner").sort_values("n_mrn_5yrs", ascending=False)
    dst = U.RESULTS / "preclean_stats.csv"
    out.to_csv(dst, index=False)
    print(f"Wrote {dst}  ({len(out)} kept analytes)")


if __name__ == "__main__":
    main()
