# %% [markdown]
# Step 7 - Export the two feature matrices to CSV alongside the parquet.
# Uses DuckDB COPY (streaming, memory-light). NULL cells are written empty.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import labwas_utils as U


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    for label in ("1yr", "5yrs"):
        src = (U.RESULTS / f"labs_for_phewas_{label}.parquet").as_posix()
        dst = (U.RESULTS / f"labs_for_phewas_{label}.csv").as_posix()
        con.execute(f"COPY (SELECT * FROM read_parquet('{src}')) TO '{dst}' (HEADER, DELIMITER ',')")
        n_rows, n_cols = con.sql(
            f"SELECT count(*), (SELECT count(*) FROM (DESCRIBE SELECT * FROM read_parquet('{src}'))) "
            f"FROM read_parquet('{src}')").fetchone()
        size_mb = Path(dst).stat().st_size / 1e6
        print(f"[{label}] {n_rows:,} rows x {n_cols} cols -> {Path(dst).name} ({size_mb:.0f} MB)")
    U.log_milestone("Step 7 export: feature matrices written to CSV alongside parquet.")


if __name__ == "__main__":
    main()
