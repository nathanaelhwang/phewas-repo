# %% [markdown]
# Step 2 - Distinct numeric lab descriptors + priority weights.
# Cheap pass over the de-exploded parquet (NOT the 101 GB CSV). The priority key
# is distinct-MRN count, so crosswalk review can be bounded to high-yield labs.
# Emits crosswalk/lab_descriptors.csv (descriptor, n_mrn, n_rows). De-identified.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb

import labwas_utils as U


def main() -> None:
    U.ensure_dirs()
    num = (U.LABS_DIR / "labs_long_numeric.parquet").as_posix()
    out = (U.CROSSWALK_DIR / "lab_descriptors.csv").as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"""
        COPY (
            SELECT lab AS descriptor,
                   count(DISTINCT id) AS n_mrn,
                   count(*)           AS n_rows
            FROM read_parquet('{num}')
            GROUP BY lab
            ORDER BY n_mrn DESC
        ) TO '{out}' (HEADER, DELIMITER ',')
    """)

    n = con.sql(f"SELECT count(*) FROM read_csv('{out}', header=true)").fetchone()[0]
    print(f"distinct numeric lab descriptors: {n:,}")
    print(f"Wrote {out}")
    U.log_milestone(f"Step 2 descriptors: {n:,} distinct numeric lab descriptors.")


if __name__ == "__main__":
    main()
