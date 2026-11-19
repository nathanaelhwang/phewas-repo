# %% [markdown]
# Unit-consistency audit (FLAG ONLY, no changes).
# PopulationCohort has no units column, so unit problems are inferred from the
# numeric value distributions:
#   (1) cross-descriptor median divergence within a collapsed LOINC (the direct
#       signal for the mean-collapsing risk),
#   (2) within-analyte multimodality (two value clusters ~a constant factor apart).
# Reference-range interpretation is left to a later step. De-identified aggregates only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import labwas_utils as U

MIN_DESC_N = 50          # descriptors need this many obs to compare medians
SPREAD_FLAG = 3.0        # cross-descriptor median ratio >= this -> flag
BAND = 3.1623           # half-decade (sqrt 10) for multimodality bands


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{U.duckdb_tmp(__file__)}'")
    num = (U.LABS_DIR / "labs_long_numeric.parquet").as_posix()
    cw = (U.CROSSWALK_DIR / "loinc_crosswalk.csv").as_posix()

    con.execute(f"""
        CREATE TEMP TABLE j AS
        SELECT x.loinc_code, n.lab, n.value_num
        FROM read_parquet('{num}') n
        JOIN (SELECT lab, loinc_code FROM read_csv('{cw}', header=true, all_varchar=true)
              WHERE loinc_code IS NOT NULL AND loinc_code <> '') x USING (lab)
    """)

    # per-descriptor medians (to compare within a LOINC)
    per_desc = con.sql(f"""
        SELECT loinc_code, lab AS descriptor, count(*) AS n,
               median(value_num) AS med
        FROM j GROUP BY loinc_code, lab
    """).df()

    # per-LOINC distribution
    per_loinc = con.sql("""
        SELECT loinc_code, count(*) AS n_obs,
               min(value_num) AS mn,
               quantile_cont(value_num, 0.01) AS p1,
               quantile_cont(value_num, 0.05) AS p5,
               quantile_cont(value_num, 0.50) AS p50,
               quantile_cont(value_num, 0.95) AS p95,
               quantile_cont(value_num, 0.99) AS p99,
               max(value_num) AS mx
        FROM j GROUP BY loinc_code
    """).df()

    # multimodality band fractions (share far above / far below the median)
    bands = con.sql(f"""
        WITH meds AS (SELECT loinc_code, median(value_num) AS med FROM j GROUP BY loinc_code)
        SELECT j.loinc_code,
               avg((j.value_num > {BAND} * m.med)::INT)                      AS frac_high,
               avg((j.value_num > 0 AND j.value_num < m.med / {BAND})::INT)  AS frac_low,
               avg((j.value_num <= 0)::INT)                                  AS frac_nonpos
        FROM j JOIN meds m USING (loinc_code)
        WHERE m.med > 0
        GROUP BY j.loinc_code
    """).df()

    # cross-descriptor divergence
    pd_ok = per_desc[(per_desc["n"] >= MIN_DESC_N) & (per_desc["med"] > 0)]
    g = pd_ok.groupby("loinc_code")["med"]
    xdesc = pd.DataFrame({
        "n_desc_compared": g.size(),
        "min_desc_med": g.min(),
        "max_desc_med": g.max(),
    }).reset_index()
    xdesc["cross_desc_ratio"] = (xdesc["max_desc_med"] / xdesc["min_desc_med"]).round(2)

    # names + coverage
    dd = pd.read_csv(U.RESULTS / "column_data_dictionary.csv")
    meta = dd[["loinc_code", "loinc_long_common_name", "source_descriptors",
               "n_source_descriptors", "n_mrn_5yrs"]]

    out = (per_loinc.merge(bands, on="loinc_code", how="left")
                    .merge(xdesc, on="loinc_code", how="left")
                    .merge(meta, on="loinc_code", how="left"))
    for c in ("mn", "p1", "p5", "p50", "p95", "p99", "mx"):
        out[c] = out[c].round(3)

    # risk flags
    out["flag_cross_descriptor"] = out["cross_desc_ratio"].fillna(0) >= SPREAD_FLAG
    out["flag_multimodal"] = (out["frac_high"].fillna(0) >= 0.15) & (out["frac_low"].fillna(0) >= 0.15)
    out["risk"] = out.apply(
        lambda r: "HIGH" if r["flag_cross_descriptor"]
        else ("MEDIUM" if r["flag_multimodal"] else ""), axis=1)

    out = out.sort_values(
        ["flag_cross_descriptor", "flag_multimodal", "n_mrn_5yrs"],
        ascending=[False, False, False])
    cols = ["loinc_code", "loinc_long_common_name", "risk", "n_mrn_5yrs",
            "n_source_descriptors", "cross_desc_ratio", "min_desc_med", "max_desc_med",
            "frac_high", "frac_low", "frac_nonpos", "p5", "p50", "p95", "mn", "mx",
            "source_descriptors"]
    out = out[cols]
    dst = U.RESULTS / "unit_risk_flags.csv"
    out.to_csv(dst, index=False)

    nH = int((out["risk"] == "HIGH").sum())
    nM = int((out["risk"] == "MEDIUM").sum())
    print(f"analytes audited: {len(out):,}")
    print(f"  HIGH (cross-descriptor median divergence >= {SPREAD_FLAG}x): {nH}")
    print(f"  MEDIUM (within-analyte multimodal): {nM}")
    print(f"Wrote {dst}")
    print("\n=== HIGH-risk analytes (collapsed descriptors disagree) ===")
    for r in out[out["risk"] == "HIGH"].itertuples(index=False):
        print(f"  {r.loinc_code:<9} ratio={r.cross_desc_ratio:<6} "
              f"meds[{r.min_desc_med:g}..{r.max_desc_med:g}]  n_mrn={int(r.n_mrn_5yrs or 0):>7,}  "
              f"{str(r.loinc_long_common_name)[:40]}")
    U.log_milestone(f"Unit audit: {nH} HIGH (cross-descriptor), {nM} MEDIUM (multimodal) flagged.")


if __name__ == "__main__":
    main()
