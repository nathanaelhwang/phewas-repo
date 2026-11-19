# %% [markdown]
# Step 5 - Build the two LOINC-keyed lab feature matrices + the data dictionary.
#
# For each lookback window (1yr / 5yrs prior to index date), aggregate per
# (MRN, LOINC): mean of all numeric values + count of observations; then pivot
# wide so each LOINC analyte is a `loinc_<code>_mean` / `loinc_<code>_count`
# pair. Synonymous descriptors collapse via the LOINC key. Rows = MRNs with >=1
# numeric lab in the window.
#
# Emits:
#   labs/labs_agg_{1yr,5yrs}_long.parquet
#   labs_for_phewas_1yr.parquet, labs_for_phewas_5yrs.parquet
#   column_data_dictionary.csv

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import labwas_utils as U

WINDOWS = [("1yr", 1), ("5yrs", 5)]
INCLUDE_INDEX_DAY = False  # default: strictly prior to index date


def slug(code: str) -> str:
    return "loinc_" + str(code).replace("-", "_")


def build_agg(con, label: str, years: int) -> Path:
    num = (U.LABS_DIR / "labs_long_numeric.parquet").as_posix()
    cw = (U.CROSSWALK_DIR / "loinc_crosswalk.csv").as_posix()
    anchor = (U.COHORT_DIR / "lab_anchor.parquet").as_posix()
    agg_path = U.LABS_DIR / f"labs_agg_{label}_long.parquet"
    upper = "<=" if INCLUDE_INDEX_DAY else "<"

    # optional per-descriptor unit-conversion rules (unit_rules.csv): for a listed
    # descriptor, multiply value_num by `factor` when value_num >= `threshold`
    # (threshold=0 => unconditional). Handles both flat scaling (flow lymphocytes)
    # and value-conditional harmonization (D-dimer ng/mL vs ug/mL). Default: unchanged.
    rules_path = U.CROSSWALK_DIR / "unit_rules.csv"
    if rules_path.exists():
        scale_cte = (f", s AS (SELECT descriptor, try_cast(threshold AS DOUBLE) AS threshold, "
                     f"try_cast(factor AS DOUBLE) AS factor "
                     f"FROM read_csv('{rules_path.as_posix()}', header=true, all_varchar=true))")
        scale_join = "LEFT JOIN s ON s.descriptor = n.lab"
        val_expr = ("CASE WHEN s.factor IS NOT NULL AND n.value_num >= s.threshold "
                    "THEN n.value_num * s.factor ELSE n.value_num END")
    else:
        scale_cte = scale_join = ""
        val_expr = "n.value_num"

    con.execute(f"""
        COPY (
          WITH n AS (SELECT id, lab, lab_date, value_num FROM read_parquet('{num}')),
               x AS (SELECT lab, loinc_code FROM read_csv('{cw}', header=true, all_varchar=true)
                     WHERE loinc_code IS NOT NULL AND loinc_code <> ''),
               a AS (SELECT id, index_date FROM read_parquet('{anchor}')){scale_cte}
          SELECT a.id, x.loinc_code,
                 avg({val_expr}) AS mean_val,
                 count(*)        AS n_obs
          FROM n JOIN x USING (lab) JOIN a USING (id) {scale_join}
          WHERE n.lab_date >= (a.index_date - to_years({years}))::DATE
            AND n.lab_date {upper} a.index_date
          GROUP BY a.id, x.loinc_code
        ) TO '{agg_path.as_posix()}' (FORMAT PARQUET)
    """)
    return agg_path


def apply_threshold(agg: pd.DataFrame) -> pd.DataFrame:
    """Keep only LOINC analytes meeting the dual floor in this window:
    >= MIN_PATIENTS distinct patients AND >= MIN_OBS total observations."""
    stats = agg.groupby("loinc_code").agg(n_pt=("id", "size"), n_obs=("n_obs", "sum"))
    keep = stats.index[(stats["n_pt"] >= U.MIN_PATIENTS) & (stats["n_obs"] >= U.MIN_OBS)]
    return agg[agg["loinc_code"].isin(set(keep))]


def pivot_wide(agg: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    a = agg.copy()
    a["base"] = a["loinc_code"].map(slug)
    val = a.pivot(index="id", columns="base", values="mean_val").astype("float32")
    cnt = a.pivot(index="id", columns="base", values="n_obs").astype("Int32")
    val.columns = [f"{c}_mean" for c in val.columns]
    cnt.columns = [f"{c}_count" for c in cnt.columns]
    wide = pd.concat([val, cnt], axis=1)
    bases = sorted({c[:-5] for c in val.columns})            # strip '_mean'
    ordered = [f"{b}_{suf}" for b in bases for suf in ("mean", "count")]
    wide = wide[ordered].reset_index()
    mat = anchor.merge(wide, on="id", how="inner")           # MRNs with >=1 lab in window
    return mat[["id", "index_date"] + [c for c in mat.columns if c not in ("id", "index_date")]]


def build_dictionary(con, aggs: dict[str, pd.DataFrame]) -> None:
    cw = pd.read_csv(U.CROSSWALK_DIR / "loinc_crosswalk.csv", dtype=str).fillna("")
    cw = cw[cw["loinc_code"] != ""].copy()
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    cw["_cr"] = cw["confidence"].map(conf_rank).fillna(3)

    g = cw.groupby("loinc_code")
    d = pd.DataFrame({
        "source_descriptors": g["lab"].apply(lambda s: "|".join(sorted(set(s.astype(str))))),
        "n_source_descriptors": g["lab"].nunique(),
        "loinc_long_common_name": g["loinc_long_common_name"].first(),
        "loinc_component": g["loinc_component"].first(),
        "loinc_system": g["loinc_system"].first(),
        "loinc_scale": g["loinc_scale"].first(),
        "mapping_confidence": g["_cr"].min().map({0: "high", 1: "medium", 2: "low", 3: "low"}),
    }).reset_index()

    for label, _ in WINDOWS:
        gg = aggs[label].groupby("loinc_code")
        d[f"n_mrn_{label}"] = d["loinc_code"].map(gg["id"].nunique()).fillna(0).astype(int)
        d[f"n_obs_{label}"] = d["loinc_code"].map(gg["n_obs"].sum()).fillna(0).astype(int)

    # keep only analytes retained in >= 1 window (passed the dual threshold there)
    kept = set().union(*[set(aggs[l]["loinc_code"].unique()) for l, _ in WINDOWS])
    d = d[d["loinc_code"].isin(kept)].copy()
    d["value_column"] = d["loinc_code"].map(lambda c: f"{slug(c)}_mean")
    d["count_column"] = d["loinc_code"].map(lambda c: f"{slug(c)}_count")
    cols = ["loinc_code", "value_column", "count_column", "loinc_long_common_name",
            "loinc_component", "loinc_system", "loinc_scale", "source_descriptors",
            "n_source_descriptors"]
    for l, _ in WINDOWS:
        cols += [f"n_mrn_{l}", f"n_obs_{l}"]
    cols += ["mapping_confidence"]
    d = d[cols].sort_values(f"n_mrn_{WINDOWS[-1][0]}", ascending=False)
    d.to_csv(U.RESULTS / "column_data_dictionary.csv", index=False)
    print(f"Wrote {U.RESULTS / 'column_data_dictionary.csv'}  ({len(d):,} LOINC analytes kept)")


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{U.duckdb_tmp(__file__)}'")
    anchor = pd.read_parquet(U.COHORT_DIR / "lab_anchor.parquet")

    print(f"Dual threshold: >= {U.MIN_PATIENTS} patients AND >= {U.MIN_OBS} observations per analyte/window")
    aggs = {}
    for label, years in WINDOWS:
        agg_path = build_agg(con, label, years)
        agg_full = pd.read_parquet(agg_path)
        agg = apply_threshold(agg_full)          # QualityLab-style dual floor, per window
        aggs[label] = agg
        mat = pivot_wide(agg, anchor)
        out = U.RESULTS / f"labs_for_phewas_{label}.parquet"
        mat.to_parquet(out, index=False)
        n_all, n_kept = agg_full["loinc_code"].nunique(), agg["loinc_code"].nunique()
        print(f"[{label}] {len(mat):,} MRNs x {n_kept:,} analytes kept "
              f"({n_all - n_kept} dropped below threshold) -> {out.name}")

    build_dictionary(con, aggs)
    U.log_milestone(
        f"Step 5 matrices: "
        + "; ".join(f"{l}={aggs[l]['id'].nunique():,} MRNs/{aggs[l]['loinc_code'].nunique():,} analytes"
                    for l, _ in WINDOWS)
    )


if __name__ == "__main__":
    main()
