# %% [markdown]
# Step 02 - Encounter counts, rates, prevalence, and LOS for the seven utilization
# categories, off the windowed store built in step 01.
#
# Encounters are COUNT(DISTINCT UTIL_ID). We first collapse the diagnosis-line store to one
# row per encounter (attributes are constant within UTIL_ID; CODE_SUBTYPE is rolled up to
# has_icd / has_nonicd flags, DEPT_SPECIALTY to an is_urgent flag). Every metric is reported
# for both windows (1yr, 5yr) and both Overall and by OSA severity (None/Mild/Moderate/Severe).
#
# Metrics per (analysis, window, severity, subcategory):
#   n_encounters  = COUNT(DISTINCT UTIL_ID)
#   n_pat_ge1     = # cohort patients with >=1 such encounter
#   pct_pat_ge1    = n_pat_ge1 / N_cohort[severity]        (zero-inflation signal)
#   per_pt_per_yr  = n_encounters / N_cohort[severity] / window_years   (population rate)
#   per_user_per_yr= n_encounters / n_pat_ge1 / window_years           (intensity among users
#                    = per_pt_per_yr / pct_pat_ge1; the population rate with zeros removed)
# Population-rate denominator = ALL cohort patients in the stratum (0-utilization included).
#
# Emits (all de-identified) to results/utilwas_results/analysis/tables/:
#   counts_tidy.csv, outpatient_decomposition.csv, los_tidy.csv, zero_inflation.csv
# and the patient-level overlap exports to results/utilwas_results/overlap/ (local disk only).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import utilwas_utils as U

ICD = U.sql_in(U.ICD_CODE_SUBTYPES)
OUT_CT = U.sql_in(U.OUTPATIENT_CARE_TYPES)
OUT_ST = U.sql_in(U.OUTPATIENT_SUBTYPES)
NONICD = f"COALESCE(CODE_SUBTYPE,'~NULL~') NOT IN ({ICD})"          # store-level (raw col)
ZERO_INFLATION_THRESHOLD = 0.40

# analysis -> (WHERE over the encounter table `enc`, subcategory expr or None)
OUTPATIENT_WHERE = f"care_type IN ({OUT_CT}) AND care_subtype IN ({OUT_ST})"
ANALYSES = [
    ("physician_level", f"{OUTPATIENT_WHERE} AND has_icd = 1", "care_subtype"),
    ("allied_health",   f"{OUTPATIENT_WHERE} AND has_nonicd = 1", "care_subtype"),
    ("other",           f"care_type = '{U.OTHER_CARE_TYPE}'", "care_subtype"),
    ("messages_phone",  f"care_subtype IN ({U.sql_in(U.MESSAGES_SUBTYPES)})", "care_subtype"),
    ("ED",              f"care_subtype IN ({U.sql_in(U.ED_SUBTYPES)})", "care_subtype"),
    ("inpatient",       f"care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)})", "care_subtype"),
    ("urgent_care",     "is_urgent = 1", None),
]
SEV_ORDER = ["Overall"] + U.SEVERITIES


def build_enc(con) -> None:
    """Collapse the diagnosis-line store to one row per (id, UTIL_ID) encounter."""
    con.execute(f"""
        CREATE TEMP TABLE enc AS
        SELECT id, UTIL_ID,
               any_value(severity)     AS severity,
               bool_or(in_1yr)         AS in_1yr,
               any_value(CARE_TYPE)    AS care_type,
               any_value(CARE_SUBTYPE) AS care_subtype,
               max(los_days)           AS los_days,
               max(CASE WHEN CODE_SUBTYPE IN ({ICD}) THEN 1 ELSE 0 END)   AS has_icd,
               max(CASE WHEN {NONICD} THEN 1 ELSE 0 END)                  AS has_nonicd,
               max(CASE WHEN DEPT_SPECIALTY = '{U.URGENT_CARE_SPECIALTY}' THEN 1 ELSE 0 END) AS is_urgent
        FROM read_parquet('{U.STORE_5YR}')
        GROUP BY id, UTIL_ID
    """)


def agg_counts(con, name, where, subcat_expr, window_label, window_flag) -> pd.DataFrame:
    """Encounter/patient counts with GROUPING SETS -> Overall+severity x Total+subcategory."""
    if subcat_expr is None:
        q = f"""
            SELECT CASE WHEN GROUPING(severity)=1 THEN 'Overall' ELSE severity END AS severity,
                   'Total' AS subcategory,
                   count(*) AS n_encounters, count(DISTINCT id) AS n_pat_ge1
            FROM enc WHERE {window_flag} AND ({where})
            GROUP BY GROUPING SETS ((severity), ())
        """
    else:
        q = f"""
            SELECT CASE WHEN GROUPING(severity)=1 THEN 'Overall' ELSE severity END AS severity,
                   CASE WHEN GROUPING(subcat)=1 THEN 'Total'
                        ELSE COALESCE(subcat,'(null)') END AS subcategory,
                   count(*) AS n_encounters, count(DISTINCT id) AS n_pat_ge1
            FROM (SELECT id, severity, {subcat_expr} AS subcat
                  FROM enc WHERE {window_flag} AND ({where})) t
            GROUP BY GROUPING SETS ((severity, subcat), (subcat), (severity), ())
        """
    df = con.sql(q).df()
    df.insert(0, "analysis", name)
    df.insert(1, "window", window_label)
    return df


def outpatient_decomposition(con, window_label, window_flag) -> pd.DataFrame:
    q = f"""
        SELECT CASE WHEN GROUPING(severity)=1 THEN 'Overall' ELSE severity END AS severity,
               count(*) AS total,
               sum(has_icd) AS phys_any,
               sum(has_nonicd) AS allied_any,
               sum(CASE WHEN has_icd=1 AND has_nonicd=1 THEN 1 ELSE 0 END) AS overlap
        FROM enc WHERE {window_flag} AND ({OUTPATIENT_WHERE})
        GROUP BY GROUPING SETS ((severity), ())
    """
    d = con.sql(q).df()
    d["phys_only"] = d["phys_any"] - d["overlap"]
    d["allied_only"] = d["allied_any"] - d["overlap"]
    d.insert(0, "window", window_label)
    return d


def los_stats(con, window_label, window_flag) -> pd.DataFrame:
    # inpatient: mean/median LOS per subtype (+Total), excluding los=0 and null
    inp = con.sql(f"""
        SELECT CASE WHEN GROUPING(severity)=1 THEN 'Overall' ELSE severity END AS severity,
               CASE WHEN GROUPING(subcat)=1 THEN 'Total' ELSE subcat END AS subcategory,
               count(*) AS n_enc_los, avg(los_days) AS mean_los, median(los_days) AS median_los
        FROM (SELECT severity, care_subtype AS subcat, los_days FROM enc
              WHERE {window_flag} AND care_subtype IN ({U.sql_in(U.INPATIENT_SUBTYPES)})
                AND los_days IS NOT NULL AND los_days > 0) t
        GROUP BY GROUPING SETS ((severity, subcat), (subcat), (severity), ())
    """).df()
    inp.insert(0, "analysis", "inpatient")
    inp["rule"] = "los>0"
    # SNF (other): mean/median LOS over SNF encounters with a recorded LOS (non-null)
    snf = con.sql(f"""
        SELECT CASE WHEN GROUPING(severity)=1 THEN 'Overall' ELSE severity END AS severity,
               'SNF' AS subcategory,
               count(*) AS n_enc_los, avg(los_days) AS mean_los, median(los_days) AS median_los
        FROM (SELECT severity, los_days FROM enc
              WHERE {window_flag} AND care_subtype = '{U.SNF_SUBTYPE}'
                AND los_days IS NOT NULL) t
        GROUP BY GROUPING SETS ((severity), ())
    """).df()
    snf.insert(0, "analysis", "other_snf")
    snf["rule"] = "los non-null"
    out = pd.concat([inp, snf], ignore_index=True)
    out.insert(1, "window", window_label)
    return out


def export_overlap(con) -> None:
    """Patient-level export of outpatient encounters with BOTH ICD and non-ICD dx rows."""
    for label, extra in [("1yr", "in_1yr AND "), ("5yr", "")]:
        out = (U.OVERLAP_DIR / f"outpatient_phys_allied_overlap_{label}.csv").as_posix()
        con.execute(f"""
            COPY (
              SELECT id, UTIL_ID, any_value(enc_date) AS enc_date,
                     any_value(CARE_SUBTYPE) AS care_subtype,
                     sum(CASE WHEN CODE_SUBTYPE IN ({ICD}) THEN 1 ELSE 0 END) AS n_icd_rows,
                     sum(CASE WHEN {NONICD} THEN 1 ELSE 0 END) AS n_nonicd_rows,
                     array_to_string(list_distinct(array_agg(COALESCE(CODE_SUBTYPE,'NULL'))), '|') AS code_subtypes
              FROM read_parquet('{U.STORE_5YR}')
              WHERE {extra} CARE_TYPE IN ({OUT_CT}) AND CARE_SUBTYPE IN ({OUT_ST})
              GROUP BY id, UTIL_ID
              HAVING sum(CASE WHEN CODE_SUBTYPE IN ({ICD}) THEN 1 ELSE 0 END) > 0
                 AND sum(CASE WHEN {NONICD} THEN 1 ELSE 0 END) > 0
            ) TO '{out}' (FORMAT CSV, HEADER)
        """)
        n = con.sql(f"SELECT count(*) FROM read_csv_auto('{out}')").fetchone()[0]
        print(f"  overlap export [{label}]: {n:,} encounters -> {out}")


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    # cohort denominators (Overall + by severity)
    den = con.sql(f"SELECT severity, count(*) n FROM read_parquet('{U.COHORT}') GROUP BY 1").df()
    N = {r.severity: int(r.n) for r in den.itertuples()}
    N["Overall"] = int(sum(N.values()))
    print("cohort denominators:", N)

    build_enc(con)
    n_enc = con.sql("SELECT count(*) FROM enc").fetchone()[0]
    print(f"encounter-level rows (distinct UTIL_ID in 5yr): {n_enc:,}")

    counts, decomp, los = [], [], []
    for label, years in U.WINDOWS:
        flag = "in_1yr" if label == "1yr" else "TRUE"
        for name, where, subcat in ANALYSES:
            counts.append(agg_counts(con, name, where, subcat, label, flag))
        decomp.append(outpatient_decomposition(con, label, flag))
        los.append(los_stats(con, label, flag))

    tidy = pd.concat(counts, ignore_index=True)
    tidy["n_cohort"] = tidy["severity"].map(N)
    yrs = {"1yr": 1, "5yr": 5}
    tidy["pct_pat_ge1"] = tidy["n_pat_ge1"] / tidy["n_cohort"]
    tidy["per_pt_per_yr"] = tidy["n_encounters"] / tidy["n_cohort"] / tidy["window"].map(yrs)
    # intensive margin: encounters/yr among patients with >=1 (denominator = users only).
    # equals per_pt_per_yr / pct_pat_ge1 -- the population rate with the zeros removed.
    tidy["per_user_per_yr"] = tidy["n_encounters"] / tidy["n_pat_ge1"] / tidy["window"].map(yrs)
    tidy["low_prevalence"] = tidy["pct_pat_ge1"] < ZERO_INFLATION_THRESHOLD
    # order
    an_order = {a[0]: i for i, a in enumerate(ANALYSES)}
    tidy["_a"] = tidy["analysis"].map(an_order)
    tidy["_s"] = tidy["severity"].map({s: i for i, s in enumerate(SEV_ORDER)})
    tidy["_sub"] = (tidy["subcategory"] != "Total").astype(int)
    tidy = (tidy.sort_values(["_a", "window", "_sub", "subcategory", "_s"])
                .drop(columns=["_a", "_s", "_sub"]).reset_index(drop=True))
    tidy = tidy[["analysis", "window", "severity", "subcategory", "n_encounters", "n_pat_ge1",
                 "n_cohort", "pct_pat_ge1", "per_pt_per_yr", "per_user_per_yr", "low_prevalence"]]

    decomp = pd.concat(decomp, ignore_index=True)
    los = pd.concat(los, ignore_index=True)
    los["mean_los"] = los["mean_los"].round(2)
    los["median_los"] = los["median_los"].round(2)

    U.TABLES.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(U.TABLES / "counts_tidy.csv", index=False)
    decomp.to_csv(U.TABLES / "outpatient_decomposition.csv", index=False)
    los.to_csv(U.TABLES / "los_tidy.csv", index=False)
    zi = tidy[(tidy["subcategory"] == "Total") & tidy["low_prevalence"]].sort_values("pct_pat_ge1")
    zi.to_csv(U.TABLES / "zero_inflation.csv", index=False)

    export_overlap(con)

    # ---- verification (de-identified) --------------------------------------
    print("\n--- verification ---")
    ok = True
    for name, _, _ in ANALYSES:
        for label, _ in U.WINDOWS:
            sub = tidy[(tidy.analysis == name) & (tidy.window == label) & (tidy.subcategory == "Total")]
            overall = sub[sub.severity == "Overall"]["n_encounters"].sum()
            bysev = sub[sub.severity != "Overall"]["n_encounters"].sum()
            if overall != bysev:
                ok = False
                print(f"  [FAIL] partition {name} {label}: overall={overall} sum(severity)={bysev}")
    # outpatient identity: phys_only + allied_only + overlap == total
    bad = decomp[(decomp.phys_only + decomp.allied_only + decomp.overlap) != decomp.total]
    if len(bad):
        ok = False
        print(f"  [FAIL] outpatient identity off for {len(bad)} rows")
    # monotonicity: 1yr Overall Total <= 5yr Overall Total
    piv = (tidy[(tidy.severity == "Overall") & (tidy.subcategory == "Total")]
           .pivot(index="analysis", columns="window", values="n_encounters"))
    for a, row in piv.iterrows():
        if pd.notna(row.get("1yr")) and pd.notna(row.get("5yr")) and row["1yr"] > row["5yr"]:
            ok = False
            print(f"  [FAIL] monotonicity {a}: 1yr={row['1yr']} > 5yr={row['5yr']}")
    print("  partition / identity / monotonicity:", "PASS" if ok else "FAIL")

    print("\n--- headline (Overall, Total) ---")
    head = tidy[(tidy.severity == "Overall") & (tidy.subcategory == "Total")]
    for _, r in head.iterrows():
        print(f"  {r.analysis:16s} {r.window}: {r.n_encounters:>10,} enc | "
              f"{r.per_pt_per_yr:6.2f}/pt/yr | {100*r.pct_pat_ge1:5.1f}% with >=1")

    U.log_milestone(f"Step 02 counts: 7 analyses x 2 windows x severity; {n_enc:,} encounters; "
                    f"verification {'PASS' if ok else 'FAIL'}; tables + overlap exports written.")


if __name__ == "__main__":
    main()
