# %% [markdown]
# Step 03 - Assemble the unified Excel workbook from the step-02 tidy tables.
#
# results/utilwas_results/analysis/utilwas_preliminary.xlsx sheets:
#   README            - cohort, windows, definitions, the messages correction, denominators
#   tidy_all          - the long master counts table
#   <analysis>        - one wide sheet per analysis: rows=subcategory,
#                       cols = window (1yr,5yr) x severity (Overall,None,Mild,Moderate,Severe)
#                              x metric (n_encounters, per_pt_per_yr, pct_pat_ge1)
#   outpatient_decomp - phys_only / allied_only / overlap / phys_any / allied_any / total
#   los_stats         - inpatient (los>0) and SNF LOS mean/median by window x severity
#   zero_inflation    - Total-level cells with pct_pat_ge1 < 0.40 (two-part model candidates)

# %%
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import utilwas_utils as U

WINDOWS = ["1yr", "5yr"]
SEVS = ["Overall"] + U.SEVERITIES
METRICS = ["n_encounters", "per_pt_per_yr", "per_user_per_yr", "pct_pat_ge1"]
ANALYSIS_ORDER = ["physician_level", "allied_health", "other", "messages_phone",
                  "ED", "inpatient", "urgent_care"]
OUT_XLSX = U.ANALYSIS / "utilwas_preliminary.xlsx"


def wide_sheet(tidy: pd.DataFrame, analysis: str) -> pd.DataFrame:
    sub = tidy[tidy["analysis"] == analysis]
    long = sub.melt(id_vars=["subcategory", "window", "severity"], value_vars=METRICS,
                    var_name="metric", value_name="val")
    wide = long.pivot_table(index="subcategory", columns=["window", "severity", "metric"],
                            values="val", aggfunc="first")
    col_order = [(w, s, m) for w in WINDOWS for s in SEVS for m in METRICS]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(col_order, names=["window", "severity", "metric"]))
    subs = list(wide.index)
    ordered = (["Total"] if "Total" in subs else []) + sorted(x for x in subs if x != "Total")
    wide = wide.reindex(ordered)
    for col in wide.columns:
        dec = {"n_encounters": 0, "per_pt_per_yr": 3, "per_user_per_yr": 3, "pct_pat_ge1": 4}[col[2]]
        wide[col] = wide[col].round(dec)
    return wide


def readme_frame(N: dict) -> pd.DataFrame:
    rows = [
        ("stage", "UtilWAS preliminary utilization descriptives"),
        ("generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("source", "D:/phewas_data/utilization.csv (~104 GB), windowed to cohort x 5yr pre-index"),
        ("cohort", "v2 ICD association cohort (results/icd_analysis_v2_results/cohort/cohort.parquet)"),
        ("cohort N (Overall)", N.get("Overall")),
        ("cohort N (None/Mild/Moderate/Severe)",
         f"{N.get('None')} / {N.get('Mild')} / {N.get('Moderate')} / {N.get('Severe')}"),
        ("windows", "[index - N yr, index); lower inclusive, upper exclusive; N in {1,5}"),
        ("encounter", "COUNT(DISTINCT UTIL_ID) after row filters (UTIL_ID = encounter key)"),
        ("per_pt_per_yr", "population rate: n_encounters / N_cohort[severity] / window_years (0-util pts in denom)"),
        ("per_user_per_yr", "intensity among users: n_encounters / n_pat_ge1 / window_years "
                            "(= per_pt_per_yr / pct_pat_ge1; the rate with zeros removed -- 'part 2' of a hurdle model)"),
        ("pct_pat_ge1", "fraction of cohort (in that stratum) with >=1 such encounter (zero-inflation signal)"),
        ("physician_level", "outpatient (CLINIC/VIRTUAL; OFFICE VISIT/TAV/VIDEO) encounters with >=1 ICD9/ICD10 dx row"),
        ("allied_health", "same outpatient set, encounters with >=1 non-ICD dx row (NULL code counts as non-ICD)"),
        ("phys/allied overlap", "encounters with both ICD and non-ICD rows are exported for investigation; "
                                "see outpatient_decomp for phys_only/allied_only/overlap"),
        ("MESSAGES CORRECTION", "PATIENT MESSAGE/PHONE live under CARE_TYPE='VIRTUAL', not 'OTHER'; "
                                "messages_phone filters on CARE_SUBTYPE only"),
        ("ED / inpatient", "ED-* subtypes appear in BOTH ED and inpatient by design (per spec)"),
        ("LOS (inpatient)", "mean/median LOS per subtype, excluding LOS=0 and NULL"),
        ("LOS (SNF, in 'other')", "mean/median LOS over SNF encounters with a recorded (non-null) LOS"),
        ("urgent_care", "DEPT_SPECIALTY = 'Urgent Care' (excludes 'Pediatric Urgent Care')"),
    ]
    return pd.DataFrame(rows, columns=["field", "detail"])


def read_tidy(path) -> pd.DataFrame:
    # keep_default_na=False so the OSA severity stratum 'None' is NOT parsed as NaN.
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def main() -> None:
    T = U.TABLES
    tidy = read_tidy(T / "counts_tidy.csv")
    decomp = read_tidy(T / "outpatient_decomposition.csv")
    los = read_tidy(T / "los_tidy.csv")
    zi_path = T / "zero_inflation.csv"
    zi = read_tidy(zi_path) if zi_path.exists() and zi_path.stat().st_size else pd.DataFrame()

    N = tidy[tidy["severity"] != "Overall"].drop_duplicates("severity").set_index("severity")["n_cohort"].to_dict()
    N["Overall"] = int(tidy[tidy["severity"] == "Overall"]["n_cohort"].iloc[0])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xl:
        readme_frame(N).to_excel(xl, sheet_name="README", index=False)
        tidy.to_excel(xl, sheet_name="tidy_all", index=False)
        for a in ANALYSIS_ORDER:
            wide_sheet(tidy, a).to_excel(xl, sheet_name=a[:31])
        decomp.to_excel(xl, sheet_name="outpatient_decomp", index=False)
        los.to_excel(xl, sheet_name="los_stats", index=False)
        if len(zi):
            zi.to_excel(xl, sheet_name="zero_inflation", index=False)
        if (T / "significance.csv").exists():
            read_tidy(T / "significance.csv").to_excel(xl, sheet_name="significance", index=False)
            read_tidy(T / "severity_descriptives.csv").to_excel(xl, sheet_name="severity_descr", index=False)
        for cat in ANALYSIS_ORDER:
            p = T / f"dx_{cat}.csv"
            if p.exists():
                read_tidy(p).to_excel(xl, sheet_name=f"dx_{cat}"[:31], index=False)
        if (T / "dx_coverage.csv").exists():
            read_tidy(T / "dx_coverage.csv").to_excel(xl, sheet_name="dx_coverage", index=False)

    print(f"wrote {OUT_XLSX}")
    print(f"  sheets: README, tidy_all, {len(ANALYSIS_ORDER)} per-analysis, "
          f"outpatient_decomp, los_stats{', zero_inflation' if len(zi) else ''}")
    U.log_milestone(f"Step 03 workbook: {OUT_XLSX.name} "
                    f"(README + tidy_all + 7 analyses + decomp + LOS + zero-inflation).")


if __name__ == "__main__":
    main()
