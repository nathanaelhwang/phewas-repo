# %% [markdown]
# v2 Step 1 - Build the v2 cohort from EXISTING v1 artifacts (no 104 GB rescan).
# New exclusions on top of the v1 complete-case cohort (302,929):
#   - prevalent OSA dx pre-index: any ICD-10 G47.3x or ICD-9 327.2x with ADATE <= STUDY_DATE
#   - require >=1 ICD-coded encounter >=1 yr before index (obs_window >= 1)
#   - require >=2 ICD-coded encounters (pre-index distinct ENC_SID)
# New covariates joined: SMOKING_STATUS (smoking.csv), H_MED_INCOME (geocode.csv, impute+indicator).
# observation-window length = age - age@first-pre-index-ICD (derived from icds_all.csv).
# Writes results/icd_analysis_v2_results/cohort/cohort.parquet + summary. DE-IDENTIFIED only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import duckdb

import utils as U

V1_COHORT = (U.COHORT_DIR / "cohort.parquet").as_posix()
ICDS_ALL = (U.PHENO_DIR / "icds_all.csv").as_posix()
SMOKING = "D:/phewas_data/smoking.csv"
GEOCODE = "D:/phewas_data/geocode.csv"
OUT = U.V2_COHORT / "cohort.parquet"
SUMMARY = U.V2_COHORT / "cohort_summary.md"

NORM = "lpad(regexp_replace(CAST(MRN AS VARCHAR), '\\.0$', ''), 12, '0')"


def main() -> None:
    U.V2_COHORT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    # per-id phenome features: earliest ICD age + prevalent-OSA flag
    icd = con.sql(f"""
        SELECT id,
               min(AgeAtICD) AS min_age_icd,
               max(CASE WHEN (ICD_TYPE=10 AND ICD_CODE LIKE 'G47.3%')
                          OR (ICD_TYPE=9  AND ICD_CODE LIKE '327.2%') THEN 1 ELSE 0 END) AS has_pre_osa
        FROM read_csv('{ICDS_ALL}', header=true,
                      columns={{'id':'VARCHAR','ICD_CODE':'VARCHAR','ICD_TYPE':'INTEGER','AgeAtICD':'DOUBLE'}})
        GROUP BY id
    """).df()

    # SMOKING_STATUS per id (priority Current>Former>Never>Unknown)
    smoke = con.sql(f"""
        SELECT {NORM} AS id,
               arg_max(SMOKING_STATUS, CASE SMOKING_STATUS WHEN 'Current' THEN 3 WHEN 'Former' THEN 2
                                       WHEN 'Never' THEN 1 ELSE 0 END) AS smoking_status
        FROM read_csv('{SMOKING}', header=true, quote='"', escape='"', all_varchar=true, ignore_errors=true)
        GROUP BY id
    """).df()

    # income GROUP per id (pre-binned categorical; Missing kept as its own level -> NO imputation)
    income = con.sql(f"""
        SELECT {NORM} AS id,
               arg_max(h_med_income_gp,
                       CASE WHEN h_med_income_gp IS NULL OR h_med_income_gp IN ('None','') THEN 0 ELSE 1 END) AS income_gp_raw
        FROM read_csv('{GEOCODE}', header=true, quote='"', escape='"', all_varchar=true, ignore_errors=true)
        GROUP BY id
    """).df()

    c = pd.read_parquet(V1_COHORT)
    n_start = len(c)
    c = c.merge(icd, on="id", how="left").merge(smoke, on="id", how="left").merge(income, on="id", how="left")

    c["obs_window"] = c["age"] - c["min_age_icd"]            # years from first pre-index ICD to index
    c["enc_per_year"] = c["n_encounters"] / c["obs_window"].clip(lower=1)  # reported, not a model covariate

    # ---- exclusions ----
    excl_osa = c["has_pre_osa"].fillna(0).astype(int) == 1
    excl_window = ~(c["obs_window"] >= 1)                    # need an encounter >=1 yr pre-index
    excl_enc = ~(c["n_encounters"] >= 2)                    # need >=2 ICD encounters
    keep = ~excl_osa & ~excl_window & ~excl_enc
    final = c[keep].copy()

    # ---- covariates: smoking (Unknown level) + income group (Missing as its own level) ----
    final["smoking_status"] = final["smoking_status"].fillna("Unknown")

    def map_income(s):  # robust to formatting of the gp strings
        s = str(s)
        if "80,001" in s:
            return "High"
        if "45,001" in s:
            return "Mid"
        if "45,000" in s:
            return "Low"
        return "Missing"
    final["income_gp"] = final["income_gp_raw"].map(map_income)

    keep_cols = ["id", "AHI", "STUDY_DATE", "study_year", "severity", "severity_ord",
                 "age", "sex_M", "bmi", "n_encounters", "obs_window", "enc_per_year",
                 "RACE_ETH", "smoking_status", "income_gp"]
    final = final[keep_cols]
    final.to_parquet(OUT, index=False)

    # ---- DE-IDENTIFIED summary ----
    L = []
    def p(m=""):
        print(m); L.append(str(m))
    p("# v2 cohort build summary\n")
    p(f"- start (v1 complete-case adults): {n_start:,}")
    p(f"- excluded prevalent OSA dx pre-index (G47.3x / 327.2x): {int(excl_osa.sum()):,}")
    p(f"- excluded <1 yr lookback (no ICD encounter >=1 yr pre-index): {int(excl_window.sum()):,}")
    p(f"- excluded <2 ICD encounters: {int(excl_enc.sum()):,}")
    p(f"- **final v2 cohort: {len(final):,}** ({100*len(final)/n_start:.1f}% of v1)")
    p("\n## Severity (final)")
    for k, v in final["severity"].value_counts().reindex(["None","Mild","Moderate","Severe"]).items():
        p(f"  - {k:9s}: {int(v):,} ({100*v/len(final):.1f}%)")
    p("\n## Covariates")
    p(f"  - age median {final['age'].median():.1f}; sex_M {100*final['sex_M'].mean():.1f}%; BMI median {final['bmi'].median():.1f}")
    p(f"  - obs_window (yr) median {final['obs_window'].median():.1f} (IQR {final['obs_window'].quantile(.25):.1f}-{final['obs_window'].quantile(.75):.1f})")
    p(f"  - smoking: {final['smoking_status'].value_counts().to_dict()}")
    p(f"  - income group (no imputation): {final['income_gp'].value_counts().to_dict()}")
    p(f"  - race: {final['RACE_ETH'].value_counts().to_dict()}")
    p(f"  - AHI>=5 {100*(final['AHI']>=5).mean():.1f}%; AHI>=15 {100*(final['AHI']>=15).mean():.1f}%")
    p(f"\nWrote {OUT}")
    SUMMARY.write_text("\n".join(L), encoding="utf-8")
    U.log_milestone(f"v2 Step 1 cohort: {len(final):,} (from {n_start:,}); excl OSA/window/enc.")


if __name__ == "__main__":
    main()
