# %% [markdown]
# Step 1 - Build the OSA exposure cohort + demographic covariates.
# Source of exposure: D:\phewas_data\psg-hst-data.xlsx (MRN, AHI, STUDY_DATE).
# Severity = AASM AHI cutoffs. Demographics + test-patient flags from member.csv.
# Writes results/icd_analysis_results/cohort/cohort_base.parquet (BMI + n_encounters
# are added by later steps). Prints DE-IDENTIFIED summaries only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datetime as dt

import pandas as pd
from openpyxl import load_workbook

import utils as U

PSG = U.DATA / "psg-hst-data.xlsx"
MEMBER = U.DATA / "member.csv"
OUT = U.COHORT_DIR / "cohort_base.parquet"
SUMMARY = U.COHORT_DIR / "cohort_summary.md"

MIN_STUDY_YEAR = 2010  # drop implausible pre-EHR study dates (a handful of 1970/2000/2004 rows)


# %%
def read_psg() -> pd.DataFrame:
    """Stream only MRN / AHI / STUDY_DATE from the sleep-study workbook."""
    wb = load_workbook(PSG, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rit = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rit)]
    idx = {c: header.index(c) for c in ["MRN", "AHI", "STUDY_DATE"]}
    mrn, ahi, sdate = [], [], []
    for r in rit:
        if r is None or len(r) <= max(idx.values()):
            continue
        mrn.append(r[idx["MRN"]])
        ahi.append(r[idx["AHI"]])
        sdate.append(r[idx["STUDY_DATE"]])
    wb.close()
    df = pd.DataFrame({"MRN": mrn, "AHI": ahi, "STUDY_DATE": sdate})
    df["id"] = U.normalize_mrn(df["MRN"])
    df["AHI"] = pd.to_numeric(df["AHI"], errors="coerce")
    # STUDY_DATE cells are SAS-datetime strings ("15MAR2018:00:00:00"); the helper
    # parses that fast path and falls back to inference for any real datetime cells.
    df["STUDY_DATE"] = U.parse_sas_datetime(df["STUDY_DATE"])
    return df.drop(columns="MRN")


# %%
def main() -> None:
    print("Reading sleep-study exposure file...")
    psg = read_psg()
    n_studies = len(psg)

    # keep usable studies: valid id, AHI present, plausible study date
    ok = (
        psg["id"].notna()
        & psg["AHI"].notna()
        & psg["STUDY_DATE"].notna()
        & (psg["STUDY_DATE"].dt.year >= MIN_STUDY_YEAR)
        & (psg["STUDY_DATE"] <= pd.Timestamp(dt.date.today()))
    )
    dropped_studies = int((~ok).sum())
    psg = psg[ok].copy()

    # index study per patient = EARLIEST plausible study (judgment call; see plan)
    psg.sort_values(["id", "STUDY_DATE"], inplace=True)
    n_multi = int((psg.groupby("id").size() > 1).sum())
    cohort = psg.groupby("id", as_index=False).first()  # earliest study + its AHI

    # AASM severity
    cohort["severity"], cohort["severity_ord"] = U.ahi_to_severity(cohort["AHI"])
    cohort["study_year"] = cohort["STUDY_DATE"].dt.year

    # ---- demographics + QA flags from member.csv ---------------------------
    print("Reading member demographics + QA flags...")
    mcols = ["MRN", "DOB", "GENDER", "RACE_ETH", "TEST_PATIENT", "BAD_MRN_FLAG", "DATASOURCE"]
    member = pd.read_csv(MEMBER, usecols=mcols, dtype=str)
    member["id"] = U.normalize_mrn(member["MRN"])
    n_member_dups = int(member["id"].duplicated().sum())
    member = member.drop_duplicates(subset="id", keep="first")  # 3 dup MRNs (DATASOURCE precedence ~ first)

    # test/invalid exclusion set: any 'Y', or blank on BOTH flags
    tp = member["TEST_PATIENT"].fillna("").str.strip().str.upper()
    bm = member["BAD_MRN_FLAG"].fillna("").str.strip().str.upper()
    excl = (tp == "Y") | (bm == "Y") | ((tp == "") & (bm == ""))
    exclude_ids = set(member.loc[excl, "id"])

    n_before = len(cohort)
    in_member = cohort["id"].isin(set(member["id"]))
    n_excluded = int(cohort["id"].isin(exclude_ids).sum())
    cohort = cohort[~cohort["id"].isin(exclude_ids)].copy()
    n_not_in_member = int((~cohort["id"].isin(set(member["id"]))).sum())

    # join demographics
    demo = member.loc[~member["id"].isin(exclude_ids),
                      ["id", "DOB", "GENDER", "RACE_ETH"]]
    cohort = cohort.merge(demo, on="id", how="left")

    # ---- covariates --------------------------------------------------------
    dob = U.parse_sas_datetime(cohort["DOB"])
    cohort["age"] = (cohort["STUDY_DATE"] - dob).dt.days / 365.25
    cohort.loc[(cohort["age"] < 0) | (cohort["age"] > 110), "age"] = pd.NA  # sanity

    # ADULTS ONLY (age >= 18): adult AASM AHI cutoffs apply only to adults; this also
    # drops rows with unknown age (NaN >= 18 is False).
    n_pre_adult = len(cohort)
    cohort = cohort[cohort["age"] >= 18].copy()
    n_dropped_nonadult = n_pre_adult - len(cohort)

    cohort["sex_M"] = (cohort["GENDER"].str.strip().str.upper() == "M").astype("Int64")
    cohort.loc[cohort["GENDER"].isna(), "sex_M"] = pd.NA
    cohort["RACE_ETH"] = cohort["RACE_ETH"].fillna("UNKNOWN").str.strip()

    keep = ["id", "AHI", "STUDY_DATE", "study_year", "severity", "severity_ord",
            "age", "sex_M", "GENDER", "RACE_ETH"]
    cohort = cohort[keep]
    U.COHORT_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(OUT, index=False)

    # ---- DE-IDENTIFIED summary --------------------------------------------
    lines = []
    def p(msg=""):
        print(msg); lines.append(str(msg))

    p("# Cohort build summary (Step 1)\n")
    p(f"- sleep-study rows read: {n_studies:,}")
    p(f"- dropped studies (bad id / missing AHI / implausible date): {dropped_studies:,}")
    p(f"- patients with >1 study (earliest indexed): {n_multi:,}")
    p(f"- member.csv duplicate MRNs dropped: {n_member_dups}")
    p(f"- cohort before test-patient drop: {n_before:,}")
    p(f"- excluded as test/bad/blank-flag patients: {n_excluded:,}")
    p(f"- dropped as pediatric / unknown age (<18): {n_dropped_nonadult:,}")
    p(f"- adult cohort after exclusions: {len(cohort):,}")
    p(f"- cohort MRNs NOT found in member (no demographics): {n_not_in_member:,} "
      f"({100*n_not_in_member/len(cohort):.1f}%)")
    p("\n## Severity distribution (earliest study)")
    sv = cohort["severity"].value_counts().reindex(["None", "Mild", "Moderate", "Severe"])
    for k, v in sv.items():
        p(f"  - {k:9s}: {int(v):,} ({100*v/len(cohort):.1f}%)")
    p("\n## Covariate coverage")
    p(f"  - age non-null: {cohort['age'].notna().sum():,} "
      f"({100*cohort['age'].notna().mean():.1f}%); "
      f"min/median/max = {cohort['age'].min():.1f}/{cohort['age'].median():.1f}/{cohort['age'].max():.1f}")
    sexvc = cohort["sex_M"].value_counts(dropna=False).to_dict()
    p(f"  - sex_M value counts (1=M,0=F/other): {sexvc}")
    p(f"  - GENDER raw values: {cohort['GENDER'].value_counts(dropna=False).to_dict()}")
    p(f"  - distinct RACE_ETH codes: {cohort['RACE_ETH'].nunique()}; "
      f"top: {cohort['RACE_ETH'].value_counts().head(8).to_dict()}")
    p(f"  - study_year distribution: {cohort['study_year'].value_counts().sort_index().to_dict()}")
    p(f"\nWrote cohort_base.parquet ({len(cohort):,} rows) -> {OUT}")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    U.log_milestone(f"Step 1 cohort built: {len(cohort):,} patients (pre-BMI/encounters).")


if __name__ == "__main__":
    main()
