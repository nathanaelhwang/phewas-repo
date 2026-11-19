# %% [markdown]
# Step 02 - Per-patient follow-up frame + the go/no-go availability audit.
#
# Joins the v2 cohort spine (index date + all m1-m4 covariates) to insurance coverage_end
# (step 00) and member death (DOD/DECEASED), then computes the administrative censoring
# boundary that is common to EVERY phecode:
#
#   censor_date    = min( 2023-05-31 , coverage_end_binding , death_date )
#   followup_years = (censor_date - index) / 365.25
#   is_death       = death terminates follow-up (competing event)
#
# Patients with followup_years <= 0 (indexed on/after their censor boundary -- notably the
# ~65k indexed after the 2023-05-31 diagnosis cutoff) contribute no observable follow-up and
# are dropped from the incidence spine. This frame is phecode-INDEPENDENT: the per-phecode
# at-risk set and event flag are derived downstream (step 03) from the event store.
#
# Also writes availability_audit.md -- patients & person-time with >=1/2/3yr potential
# follow-up, by severity and index year -- the honest answer to "how much signal exists".
# DE-IDENTIFIED aggregates only; the frame itself lands on local disk.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import incwas_utils as U

CUTOFF = pd.Timestamp(U.ADATE_CUTOFF)
COHORT_COLS = ["id", "AHI", "STUDY_DATE", "study_year", "severity", "severity_ord",
               "age", "sex_M", "bmi", "n_encounters", "obs_window", "enc_per_year",
               "RACE_ETH", "smoking_status", "income_gp"]


def load_death() -> pd.DataFrame:
    """member.csv -> per-id death_date (DOD where DECEASED==1), MRN deduped."""
    m = pd.read_csv(U.MEMBER, usecols=["MRN", "DOD", "DECEASED"], dtype=str)
    m["id"] = U.normalize_mrn(m["MRN"])
    m["deceased"] = pd.to_numeric(m["DECEASED"], errors="coerce").fillna(0).astype(int)
    m["dod"] = U.parse_sas_datetime(m["DOD"])
    m.loc[m["deceased"] != 1, "dod"] = pd.NaT
    g = m.dropna(subset=["id"]).groupby("id", as_index=False).agg(
        death_date=("dod", "min"), deceased=("deceased", "max"))
    return g[["id", "death_date"]]


def main() -> None:
    U.ensure_dirs()

    coh = pd.read_parquet(U.COHORT, columns=COHORT_COLS)
    coh["idx"] = pd.to_datetime(coh["STUDY_DATE"]).dt.normalize()
    n0 = len(coh)

    cov = pd.read_parquet(U.INS_COVERAGE, columns=["id", "coverage_end_binding"])
    cov["coverage_end_binding"] = pd.to_datetime(cov["coverage_end_binding"]).astype("datetime64[ns]")
    death = load_death()

    df = coh.merge(cov, on="id", how="left").merge(death, on="id", how="left")

    # censor_date = earliest of {cutoff, binding disenrollment, death}; NaT boundaries ignored.
    comp = pd.DataFrame({
        "cutoff": CUTOFF,
        "coverage": df["coverage_end_binding"],
        "death": df["death_date"],
    }, index=df.index)
    df["censor_date"] = comp.min(axis=1)
    df["is_death"] = df["death_date"].notna() & (df["death_date"] <= df["censor_date"])
    df["followup_years"] = (df["censor_date"] - df["idx"]).dt.days / 365.25

    # drop patients with no observable follow-up (indexed on/after their censor boundary)
    keep = df["followup_years"] > 0
    n_drop = int((~keep).sum())
    n_post_cut = int((df["idx"] > CUTOFF).sum())
    df = df[keep].copy()

    out_cols = ["id", "idx", "censor_date", "followup_years", "is_death",
                "AHI", "severity", "severity_ord", "study_year",
                "age", "sex_M", "bmi", "n_encounters", "obs_window", "enc_per_year",
                "RACE_ETH", "smoking_status", "income_gp"]
    df[out_cols].to_parquet(U.FOLLOWUP_FRAME, index=False)

    # ---- availability audit -------------------------------------------------
    fy = df["followup_years"].to_numpy()
    def pt_capped(years):  # person-time capped at horizon (matches the incidence-bucket denominator ceiling)
        return float(np.minimum(fy, years).sum())

    lines = []
    lines.append("# incwas — follow-up availability audit\n")
    lines.append(f"- Diagnosis outcome cutoff: **{U.ADATE_CUTOFF}** (administrative censoring).")
    lines.append(f"- Cohort spine: {n0:,} patients; dropped {n_drop:,} with no observable follow-up "
                 f"(of which {n_post_cut:,} indexed after the cutoff).")
    lines.append(f"- **Incidence spine: {len(df):,} patients** with >0 follow-up.\n")
    lines.append(f"- Median follow-up: {np.median(fy):.2f} yr; total person-time: {fy.sum():,.0f} yr.")
    lines.append(f"- Deaths during follow-up (competing events): {int(df['is_death'].sum()):,} "
                 f"({100*df['is_death'].mean():.1f}%).\n")

    # by severity: N with >=1/2/3yr potential follow-up + capped person-time
    lines.append("## Patients with >=k-year potential follow-up, by severity\n")
    lines.append("| severity | N (spine) | >=1yr | >=2yr | >=3yr | PT<=1yr | PT<=2yr | PT<=3yr |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for sev in U.SEVERITIES:
        s = df[df["severity"] == sev]
        sfy = s["followup_years"].to_numpy()
        lines.append(f"| {sev} | {len(s):,} | {int((sfy>=1).sum()):,} | {int((sfy>=2).sum()):,} | "
                     f"{int((sfy>=3).sum()):,} | {np.minimum(sfy,1).sum():,.0f} | "
                     f"{np.minimum(sfy,2).sum():,.0f} | {np.minimum(sfy,3).sum():,.0f} |")
    lines.append(f"| **All** | **{len(df):,}** | **{int((fy>=1).sum()):,}** | **{int((fy>=2).sum()):,}** "
                 f"| **{int((fy>=3).sum()):,}** | **{pt_capped(1):,.0f}** | **{pt_capped(2):,.0f}** "
                 f"| **{pt_capped(3):,.0f}** |\n")

    # by index year: how follow-up shrinks toward the cutoff
    lines.append("## Median follow-up & N by index year\n")
    lines.append("| study_year | N | median FU (yr) | N >=1yr | N >=3yr |")
    lines.append("|---|--:|--:|--:|--:|")
    for yr, g in df.groupby("study_year"):
        gfy = g["followup_years"].to_numpy()
        lines.append(f"| {yr} | {len(g):,} | {np.median(gfy):.2f} | {int((gfy>=1).sum()):,} | "
                     f"{int((gfy>=3).sum()):,} |")

    audit = U.FOLLOWUP / "availability_audit.md"
    audit.write_text("\n".join(lines), encoding="utf-8")

    print(f"incidence spine: {len(df):,}/{n0:,} patients (dropped {n_drop:,}; {n_post_cut:,} post-cutoff)")
    print(f"median follow-up {np.median(fy):.2f} yr; person-time {fy.sum():,.0f} yr; "
          f"deaths {int(df['is_death'].sum()):,}")
    print(f">=1yr {int((fy>=1).sum()):,} | >=2yr {int((fy>=2).sum()):,} | >=3yr {int((fy>=3).sum()):,}")
    print(f"wrote {U.FOLLOWUP_FRAME}")
    print(f"wrote {audit}")

    U.log_milestone(
        f"Step 02 follow-up frame: incidence spine {len(df):,}/{n0:,} patients (>0 follow-up); "
        f"median FU {np.median(fy):.2f}yr, {int(df['is_death'].sum()):,} deaths; "
        f">=1/2/3yr = {int((fy>=1).sum()):,}/{int((fy>=2).sum()):,}/{int((fy>=3).sum()):,}. "
        f"Audit -> availability_audit.md.")


if __name__ == "__main__":
    main()
