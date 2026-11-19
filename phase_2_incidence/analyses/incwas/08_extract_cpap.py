# %% [markdown]
# Step 08 - Extract per-patient CPAP/PAP usage from PopulationCohort251103.csv (101 GB, exploded).
#
# The device-adherence block (mean_pap_usage, percentage_over_4h, total_days_usage, day_hit,
# adherence_flag) is patient-level but buried in the cartesian-exploded vitals file. We collapse to
# one row per MRN (max over the explosion) for cohort patients. `mean_pap_usage` units are UNCONFIRMED
# in the data dictionary ("likely minutes/night") -- this script prints its distribution so the
# min/night-vs-hr/night question is settled before binning (<2 / 2-4 / 4+ hrs/night).
#
# Emits results/incwas_results/followup/cpap_usage.parquet. DE-IDENTIFIED aggregates only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import numpy as np
import pandas as pd

import incwas_utils as U

PC = "D:/phewas_data/PopulationCohort251103.csv"
OUT = (U.FOLLOWUP / "cpap_usage.parquet").as_posix()
PC_READ = "header=true, quote='\"', escape='\"', all_varchar=true, ignore_errors=true"


def main() -> None:
    U.ensure_dirs()
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=96, threads=12)

    print("Scanning PopulationCohort for the PAP-usage block (~101 GB, one pass)...", flush=True)
    con.execute(f"""
        COPY (
          WITH raw AS (
            SELECT {U.MRN_SQL} AS id,
                   try_cast(mean_pap_usage AS DOUBLE)     AS mean_pap_usage,
                   try_cast(percentage_over_4h AS DOUBLE) AS pct_over_4h,
                   try_cast(total_days_usage AS DOUBLE)   AS total_days_usage,
                   try_cast(day_hit AS DOUBLE)            AS day_hit,
                   try_cast(adherence_flag AS DOUBLE)     AS adherence_flag,
                   coalesce(try_cast(setup_date AS DATE),
                            try_strptime(setup_date, '%m/%d/%Y')::DATE,
                            try_cast(SETUP_DTM AS TIMESTAMP)::DATE) AS setup_date
            FROM read_csv('{PC}', {PC_READ})
            WHERE MRN IS NOT NULL
          ),
          coh AS (SELECT id FROM read_parquet('{U.COHORT}'))
          SELECT id,
                 max(mean_pap_usage)   AS mean_pap_usage,
                 max(pct_over_4h)      AS pct_over_4h,
                 max(total_days_usage) AS total_days_usage,
                 max(day_hit)          AS day_hit,
                 max(adherence_flag)   AS adherence_flag,
                 min(setup_date)       AS setup_date   -- earliest CPAP setup ("received CPAP" signal)
          FROM raw WHERE id IN (SELECT id FROM coh)
          GROUP BY id
          -- keep anyone with usage OR a device-setup signal (setup-but-no-usage = the "unknown" arm)
          HAVING max(mean_pap_usage) IS NOT NULL OR max(pct_over_4h) IS NOT NULL
                 OR max(total_days_usage) IS NOT NULL OR min(setup_date) IS NOT NULL
        ) TO '{OUT}' (FORMAT PARQUET)
    """)

    c = pd.read_parquet(OUT)
    n_cohort = con.sql(f"SELECT count(*) FROM read_parquet('{U.COHORT}')").fetchone()[0]
    print(f"\ncohort patients: {n_cohort:,}")
    print(f"patients with a PAP/device record: {len(c):,} ({100*len(c)/n_cohort:.1f}%)")
    n_setup = c["setup_date"].notna().sum()
    n_usage = c["mean_pap_usage"].notna().sum()
    print(f"  with a setup date (received CPAP): {n_setup:,} ({100*n_setup/n_cohort:.1f}% of cohort)")
    print(f"  with usage data: {n_usage:,}; setup-but-no-usage (unknown arm): {int((c['setup_date'].notna() & c['mean_pap_usage'].isna()).sum()):,}")

    mp = c["mean_pap_usage"].dropna()
    print(f"\nmean_pap_usage (n={len(mp):,}) — UNIT CHECK:")
    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        print(f"   p{int(q*100):02d} = {mp.quantile(q):.1f}")
    print(f"   min/max = {mp.min():.1f} / {mp.max():.1f}")
    med = mp.median()
    unit = "MINUTES/night" if med > 30 else "HOURS/night"
    print(f"   -> median {med:.0f} implies units are ~{unit}")

    # sanity cross-check: percentage_over_4h vs mean_pap_usage (a robust unit-independent adherence)
    p4 = c["pct_over_4h"].dropna()
    if len(p4):
        print(f"\npercentage_over_4h (n={len(p4):,}): median {p4.median():.2f}, "
              f"range {p4.min():.2f}-{p4.max():.2f} (fraction of nights >=4h)")
    af = c["adherence_flag"].dropna()
    if len(af):
        print(f"adherence_flag: {100*af.mean():.1f}% flagged adherent (n={len(af):,})")

    U.log_milestone(f"Step 08 CPAP: per-MRN PAP-usage block for {len(c):,}/{n_cohort:,} cohort patients "
                    f"({100*len(c)/n_cohort:.1f}%); mean_pap_usage median {med:.0f} ({unit}).")


if __name__ == "__main__":
    main()
