# %% [markdown]
# Step 00 - Collapse the insurance coverage-span table to a per-patient coverage-end date.
#
# D:/sas_data/ins.sas7bdat is a coverage-span table: ~3.88M rows, ~8.6 spans per patient
# (MRN, STRT, STOP, payer flags). We take the LATEST coverage-end per patient (max STOP) as the
# administrative disenrollment boundary for the incidence follow-up. The value 4000-12-30 is an
# open-ended "still enrolled" sentinel; that plus any forward-dated STOP (>= the diagnosis cutoff)
# is NON-BINDING -- those patients are censored by the 2023-05-31 data cutoff instead.
#
# read_sas is used because DuckDB cannot read sas7bdat (pyreadstat is absent). Everything stays in
# millisecond datetime resolution so the 4000-12-30 sentinel does not overflow ns timestamps, and
# only the *binding* (pre-cutoff) coverage end is written downstream.
#
# Emits results/incwas_results/followup/insurance_coverage.parquet. DE-IDENTIFIED summary to stdout.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import incwas_utils as U

CHUNK = 500_000
CUTOFF_MS = np.datetime64(U.ADATE_CUTOFF, "ms")  # compare in ms to avoid ns overflow on the sentinel


def main() -> None:
    U.ensure_dirs()

    # cohort id set -- only need coverage for the analytic spine
    cohort_ids = set(pd.read_parquet(U.COHORT, columns=["id"])["id"].astype(str))
    print(f"cohort ids: {len(cohort_ids):,}")

    parts = []
    n_rows = 0
    it = pd.read_sas(U.INS_SAS, format="sas7bdat", chunksize=CHUNK, encoding="latin-1")
    for chunk in it:
        n_rows += len(chunk)
        ids = U.normalize_mrn(chunk["MRN"])
        stop = pd.to_datetime(chunk["STOP"], errors="coerce").astype("datetime64[ms]")
        sub = pd.DataFrame({"id": ids, "STOP": stop})
        sub = sub[sub["id"].isin(cohort_ids) & sub["STOP"].notna()]
        if len(sub):
            parts.append(sub.groupby("id", as_index=False)["STOP"].max())
    it.close()
    print(f"ins rows scanned: {n_rows:,}")

    allsp = pd.concat(parts, ignore_index=True)
    cov = allsp.groupby("id", as_index=False)["STOP"].max().rename(columns={"STOP": "coverage_end"})
    cov["coverage_end"] = cov["coverage_end"].astype("datetime64[ms]")

    # binding only if coverage truly ends before the diagnosis cutoff (else sentinel/forward-dated)
    binding = cov["coverage_end"].to_numpy() < CUTOFF_MS
    cov["active"] = ~binding                                   # coverage extends to/through the cutoff
    cov["coverage_end_binding"] = cov["coverage_end"].where(binding)

    cov.to_parquet(U.INS_COVERAGE, index=False)

    # ---- de-identified summary ---------------------------------------------
    n = len(cov)
    n_cohort = len(cohort_ids)
    n_bind = int(binding.sum())
    print(f"patients with coverage record: {n:,}/{n_cohort:,} ({100*n/n_cohort:.1f}%)")
    print(f"  active through cutoff (non-binding): {n-n_bind:,} ({100*(n-n_bind)/n:.1f}%)")
    print(f"  disenrolled before {U.ADATE_CUTOFF} (binding censor): {n_bind:,} ({100*n_bind/n:.1f}%)")
    if n_bind:
        be = cov.loc[binding, "coverage_end"]
        print(f"  binding coverage_end range: {be.min().date()} .. {be.max().date()}, "
              f"median {be.median().date()}")
    print(f"cohort patients with NO coverage record: {n_cohort-n:,} "
          f"(will be censored at cutoff/death only)")

    U.log_milestone(
        f"Step 00 insurance: collapsed {n_rows:,} coverage spans -> per-patient coverage_end for "
        f"{n:,}/{n_cohort:,} cohort patients; {n_bind:,} disenrolled before {U.ADATE_CUTOFF} (binding).")


if __name__ == "__main__":
    main()
