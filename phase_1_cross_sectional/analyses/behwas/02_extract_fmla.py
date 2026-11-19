# %% [markdown]
# Step 02 - Extract structured FMLA fields from fmla_380.csv (the WH-380 certification file).
#
# Grain in source: one row per WH-380 form line (multiple per patient / encounter). We:
#   - read ONLY the six approved structured columns (+ FMLA_VERSION, PAT_ENC_CSN_ID for the
#     filter/dedupe) via an explicit allow-list, so the PHI-heavy free-text / identifier
#     columns (names, employer, address, JOB_FUNCTIONS, ...) are NEVER loaded;
#   - read as latin-1 (the file is cp1252, not utf-8);
#   - filter FMLA_VERSION == 1 (v2 rows are metadata-only stubs with empty clinical fields);
#   - normalize MRN -> 12-char id; parse CONTACT_DATE (SAS datetime);
#   - standardize PROB_COND_DUR to days via PROB_COND_DUR_UNITS (1=days,2=weeks,3=months);
#   - dedupe to one record per leave request (id + PAT_ENC_CSN_ID).
#
# Emits features/fmla.parquet : id, contact_date, prob_cond_dur_days, admitted, pregnancy
# DE-IDENTIFIED summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import behwas_utils as U

USECOLS = ["MRN", "CONTACT_DATE", "PROB_COND_DUR", "PROB_COND_DUR_UNITS",
           "ADMITTED", "PREGNANCY", "FMLA_VERSION", "PAT_ENC_CSN_ID"]


def main() -> None:
    U.ensure_dirs()
    print(f"Reading {U.FMLA_CSV} (latin-1, {len(USECOLS)} approved columns only)...", flush=True)
    df = pd.read_csv(U.FMLA_CSV, encoding="latin-1", usecols=USECOLS,
                     dtype=str, on_bad_lines="skip")
    n_raw = len(df)

    # filter to fully-extracted forms (v1); v2 are empty clinical stubs
    ver = pd.to_numeric(df["FMLA_VERSION"], errors="coerce")
    df = df[ver == 1].copy()
    n_v1 = len(df)

    df["id"] = U.normalize_mrn(df["MRN"])
    df["contact_date"] = U.parse_sas_datetime(df["CONTACT_DATE"])

    dur = pd.to_numeric(df["PROB_COND_DUR"], errors="coerce")
    unit = pd.to_numeric(df["PROB_COND_DUR_UNITS"], errors="coerce")
    factor = unit.map(U.DUR_UNIT_DAYS)  # 1->1, 2->7, 3->30; other/NaN -> NaN
    df["prob_cond_dur_days"] = dur * factor

    df["admitted"] = (pd.to_numeric(df["ADMITTED"], errors="coerce") == 1).astype("Int64")
    df["pregnancy"] = (pd.to_numeric(df["PREGNANCY"], errors="coerce") == 1).astype("Int64")

    # one record per leave request (id + encounter CSN); keep the most-informative row
    df = df.dropna(subset=["id"])
    df = (df.sort_values(["id", "PAT_ENC_CSN_ID", "prob_cond_dur_days"])
            .drop_duplicates(subset=["id", "PAT_ENC_CSN_ID"], keep="last"))

    out = df[["id", "contact_date", "prob_cond_dur_days", "admitted", "pregnancy"]].reset_index(drop=True)
    out_path = U.FEATURES_DIR / "fmla.parquet"
    out.to_parquet(out_path, index=False)

    # ---- DE-IDENTIFIED summary --------------------------------------------------
    n_pat = out["id"].nunique()
    dur_ok = out["prob_cond_dur_days"].notna()
    print("\n==== Step 02 FMLA summary ====")
    print(f"raw form-line rows:            {n_raw:,}")
    print(f"FMLA_VERSION==1 rows:          {n_v1:,}")
    print(f"leave records after dedupe:    {len(out):,}  ({n_pat:,} distinct patients)")
    print(f"records with a usable duration: {int(dur_ok.sum()):,} "
          f"({100*dur_ok.mean():.1f}%); median {np.nanmedian(out['prob_cond_dur_days']):.0f} days")
    print(f"admitted=1: {int((out['admitted'] == 1).sum()):,}; "
          f"pregnancy=1: {int((out['pregnancy'] == 1).sum()):,}")
    print(f"contact_date range: {out['contact_date'].min()} .. {out['contact_date'].max()}")
    print(f"Wrote {out_path}")

    U.log_milestone(
        f"Step 02 FMLA: {n_raw:,} form-lines -> {len(out):,} deduped v1 leave records "
        f"for {n_pat:,} patients (duration standardized to days).")


if __name__ == "__main__":
    main()
