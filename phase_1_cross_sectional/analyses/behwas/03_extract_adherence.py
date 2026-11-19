# %% [markdown]
# Step 03 - Extract the pre-computed medication adherence table (medication_adherence.csv).
#
# Grain: one row per patient x drug-class (GPI) adherence episode, with a per-episode PDC
# (`adherence`, 0-1) and its `days_supply`, anchored by `med_start_date`. This is the all-class
# PDC table the user chose for now (median PDC ~0.99); step 07 separately compares it against a
# raw-medication.csv-derived PDC to decide whether to switch source.
#
# We normalize MRN, parse the (mixed-format) med_start_date, and keep the columns needed to
# window and aggregate per patient in step 04. Small file (~23 MB) -> pandas.
#
# Emits features/adherence.parquet : id, gpi, med_start_date, adherence, days_supply
# DE-IDENTIFIED summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import behwas_utils as U

USECOLS = ["MRN", "GPI", "med_start_date", "adherence", "days_supply"]


def main() -> None:
    U.ensure_dirs()
    print(f"Reading {U.ADH_CSV} ...", flush=True)
    df = pd.read_csv(U.ADH_CSV, usecols=USECOLS, dtype=str)
    n_raw = len(df)

    df["id"] = U.normalize_mrn(df["MRN"])
    df["gpi"] = df["GPI"].astype("string").str.strip()
    df["med_start_date"] = U.parse_sas_datetime(df["med_start_date"])
    df["adherence"] = pd.to_numeric(df["adherence"], errors="coerce")
    df["days_supply"] = pd.to_numeric(df["days_supply"], errors="coerce")

    out = (df.dropna(subset=["id", "med_start_date", "adherence"])
             [["id", "gpi", "med_start_date", "adherence", "days_supply"]]
             .reset_index(drop=True))
    out_path = U.FEATURES_DIR / "adherence.parquet"
    out.to_parquet(out_path, index=False)

    # ---- DE-IDENTIFIED summary --------------------------------------------------
    n_pat = out["id"].nunique()
    print("\n==== Step 03 adherence summary ====")
    print(f"raw rows:                    {n_raw:,}")
    print(f"usable episodes:             {len(out):,}  ({n_pat:,} distinct patients, "
          f"{out['gpi'].nunique()} GPI classes)")
    print(f"adherence (PDC) min/median/max: {out['adherence'].min():.3f} / "
          f"{out['adherence'].median():.3f} / {out['adherence'].max():.3f}")
    print(f"days_supply median: {np.nanmedian(out['days_supply']):.0f}")
    print(f"med_start_date range: {out['med_start_date'].min()} .. {out['med_start_date'].max()}")
    print(f"Wrote {out_path}")

    U.log_milestone(
        f"Step 03 adherence: {n_raw:,} rows -> {len(out):,} episodes for {n_pat:,} patients "
        f"(pre-computed all-class PDC).")


if __name__ == "__main__":
    main()
