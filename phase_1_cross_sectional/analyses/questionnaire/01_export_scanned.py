# %% [markdown]
# Step 1 - Export the MASTER workbook's `Scanned` sheet as its own dataset.
#
# `Scanned` is a standalone wide dataset: 7,705 digitized paper-survey patients,
# one column per questionnaire item under friendly names that line up with the
# 54-column target codes. Per the plan it is a standalone deliverable this round
# (NOT merged into the unified target).
#
# Sheet layout (1-indexed): row 1 = source marker (Somnowre/KPHC), row 2 = KPHC
# QUEST_IDs, row 3 = friendly column names, rows 4+ = patient data.
#
# Emits:
#   results/questionnaire_results/scanned_dataset.csv        (faithful, all cols)
#   results/questionnaire_results/scanned_crosswalk_seed.csv (quest_id,friendly,code)
#
# Data-safety: `scanned_dataset.csv` contains DOB/Name/MRN BY REQUEST (faithful
# export). It is written straight from the sheet; NO row values are printed here.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import questionnaire_utils as U


def main() -> None:
    U.ensure_dirs()

    # ---- faithful dataset: friendly-name header (row 3), data rows 4+ -------
    # dtype=str preserves MRN/DOB leading zeros and exact answer strings.
    df = pd.read_excel(U.MASTER_XLSX, sheet_name="Scanned", header=2, dtype=str)
    # Drop fully-empty trailing columns/rows openpyxl sometimes pads.
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    out = U.RESULTS / "scanned_dataset.csv"
    U.safe_to_csv(df, out)
    print(f"[scanned_dataset] {df.shape[0]:,} rows x {df.shape[1]} cols -> {out.name}")

    # ---- crosswalk seed: quest_id <-> friendly <-> target code -------------
    seed = U.load_scanned_seed()
    seed_out = U.RESULTS / "scanned_crosswalk_seed.csv"
    U.safe_to_csv(seed, seed_out)

    coded = seed.dropna(subset=["code"])
    covered = sorted(set(coded["code"]) & set(U.TARGET_CODES))
    missing = sorted(set(U.TARGET_CODES) - set(coded["code"]))
    print(f"[seed] {len(seed)} quest_id<->friendly pairs; "
          f"{len(coded)} carry a code -> {seed_out.name}")
    print(f"[seed] target codes covered by Scanned: {len(covered)}/{len(U.TARGET_CODES)}")
    if missing:
        print(f"[seed] target codes NOT in Scanned ({len(missing)}): {missing}")

    U.log_milestone(
        f"01_export_scanned: wrote scanned_dataset.csv "
        f"({df.shape[0]} x {df.shape[1]}) + crosswalk seed "
        f"({len(covered)}/{len(U.TARGET_CODES)} target codes covered)."
    )


if __name__ == "__main__":
    main()
