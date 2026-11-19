# %% [markdown]
# Step 7 - Unify KPHC + Somnoware + Scanned into the final target frame.
#
# One row per encounter (KPHC encounter / Somnoware form instance / Scanned
# survey). CONTACT_DATE is normalized to a date. Cross-source collisions on
# (MRN, CONTACT_DATE) are resolved by precedence KPHC > Somnoware > Scanned
# (the higher-precedence row wins wholesale; KPHC forms are complete). Within a
# source, all distinct encounters are kept.
#
# Emits: results/questionnaire_results/questionnaire_wide.{parquet,csv}

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import questionnaire_utils as U


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(U.RESULTS / name)
    df["CONTACT_DATE"] = pd.to_datetime(df["CONTACT_DATE"], errors="coerce").dt.normalize()
    df["STUDY_DATE"] = pd.to_datetime(df["STUDY_DATE"], errors="coerce").dt.normalize()
    return df


def drop_collisions(df: pd.DataFrame, keyset: set) -> pd.DataFrame:
    """Drop rows whose (MRN, CONTACT_DATE) is already claimed by a higher source
    (only for non-null dates)."""
    hit = pd.Series(((m, c) in keyset for m, c in zip(df["MRN"], df["CONTACT_DATE"])),
                    index=df.index)
    return df[~(df["CONTACT_DATE"].notna() & hit)]


def keys_of(df: pd.DataFrame) -> set:
    sub = df[df["CONTACT_DATE"].notna()]
    return set(zip(sub["MRN"], sub["CONTACT_DATE"]))


def main() -> None:
    kphc = load("kphc_wide.parquet")
    somno = load("somnoware_wide.parquet")
    scanned = load("scanned_wide.parquet")

    k_keys = keys_of(kphc)
    somno = drop_collisions(somno, k_keys)
    ks_keys = k_keys | keys_of(somno)
    scanned = drop_collisions(scanned, ks_keys)

    unified = pd.concat([kphc, somno, scanned], ignore_index=True)

    # derive binned BEDTIME/WAKETIME labels from the continuous minutes
    for base, cont in [("BEDTIME", "BEDTIME_CONT"), ("WAKETIME", "WAKETIME_CONT")]:
        unified[base] = pd.to_numeric(unified[cont], errors="coerce").map(U.bin_from_minutes)
    # one-hot the CC1 chief complaint into CC_<category> binaries
    cc = unified["CC1"].map(U.cc_onehot).apply(pd.Series)
    for c in U.CC_COLUMNS:
        unified[c] = cc[c].astype("Int64")
    # re-assert nullable dtypes
    for code in U.TARGET_CODES:
        unified[code] = unified[code].astype(U.code_dtype(code))
    unified = unified[U.FULL_COLUMNS + ["source"]]

    pq_path = U.RESULTS / "questionnaire_wide.parquet"
    unified.to_parquet(pq_path, index=False)
    U.safe_to_csv(unified, U.RESULTS / "questionnaire_wide.csv")

    by_src = unified["source"].value_counts().to_dict()
    print(f"[unify] {len(unified):,} encounters x {len(U.TARGET_COLUMNS)} cols -> {pq_path.name}")
    print(f"[unify] by source: {by_src}")
    print(f"[unify] distinct MRNs: {unified['MRN'].nunique():,}")
    U.log_milestone(
        f"07_unify: {len(unified)} encounters ({by_src}); "
        f"{unified['MRN'].nunique()} distinct MRNs -> questionnaire_wide.")


if __name__ == "__main__":
    main()
