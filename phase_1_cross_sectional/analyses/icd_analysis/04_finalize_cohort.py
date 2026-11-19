# %% [markdown]
# Step 4 - Finalize the analysis cohort: merge cohort_base + BMI + n_encounters,
# then COMPLETE-CASE on the model covariates (age, sex, BMI, race, n_encounters) so
# every phecode regression uses an identical subject set. Writes cohort/cohort.parquet.
# DE-IDENTIFIED summary only.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import utils as U

BASE = U.COHORT_DIR / "cohort_base.parquet"
BMI = U.COHORT_DIR / "bmi.parquet"
ENC = U.COHORT_DIR / "encounters.parquet"
OUT = U.COHORT_DIR / "cohort.parquet"
SUMMARY = U.COHORT_DIR / "cohort_final_summary.md"


def main() -> None:
    base = pd.read_parquet(BASE)
    bmi = pd.read_parquet(BMI)[["id", "bmi"]]
    enc = pd.read_parquet(ENC)[["id", "n_encounters"]]

    df = base.merge(bmi, on="id", how="left").merge(enc, on="id", how="left")
    n0 = len(df)

    # complete-case mask (race is never null -> 'Unknown'); n_encounters present => >=1 pre-study dx
    have_age = df["age"].notna()
    have_sex = df["sex_M"].notna()
    have_bmi = df["bmi"].notna()
    have_enc = df["n_encounters"].notna()
    keep = have_age & have_sex & have_bmi & have_enc
    final = df[keep].copy()
    final["n_encounters"] = final["n_encounters"].astype(int)

    final.to_parquet(OUT, index=False)

    # ---- DE-IDENTIFIED summary ----
    lines = []
    def p(m=""):
        print(m); lines.append(str(m))

    p("# Final analysis cohort (Step 4)\n")
    p(f"- merged cohort: {n0:,}")
    p(f"- dropped: no age {int((~have_age).sum()):,}; no sex {int((~have_sex).sum()):,}; "
      f"no BMI {int((~have_bmi).sum()):,}; no pre-study dx/encounters {int((~have_enc).sum()):,}")
    p(f"- **final complete-case cohort: {len(final):,}** ({100*len(final)/n0:.1f}% of adults)")
    p("\n## Severity distribution (final)")
    sv = final["severity"].value_counts().reindex(["None", "Mild", "Moderate", "Severe"])
    for k, v in sv.items():
        p(f"  - {k:9s}: {int(v):,} ({100*v/len(final):.1f}%)")
    p("\n## Covariate summary (final cohort)")
    p(f"  - age: median {final['age'].median():.1f} (IQR {final['age'].quantile(.25):.0f}-{final['age'].quantile(.75):.0f})")
    p(f"  - sex_M: {100*final['sex_M'].mean():.1f}% male")
    p(f"  - BMI: median {final['bmi'].median():.1f} (IQR {final['bmi'].quantile(.25):.1f}-{final['bmi'].quantile(.75):.1f})")
    p(f"  - n_encounters: median {final['n_encounters'].median():.0f} "
      f"(p90 {final['n_encounters'].quantile(.9):.0f}, max {final['n_encounters'].max():,})")
    p(f"  - race/ethnicity: {final['RACE_ETH'].value_counts().to_dict()}")
    p(f"  - AHI>=5 (any OSA): {100*(final['AHI']>=5).mean():.1f}%; AHI>=15: {100*(final['AHI']>=15).mean():.1f}%")
    p(f"\nWrote cohort.parquet ({len(final):,} rows) -> {OUT}")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    U.log_milestone(f"Step 4 final cohort: {len(final):,} complete-case adults.")


if __name__ == "__main__":
    main()
